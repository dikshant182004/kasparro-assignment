"""Groq live adapter using Instructor + structured outputs.

IMPORTANT (Oct 2026): Groq retired llama-3.3-70b-versatile and
llama-3.1-8b-instant on 16 August 2026 (free + developer tiers).
Default primary is now `openai/gpt-oss-120b` (production GA, $0.15/M
input, $0.60/M output). Fallback is `openai/gpt-oss-20b`.

Free-tier rate limit: 30 RPM / 1,000 RPD / 8K TPM. With concurrency=2
and 150 calls, the run takes ~5 minutes minimum — right at the limit.
Add 429 retry with exponential backoff.

Per-call model name tracking: the returned Judgment carries the model
name actually used (not the adapter's default), so concurrent calls
that fall back don't get mislabelled.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import instructor
from groq import AsyncGroq

try:
    from instructor.core import InstructorRetryException
except ImportError:  # backwards-compat for older instructor versions
    from instructor.exceptions import InstructorRetryException

from kasparro_auditor.config import Settings, get_settings
from kasparro_auditor.db.models import ErrorCode
from kasparro_auditor.llm.adapter import (
    LLMContext,
    LLMProviderError,
    LLMTimeout,
    MalformedLLMOutput,
    SYSTEM_PROMPT,
    build_user_prompt,
    prompt_hash,
    verify_evidence,
)
from kasparro_auditor.schemas.results import Judgment  # noqa: F401 — re-export


# Cache file path keyed by (response_id, prompt_hash, model_name) so stale
# caches can't be served across prompt or model changes.
def _cache_path(cache_dir: Path, response_id: str, p_hash: str, model: str) -> Path:
    safe_model = model.replace("/", "_")
    return cache_dir / f"{response_id}__{p_hash}__{safe_model}.json"


class GroqAdapter:
    """Live Groq adapter. Caches each successful response for replay."""

    mode = "live"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.groq_api_key:
            # Missing API key is a provider/config error, NOT a timeout.
            raise LLMProviderError(
                "GROQ_API_KEY not set. Use EXECUTION_MODE=replay or stub for offline runs, "
                "or EXECUTION_MODE=local for an Ollama adapter.",
                error_code=ErrorCode.PROVIDER_ERROR,
            )
        self.client = instructor.from_groq(
            AsyncGroq(api_key=self.settings.groq_api_key),
        )
        self.model_name = self.settings.groq_model  # default; per-call returns override
        self._cache_dir = self.settings.replay_cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def classify(self, ctx: LLMContext):
        """Call Groq with structured outputs; cache the result on success.

        Returns the Judgment object. The caller is responsible for
        combining it with Facts to produce a ResponseResult row.
        """
        from kasparro_auditor.llm.adapter import Judgment as _J

        user_prompt = build_user_prompt(ctx)
        started = time.perf_counter()

        # Try the primary model, then the fallback. Each model gets up to
        # `groq_max_retries` attempts for transient errors (timeout/429/5xx).
        last_error: Exception | None = None
        for model_name in (self.settings.groq_model, self.settings.groq_fallback_model):
            for attempt in range(1, self.settings.groq_max_retries + 2):
                try:
                    result = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=model_name,
                            response_model=_J,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=self.settings.groq_temperature,
                            max_tokens=self.settings.groq_max_output_tokens,
                        ),
                        timeout=self.settings.groq_timeout_seconds,
                    )
                    # Verify evidence locally — never trust the LLM to vouch for itself.
                    verify_evidence(result, ctx.answer_text, response_id=ctx.response_id)

                    # Cache for replay mode — key includes prompt_hash + model_name
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    await self._cache_result(ctx, result, model_name, latency_ms, attempt)

                    # Attach the actual model used so the caller can record it per-call
                    result._model_used = model_name  # type: ignore[attr-defined]
                    result._attempts = attempt  # type: ignore[attr-defined]
                    return result

                except asyncio.TimeoutError as err:
                    last_error = LLMTimeout(
                        f"Groq {model_name} attempt {attempt} timed out after "
                        f"{self.settings.groq_timeout_seconds}s",
                        response_id=ctx.response_id,
                    )
                    # Exponential backoff before next attempt
                    await asyncio.sleep(min(2 ** attempt, 8))
                except InstructorRetryException as err:
                    # Instructor already retried internally; surface as malformed.
                    last_error = MalformedLLMOutput(
                        f"Groq {model_name} returned non-conforming output: {err}",
                        response_id=ctx.response_id,
                    )
                    break  # no point retrying the same model on schema failure
                except Exception as err:
                    # 429 / 5xx / network errors → retry with backoff
                    last_error = LLMProviderError(
                        f"Groq {model_name} attempt {attempt} failed: {err}",
                        response_id=ctx.response_id,
                    )
                    await asyncio.sleep(min(2 ** attempt, 8))

        # All attempts across both models failed
        if isinstance(last_error, (LLMTimeout, LLMProviderError, MalformedLLMOutput)):
            raise last_error
        raise LLMProviderError(
            f"All Groq attempts failed: {last_error}",
            response_id=ctx.response_id,
        )

    async def _cache_result(
        self, ctx: LLMContext, result, model_name: str, latency_ms: int, attempts: int
    ) -> None:
        p_hash = prompt_hash()
        cache_path = _cache_path(self._cache_dir, ctx.response_id, p_hash, model_name)
        payload = {
            "response_id": ctx.response_id,
            "prompt_hash": p_hash,
            "model": model_name,
            "attempts": attempts,
            "latency_ms": latency_ms,
            "cached_at": time.time(),
            "judgment": result.model_dump(),
        }
        cache_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
