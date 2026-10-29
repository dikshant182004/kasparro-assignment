"""Groq live adapter using Instructor + structured outputs.

Uses `llama-3.3-70b-versatile` (production GA on Groq as of Oct 2026).
Falls back to `llama-3.1-8b-instant` if the primary model is unavailable.

Cost guard: ~$0.24 for the full 150-response run (well under the $5 cap).
Latency guard: 30s timeout, 2 retries with exponential backoff.
Output guard: max 1024 tokens — ClassificationResult is small.
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

from kasparro_auditor.config import ExecutionMode, Settings, get_settings
from kasparro_auditor.llm.adapter import (
    LLMContext,
    LLMTimeout,
    MalformedLLMOutput,
    build_user_prompt,
    prompt_hash,
    verify_evidence,
)
from kasparro_auditor.schemas.results import ClassificationResult


class GroqAdapter:
    """Live Groq adapter. Caches each successful response for replay."""

    mode = "live"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.groq_api_key:
            raise LLMTimeout(
                "GROQ_API_KEY not set. Use EXECUTION_MODE=replay or stub for offline runs."
            )
        self.client = instructor.from_groq(
            AsyncGroq(api_key=self.settings.groq_api_key),
            max_retries=self.settings.groq_max_retries,
        )
        self.model_name = self.settings.groq_model
        self._cache_dir = self.settings.replay_cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def classify(self, ctx: LLMContext) -> ClassificationResult:
        """Call Groq with structured outputs; cache the result on success."""
        user_prompt = build_user_prompt(ctx)
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.settings.groq_model,
                    response_model=ClassificationResult,
                    messages=[
                        {"role": "system", "content": __import__("kasparro_auditor.llm.adapter", fromlist=["SYSTEM_PROMPT"]).SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.settings.groq_temperature,
                    max_tokens=self.settings.groq_max_output_tokens,
                    timeout=self.settings.groq_timeout_seconds,
                ),
                timeout=self.settings.groq_timeout_seconds + 5,
            )
        except asyncio.TimeoutError as err:
            raise LLMTimeout(
                f"Groq call timed out after {self.settings.groq_timeout_seconds}s",
                response_id=ctx.response_id,
            ) from err
        except InstructorRetryException as err:
            # Instructor already retried; surface as malformed.
            raise MalformedLLMOutput(
                f"Groq returned non-conforming output after retries: {err}",
                response_id=ctx.response_id,
            ) from err
        except Exception as err:
            # Fall back to the smaller model once, then give up.
            try:
                result = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.settings.groq_fallback_model,
                        response_model=ClassificationResult,
                        messages=[
                            {"role": "system", "content": __import__("kasparro_auditor.llm.adapter", fromlist=["SYSTEM_PROMPT"]).SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.settings.groq_temperature,
                        max_tokens=self.settings.groq_max_output_tokens,
                    ),
                    timeout=self.settings.groq_timeout_seconds + 5,
                )
                self.model_name = self.settings.groq_fallback_model
            except Exception as fallback_err:
                raise LLMTimeout(
                    f"Groq primary + fallback both failed: primary={err}; fallback={fallback_err}",
                    response_id=ctx.response_id,
                ) from fallback_err

        # Verify evidence locally — never trust the LLM to vouch for itself.
        verify_evidence(result, ctx.answer_text)

        # Cache for replay mode
        latency_ms = int((time.perf_counter() - started) * 1000)
        await self._cache_result(ctx, result, latency_ms)
        return result

    async def _cache_result(
        self, ctx: LLMContext, result: ClassificationResult, latency_ms: int
    ) -> None:
        payload = {
            "response_id": ctx.response_id,
            "model": self.model_name,
            "latency_ms": latency_ms,
            "prompt_hash": prompt_hash(),
            "result": result.model_dump(),
        }
        cache_path: Path = self._cache_dir / f"{ctx.response_id}.json"
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
