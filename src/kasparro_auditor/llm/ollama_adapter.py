"""Ollama adapter — approved local model access.

The brief requires "a local model or hosted access explicitly provided or
approved by the hiring team". A personal Groq key is NOT that by default.
Ollama (https://ollama.com) is an OpenAI-compatible local endpoint.

Usage:
  1. Install Ollama: https://ollama.com/download
  2. Pull a model: `ollama pull llama3.1:8b`
  3. Set EXECUTION_MODE=local in .env
  4. Set OLLAMA_MODEL=llama3.1:8b (or another model you've pulled)

Instructor supports OpenAI-compatible endpoints with a custom base_url,
so this adapter reuses the same Pydantic schema validation.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import instructor
from openai import AsyncOpenAI

try:
    from instructor.core import InstructorRetryException
except ImportError:  # backwards-compat
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


class OllamaAdapter:
    """Local Ollama adapter. OpenAI-compatible endpoint via Instructor.

    No API key required (Ollama doesn't need one). Default base_url is
    http://localhost:11434/v1.
    """

    mode = "local"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = instructor.from_openai(
            AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key=self.settings.ollama_api_key,  # placeholder, Ollama ignores it
            )
        )
        self.model_name = self.settings.ollama_model
        self._cache_dir = self.settings.replay_cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def classify(self, ctx: LLMContext):
        from kasparro_auditor.llm.adapter import Judgment as _J

        user_prompt = build_user_prompt(ctx)
        started = time.perf_counter()

        # Ollama tends to be slower than Groq; give it 60s timeout per call.
        timeout = max(self.settings.groq_timeout_seconds, 60.0)
        try:
            result = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.settings.ollama_model,
                    response_model=_J,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.settings.groq_temperature,
                    max_tokens=self.settings.groq_max_output_tokens,
                ),
                timeout=timeout + 5,
            )
            verify_evidence(result, ctx.answer_text, response_id=ctx.response_id)

            latency_ms = int((time.perf_counter() - started) * 1000)
            await self._cache_result(ctx, result, latency_ms)
            result._model_used = self.model_name  # type: ignore[attr-defined]
            result._attempts = 1  # type: ignore[attr-defined]
            return result

        except asyncio.TimeoutError as err:
            raise LLMTimeout(
                f"Ollama {self.model_name} timed out after {timeout}s",
                response_id=ctx.response_id,
            ) from err
        except InstructorRetryException as err:
            raise MalformedLLMOutput(
                f"Ollama returned non-conforming output: {err}",
                response_id=ctx.response_id,
            ) from err
        except Exception as err:
            raise LLMProviderError(
                f"Ollama call failed: {err}. Is `ollama serve` running and "
                f"is model `{self.model_name}` pulled?",
                response_id=ctx.response_id,
            ) from err

    async def _cache_result(self, ctx: LLMContext, result, latency_ms: int) -> None:
        p_hash = prompt_hash()
        cache_path = self._cache_dir / f"{ctx.response_id}__{p_hash}__{self.model_name}.json"
        payload = {
            "response_id": ctx.response_id,
            "prompt_hash": p_hash,
            "model": self.model_name,
            "attempts": 1,
            "latency_ms": latency_ms,
            "cached_at": time.time(),
            "judgment": result.model_dump(),
        }
        cache_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
