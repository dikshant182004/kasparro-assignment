"""Replay adapter: returns cached JSON for known (response_id, prompt_hash, model).

If no cached file exists for the given response_id+prompt_hash+model,
returns a SKIP marker (skipped: no_recorded_output) — NEVER silently
falls back to the stub. The previous behaviour labelled stub output as
'replay-v1', which was misleading.

Cache files are keyed by (response_id, prompt_hash, model_name) so
stale caches can't be served across prompt or model changes.
"""
from __future__ import annotations

import json
from pathlib import Path

from kasparro_auditor.config import get_settings
from kasparro_auditor.db.models import ErrorCode
from kasparro_auditor.llm.adapter import (
    LLMContext,
    LLMProviderError,
    MalformedLLMOutput,
    Judgment,
    prompt_hash,
    verify_evidence,
)


class ReplaySkip(Exception):
    """Raised when no cached output exists for this (response_id, prompt_hash, model).

    The caller should record this as skipped=True with
    error_code=NO_RECORDED_OUTPUT, NOT as a failed row.
    """

    def __init__(self, response_id: str, reason: str = "no_recorded_output") -> None:
        self.response_id = response_id
        self.reason = reason
        super().__init__(
            f"No cached output for response_id={response_id} "
            f"(prompt_hash={prompt_hash()}, model={_get_model_name()}). "
            "Run with EXECUTION_MODE=live or local to populate the cache."
        )


def _get_model_name() -> str:
    """The model name we'd expect to find in the cache."""
    settings = get_settings()
    # In replay mode we look for caches from EITHER the live Groq model
    # or the local Ollama model — whichever was used last.
    return settings.groq_model


class ReplayAdapter:
    """Returns cached LLM output for known (response_id, prompt_hash, model).

    mode='replay' — but each result row's execution_mode is set per-call
    based on what the cached file actually says (so a cached 'live' result
    is labelled 'live' even when replayed).
    """

    mode = "replay"
    model_name = "replay"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.cache_dir: Path = self.settings.replay_cache_dir

    async def classify(self, ctx: LLMContext):
        p_hash = prompt_hash()
        # Search for any cache file matching this response_id + prompt_hash
        # (model-agnostic — we accept caches from any model that was used).
        candidates = list(
            self.cache_dir.glob(f"{ctx.response_id}__{p_hash}__*.json")
        )
        if not candidates:
            raise ReplaySkip(ctx.response_id)

        # Pick the most recent matching cache file
        cache_path = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            result = Judgment.model_validate(payload["judgment"])
        except (json.JSONDecodeError, KeyError, ValueError) as err:
            raise MalformedLLMOutput(
                f"Cached replay file {cache_path.name} is corrupted: {err}",
                response_id=ctx.response_id,
            ) from err

        # Re-verify the cached evidence — defense in depth.
        try:
            verify_evidence(result, ctx.answer_text, response_id=ctx.response_id)
        except Exception:
            # Cached excerpt drifted from input (e.g. pack was re-imported
            # with a different answer_text). Treat as a skip, not a failure.
            raise ReplaySkip(ctx.response_id, reason="cache_stale")

        # Attach metadata so the caller can record the real model + mode
        result._model_used = payload.get("model", "replay")  # type: ignore[attr-defined]
        result._attempts = payload.get("attempts", 1)  # type: ignore[attr-defined]
        # Carry the original execution mode (live/local) so the result row
        # reflects where the judgment actually came from.
        result._actual_mode = payload.get("model", "").startswith("gpt-oss") and "live" or "local"  # type: ignore[attr-defined]
        return result
