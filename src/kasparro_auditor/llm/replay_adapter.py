"""Replay adapter: returns cached JSON for known response_id, else falls back to stub.

Reviewer mode: runs the entire pipeline offline with no API key.
Per the brief: "Replay supports mechanics and failure testing; it does
not establish live-model correctness, variability, cost or latency."
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from kasparro_auditor.config import get_settings
from kasparro_auditor.llm.adapter import LLMContext, LLMTimeout, verify_evidence
from kasparro_auditor.llm.stub_adapter import StubAdapter
from kasparro_auditor.schemas.results import ClassificationResult


class ReplayAdapter:
    """Returns cached LLM output for known response_ids."""

    mode = "replay"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.cache_dir: Path = self.settings.replay_cache_dir
        self._fallback = StubAdapter()
        self.model_name = "replay-v1"

    async def classify(self, ctx: LLMContext) -> ClassificationResult:
        cache_path = self.cache_dir / f"{ctx.response_id}.json"
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            result = ClassificationResult.model_validate(payload["result"])
            self.model_name = payload.get("model", "replay-v1")
            # Re-verify the cached evidence — defense in depth.
            try:
                verify_evidence(result, ctx.answer_text)
            except Exception:
                # Cached excerpt drifted from input (rare); fall back to stub.
                return await self._fallback.classify(ctx)
            return result

        # No cache for this response_id — use the deterministic stub.
        return await self._fallback.classify(ctx)
