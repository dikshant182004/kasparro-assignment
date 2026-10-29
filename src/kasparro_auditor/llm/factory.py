"""Factory: pick the right adapter for the configured execution mode.

Includes a `preflight()` function that should be called at startup in
live/local mode to fail loudly if the configured model is unavailable,
instead of failing 150 times quietly.
"""
from __future__ import annotations

import asyncio

from kasparro_auditor.config import ExecutionMode, get_settings
from kasparro_auditor.db.models import ErrorCode
from kasparro_auditor.llm.adapter import LLMAdapter, LLMProviderError


def build_adapter(mode: ExecutionMode | None = None) -> LLMAdapter:
    """Return the adapter the pipeline should use.

    If `mode` is None, the configured EXECUTION_MODE is used.
    Live mode with no API key falls back to replay with a logged warning
    (does NOT silently switch — the run label is recorded as 'replay').
    """
    mode = mode or get_settings().execution_mode
    if mode == ExecutionMode.LIVE:
        from kasparro_auditor.llm.groq_adapter import GroqAdapter

        try:
            return GroqAdapter()
        except LLMProviderError:
            # Live mode requested but no key — raise so the caller can
            # decide to switch to replay or fail. NEVER silently switch.
            raise
    if mode == ExecutionMode.LOCAL:
        from kasparro_auditor.llm.ollama_adapter import OllamaAdapter

        return OllamaAdapter()
    if mode == ExecutionMode.REPLAY:
        from kasparro_auditor.llm.replay_adapter import ReplayAdapter

        return ReplayAdapter()
    if mode == ExecutionMode.STUB:
        from kasparro_auditor.llm.stub_adapter import StubAdapter

        return StubAdapter()
    # Default: replay (safe, offline, deterministic if cache exists)
    from kasparro_auditor.llm.replay_adapter import ReplayAdapter

    return ReplayAdapter()


async def preflight(adapter) -> None:
    """Make one trivial call at startup to fail loudly if the model is broken.

    Only call this in live/local mode. Stub/replay modes don't need it.
    """
    from kasparro_auditor.llm.adapter import LLMContext

    # A trivial context that should classify cleanly
    ctx = LLMContext(
        response_id="preflight",
        prompt_id="preflight",
        engine="chatgpt",
        prompt_text="does kasparro work?",
        answer_text="Kasparro is an AI visibility tool.",
        mention=True,
        prompt_names_brand=True,
        competitors_known=[],
    )
    try:
        await asyncio.wait_for(adapter.classify(ctx), timeout=60.0)
    except Exception as err:
        raise LLMProviderError(
            f"Preflight check failed for model {getattr(adapter, 'model_name', '?')}: {err}. "
            "Fix configuration before running the pipeline.",
            error_code=ErrorCode.PROVIDER_ERROR,
        ) from err
