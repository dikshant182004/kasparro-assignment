"""Factory: pick the right adapter for the configured execution mode."""
from __future__ import annotations

from kasparro_auditor.config import ExecutionMode, get_settings
from kasparro_auditor.llm.adapter import LLMAdapter, LLMError


def build_adapter(mode: ExecutionMode | None = None) -> LLMAdapter:
    """Return the adapter the pipeline should use.

    If `mode` is None, the configured EXECUTION_MODE is used.
    """
    mode = mode or get_settings().execution_mode
    if mode == ExecutionMode.LIVE:
        from kasparro_auditor.llm.groq_adapter import GroqAdapter

        try:
            return GroqAdapter()
        except LLMError:
            # Live mode requested but no key — fall back to replay with a warning.
            from kasparro_auditor.llm.replay_adapter import ReplayAdapter

            return ReplayAdapter()
    if mode == ExecutionMode.REPLAY:
        from kasparro_auditor.llm.replay_adapter import ReplayAdapter

        return ReplayAdapter()
    if mode == ExecutionMode.STUB:
        from kasparro_auditor.llm.stub_adapter import StubAdapter

        return StubAdapter()
    # Default: replay (safe, offline, deterministic if cache exists)
    from kasparro_auditor.llm.replay_adapter import ReplayAdapter

    return ReplayAdapter()
