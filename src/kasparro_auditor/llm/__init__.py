"""LLM adapter package.

One Protocol, three implementations:
  - GroqAdapter   : live Groq calls (llama-3.3-70b-versatile) via Instructor
  - ReplayAdapter : returns cached JSON for known response_id; falls back to stub
  - StubAdapter   : deterministic regex-based stub used in tests

All adapters return a `ClassificationResult` or raise `LLMError`. They
never silently manufacture a confident result on failure.
"""
from kasparro_auditor.llm.adapter import (
    LLMAdapter,
    LLMContext,
    LLMError,
    LLMTimeout,
    MalformedLLMOutput,
    classify_response,
    prompt_hash,
    verify_evidence,
)
from kasparro_auditor.llm.groq_adapter import GroqAdapter
from kasparro_auditor.llm.replay_adapter import ReplayAdapter
from kasparro_auditor.llm.stub_adapter import StubAdapter
from kasparro_auditor.llm.factory import build_adapter

__all__ = [
    "LLMAdapter",
    "LLMContext",
    "LLMError",
    "LLMTimeout",
    "MalformedLLMOutput",
    "classify_response",
    "prompt_hash",
    "verify_evidence",
    "GroqAdapter",
    "ReplayAdapter",
    "StubAdapter",
    "build_adapter",
]
