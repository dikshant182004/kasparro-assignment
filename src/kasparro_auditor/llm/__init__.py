"""LLM adapter package.

One Protocol, four implementations:
  - GroqAdapter   : live Groq calls (openai/gpt-oss-120b) via Instructor
  - OllamaAdapter : local Ollama endpoint (OpenAI-compatible) for approved local models
  - ReplayAdapter : returns cached JSON; raises ReplaySkip on cache miss
  - StubAdapter   : deterministic regex-based stub used in tests

Adapters return only Judgment fields. Facts (mention, competitors_known,
citation counts) are computed deterministically in scoring/facts.py.
"""
from kasparro_auditor.llm.adapter import (
    EvidenceUnsupported,
    Judgment,
    LLMAdapter,
    LLMContext,
    LLMError,
    LLMProviderError,
    LLMTimeout,
    MalformedLLMOutput,
    classify_response,
    prompt_hash,
    verify_evidence,
)
from kasparro_auditor.llm.factory import build_adapter, preflight
from kasparro_auditor.llm.groq_adapter import GroqAdapter
from kasparro_auditor.llm.ollama_adapter import OllamaAdapter
from kasparro_auditor.llm.replay_adapter import ReplayAdapter, ReplaySkip
from kasparro_auditor.llm.stub_adapter import StubAdapter

__all__ = [
    "EvidenceUnsupported",
    "Judgment",
    "LLMAdapter",
    "LLMContext",
    "LLMError",
    "LLMProviderError",
    "LLMTimeout",
    "MalformedLLMOutput",
    "classify_response",
    "prompt_hash",
    "verify_evidence",
    "build_adapter",
    "preflight",
    "GroqAdapter",
    "OllamaAdapter",
    "ReplayAdapter",
    "ReplaySkip",
    "StubAdapter",
]
