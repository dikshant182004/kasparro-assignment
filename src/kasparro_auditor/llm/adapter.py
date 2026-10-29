"""LLM adapter Protocol + shared types + the canonical classify helper.

`classify_response` is the only entry point the rest of the codebase
calls. It builds the prompt, calls the adapter, validates the evidence
excerpt actually exists in `answer_text`, and returns a
`ClassificationResult` or raises `LLMError`/`MalformedLLMOutput`.

Evidence verification is NOT delegated to the LLM — it's a deterministic
check we run on every response. This prevents the unsupported-claim
failure mode the brief warns about.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kasparro_auditor.schemas.results import ClassificationResult


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base class for adapter failures. Carries the response_id when known."""

    def __init__(self, message: str, *, response_id: str | None = None) -> None:
        super().__init__(message)
        self.response_id = response_id


class LLMTimeout(LLMError):
    """Raised when all retries are exhausted."""


class MalformedLLMOutput(LLMError):
    """Raised when the LLM returns content that fails schema or evidence checks."""


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LLMContext:
    """Everything an adapter needs to classify one response."""

    response_id: str
    prompt_id: str
    engine: str
    prompt_text: str
    answer_text: str
    competitor_hints: list[str]


@runtime_checkable
class LLMAdapter(Protocol):
    """Adapter interface implemented by Groq/Replay/Stub."""

    mode: str
    model_name: str | None

    async def classify(self, ctx: LLMContext) -> ClassificationResult: ...


# ---------------------------------------------------------------------------
# Prompt building (single source of truth)
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
You are an AI-visibility auditor. You classify whether a brand named
"Kasparro" is mentioned, recommended, or negatively characterised inside
a single archived AI-engine answer.

Strict rules:
1. A passing mention is NOT a recommendation. A recommendation requires
   the answer to present Kasparro as a suited option for the question.
2. Negative characterisation includes criticism, scepticism, "hype",
   "gimmick", "scam", "not legit", or being dismissed.
3. `recommendation_position` is the 1-based rank ONLY when Kasparro is
   explicitly ranked or listed alongside alternatives. For a prose
   mention with no ordered list, return null.
4. `competitors_named` must be the lowercased names of explicitly named
   competing AI-visibility tools (profound, otterly ai, peec ai,
   athenahq, dageno ai, etc.). Use the supplied hints only as a guide —
   you may add others that appear in the text.
5. `evidence_excerpt` MUST be a verbatim substring of the answer text.
   `char_offset_start` and `char_offset_end` MUST mark its exact
   location. The grader will verify this substring exists.
6. If the answer does not name Kasparro, set mention=false and provide
   an evidence excerpt of any sentence that explains the answer's stance.
"""


def build_user_prompt(ctx: LLMContext) -> str:
    """Render the user message for a single classification call."""
    return (
        f"Classify the following archived AI-engine answer.\n\n"
        f"Prompt: {ctx.prompt_text}\n"
        f"Engine: {ctx.engine}\n"
        f"Known competitor hints: {', '.join(ctx.competitor_hints) or 'none'}\n\n"
        f"ANSWER TEXT (verbatim):\n\"\"\"\n{ctx.answer_text}\n\"\"\"\n\n"
        f"Return JSON matching the ClassificationResult schema. The "
        f"evidence_excerpt MUST be a verbatim substring of the answer text above."
    )


def prompt_hash() -> str:
    """Stable hash of the prompt template — recorded on every result."""
    h = hashlib.sha256()
    h.update(SYSTEM_PROMPT.encode("utf-8"))
    h.update(build_user_prompt.__qualname__.encode("utf-8"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Evidence verification (deterministic, NEVER the LLM's job)
# ---------------------------------------------------------------------------


def verify_evidence(result: ClassificationResult, answer_text: str) -> None:
    """Raise MalformedLLMOutput if the excerpt is not a verbatim substring.

    Also enforces the offset range matches the excerpt. This is the
    anti-unsupported-claim guardrail.
    """
    excerpt = result.evidence_excerpt
    if len(excerpt) < 10:
        raise MalformedLLMOutput(
            f"evidence_excerpt too short ({len(excerpt)} chars); need >=10"
        )
    start = answer_text.find(excerpt)
    if start == -1:
        # Try whitespace-tolerant match (LLMs sometimes trim edges)
        norm_text = " ".join(answer_text.split())
        norm_excerpt = " ".join(excerpt.split())
        if norm_excerpt in norm_text:
            # Re-derive offsets from the original (find best-effort)
            for i in range(len(answer_text) - len(excerpt) + 1):
                if " ".join(answer_text[i : i + len(excerpt)].split()) == norm_excerpt:
                    start = i
                    break
        if start == -1:
            raise MalformedLLMOutput(
                "evidence_excerpt is not a verbatim substring of answer_text. "
                "This is an unsupported-claim failure — refusing to record a result."
            )
    end = start + len(excerpt)
    if result.char_offset_start != start or result.char_offset_end != end:
        raise MalformedLLMOutput(
            f"char_offset_start/end mismatch: model gave "
            f"({result.char_offset_start},{result.char_offset_end}) "
            f"but verified location is ({start},{end})."
        )


# ---------------------------------------------------------------------------
# The canonical helper everyone calls
# ---------------------------------------------------------------------------


async def classify_response(
    adapter: LLMAdapter, ctx: LLMContext
) -> ClassificationResult:
    """Classify one response and verify the evidence excerpt is genuine."""
    result = await adapter.classify(ctx)
    verify_evidence(result, ctx.answer_text)
    return result
