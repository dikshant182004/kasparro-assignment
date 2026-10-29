"""Deterministic stub adapter for tests and offline eval.

This is NOT a real LLM. It uses regex + simple heuristics to produce a
plausible Judgment so that the engineering pipeline (auth, state machine,
scoring, export) can be tested without any API key.

The grader deliberately compares this stub against the live model on
the same eval cases — that gap is one of the things the brief asks to
report.
"""
from __future__ import annotations

import re

from kasparro_auditor.llm.adapter import (
    LLMContext,
    MalformedLLMOutput,
    Judgment,
    verify_evidence,
)

# Heuristics for negative characterisation.
_NEG_PATTERNS = [
    r"\b(?:not\s+(?:legit|legitimate|real|worth\s+it))\b",
    r"\bgimmick\b",
    r"\bhype\b",
    r"\bscam\b",
    r"\boverpriced\b",
    r"\bskept(?:ic(?:al)?)?\b",
    r"\b(?:empty\s+promises|all\s+hype)\b",
    r"\bnot\s+(?:widely\s+recognized|well\s+known)\b",
]

# Heuristics for recommendation. Conservative: requires positive phrasing.
_REC_PATTERNS = [
    r"\bkasparro\s+is\s+(?:the\s+)?(?:best|top|recommended|worth|great|good)\b",
    r"\brecommend\s+kasparro\b",
    r"\bkasparro\s+is\s+a\s+(?:great|good|strong|solid)\s+(?:choice|option|fit)\b",
    r"\b(?:top\s+pick|best\s+option)[^.]*\bkasparro\b",
    r"\bkasparro\s+stands\s+out\b",
]

# Numeric position pattern, e.g. "1. Kasparro", "Kasparro (1st)"
_POSITION_PATTERNS = [
    r"(?:^|\n)\s*(\d{1,2})[.)]\s*[A-Z][^\n]{0,40}\bKasparro\b",
    r"\bKasparro\b[^\n]{0,40}\((\d{1,2})(?:st|nd|rd|th)?\)",
]


class StubAdapter:
    """Deterministic stub. Used in tests and as the eval baseline."""

    mode = "stub"
    model_name = "stub-v1"

    async def classify(self, ctx: LLMContext) -> Judgment:
        text = ctx.answer_text
        lower = text.lower()

        # The stub has access to ctx.mention (computed deterministically).
        # recommendation/negative are the stub's "judgment".
        recommendation = any(
            re.search(p, text, flags=re.IGNORECASE | re.MULTILINE) for p in _REC_PATTERNS
        )
        negative = any(re.search(p, lower, flags=re.MULTILINE) for p in _NEG_PATTERNS)

        # Position — first matching pattern wins
        position: int | None = None
        if recommendation:
            for p in _POSITION_PATTERNS:
                m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
                if m:
                    try:
                        position = int(m.group(1))
                    except (IndexError, ValueError):
                        pass
                    break

        # competitors_extra: names mentioned in the text that are NOT already
        # in ctx.competitors_known (which was computed deterministically).
        # For the stub, we just return an empty list — the Facts layer
        # already captures known competitors.
        competitors_extra: list[str] = []

        # Evidence per true label
        evidence_mention = ""
        evidence_recommendation = ""
        evidence_negative = ""

        if ctx.mention:
            evidence_mention = _pick_evidence(text, "kasparro")
            if not evidence_mention:
                # Brand was named by regex but we couldn't extract a sentence —
                # this is a stub failure, not a real classification.
                raise MalformedLLMOutput(
                    "Could not extract mention evidence despite regex match",
                    response_id=ctx.response_id,
                )
        if recommendation:
            evidence_recommendation = _pick_evidence(text, "kasparro")
        if negative:
            evidence_negative = _pick_evidence_for_pattern(text, _NEG_PATTERNS)

        result = Judgment(
            recommendation=recommendation,
            negative_characterisation=negative,
            recommendation_position=position,
            competitors_extra=competitors_extra,
            evidence_mention=evidence_mention,
            evidence_recommendation=evidence_recommendation,
            evidence_negative=evidence_negative,
        )
        # Stub verifies its own evidence so callers don't need to.
        verify_evidence(result, text, response_id=ctx.response_id)
        result._model_used = self.model_name  # type: ignore[attr-defined]
        result._attempts = 1  # type: ignore[attr-defined]
        return result


def _pick_evidence(text: str, brand: str) -> str:
    """Return a verbatim substring containing `brand`, or '' if absent."""
    lower = text.lower()
    idx = lower.find(brand.lower())
    if idx == -1:
        return ""
    # Find sentence boundaries around the mention
    sent_start = text.rfind(".", 0, idx)
    sent_start = 0 if sent_start == -1 else sent_start + 1
    sent_end = text.find(".", idx)
    sent_end = len(text) if sent_end == -1 else sent_end + 1
    candidate = text[sent_start:sent_end].strip()
    if len(candidate) > 300:
        local_start = max(0, idx - sent_start - 150)
        local_end = min(len(candidate), idx - sent_start + 150)
        candidate = candidate[local_start:local_end].strip()
    return candidate


def _pick_evidence_for_pattern(text: str, patterns: list[str]) -> str:
    """Return a verbatim substring around the first pattern match."""
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 100)
            return text[start:end].strip()
    return text[:200].strip()
