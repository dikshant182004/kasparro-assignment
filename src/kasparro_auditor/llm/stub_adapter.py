"""Deterministic stub adapter for tests and offline eval.

This is NOT a real LLM. It uses regex + simple heuristics to produce a
plausible ClassificationResult so that the engineering pipeline (auth,
state machine, scoring, export) can be tested without any API key.

The grader deliberately compares this stub against the live model on
the same eval cases — that gap is one of the things the brief asks to
report.
"""
from __future__ import annotations

import re
import time

from kasparro_auditor.llm.adapter import LLMContext, MalformedLLMOutput, verify_evidence
from kasparro_auditor.schemas.results import ClassificationResult

# A short, conservative list of competitor names. Used both as hints and
# for the stub's competitor extraction.
KNOWN_COMPETITORS = [
    "profound",
    "otterly ai",
    "otterly",
    "peec ai",
    "peec",
    "athenahq",
    "athena hq",
    "dageno ai",
    "dageno",
    "authoritas",
    "aeo checker",
    "shareofmodel",
    "waikay",
    "rankscale",
    "amionai",
    "xfunnel",
    "quno",
]

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
    """Deterministic stub. Used in tests and as a replay fallback."""

    mode = "stub"
    model_name = "stub-v1"

    async def classify(self, ctx: LLMContext) -> ClassificationResult:
        text = ctx.answer_text
        lower = text.lower()
        brand = "kasparro"
        brand_lower = brand

        # Mention
        mention = brand_lower in lower

        # Recommendation — conservative
        recommendation = any(
            re.search(p, text, flags=re.IGNORECASE | re.MULTILINE) for p in _REC_PATTERNS
        )

        # Negative
        negative = any(
            re.search(p, lower, flags=re.MULTILINE) for p in _NEG_PATTERNS
        )

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

        # Competitors actually mentioned in the text
        competitors: list[str] = []
        seen = set()
        for name in KNOWN_COMPETITORS:
            if name in lower and name not in seen:
                # Coalesce to canonical name
                canon = _canonical_competitor(name)
                if canon not in seen:
                    competitors.append(canon)
                    seen.add(canon)

        # Evidence — pick a sentence containing "Kasparro" (or first sentence)
        evidence = _pick_evidence(text, brand)
        if not evidence:
            # No Kasparro in text: pick first 200 chars as evidence of absence
            evidence = text[:200].strip()
            if len(evidence) < 10:
                raise MalformedLLMOutput(
                    "Answer text too short to extract evidence", response_id=ctx.response_id
                )

        start = text.find(evidence)
        end = start + len(evidence)
        if start == -1:
            # Fallback to the very first sentence
            evidence = text.split(".")[0][:200].strip()
            start = text.find(evidence)
            end = start + len(evidence)

        result = ClassificationResult(
            mention=mention,
            recommendation=recommendation,
            negative_characterisation=negative,
            recommendation_position=position,
            competitors_named=competitors,
            evidence_excerpt=evidence,
            char_offset_start=start,
            char_offset_end=end,
        )
        # Stub verifies its own evidence so callers don't need to.
        verify_evidence(result, text)
        return result


def _canonical_competitor(name: str) -> str:
    """Map variants to one canonical form."""
    name = name.lower().strip()
    if name.startswith("otterly"):
        return "otterly ai"
    if name.startswith("peec"):
        return "peec ai"
    if name.startswith("athena"):
        return "athenahq"
    if name.startswith("dageno"):
        return "dageno ai"
    return name


def _pick_evidence(text: str, brand: str) -> str:
    """Return a verbatim substring containing `brand`, or '' if absent.

    Prefers a full sentence; falls back to ±100 chars around the brand.
    """
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
    # If the sentence is huge, trim to ±200 chars around the brand mention.
    if len(candidate) > 400:
        local_start = max(0, idx - sent_start - 200)
        local_end = min(len(candidate), idx - sent_start + 200)
        candidate = candidate[local_start:local_end].strip()
    return candidate
