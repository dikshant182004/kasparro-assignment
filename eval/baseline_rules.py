"""Truly independent rule-based baseline classifier.

Previous version imported the stub's regex patterns + _pick_evidence
function — making the comparison meaningless (a classifier against
itself). This version uses PLAIN keyword matching only:
  - mention: substring "kasparro" (case-insensitive)
  - recommendation: keyword patterns ("kasparro is best", "recommend kasparro", etc.)
  - negative: keyword patterns ("gimmick", "hype", "scam", "not legit", etc.)
  - position: pattern "[N]. Kasparro" or "Kasparro (Nth)"
  - evidence: substring around the keyword match (truncated to 300 chars)

NO imports from stub_adapter.py. NO aliasing of the stub's regexes.
This is genuinely independent — when the stub wins on a label that the
baseline also predicts, it's because the label is easy; when the stub
wins on a label the baseline misses, that's the LLM's value-add
(measurable in stub-vs-live mode).
"""
from __future__ import annotations

import re

from kasparro_auditor.llm.adapter import (
    LLMContext,
    Judgment,
    verify_evidence,
)


# PLAIN keyword patterns — independent of the stub's _NEG_PATTERNS.
# These are intentionally simpler/dumber than the stub's regexes so the
# comparison is meaningful.
_BASLINE_NEG_KEYWORDS = [
    "gimmick",
    "scam",
    "hype",
    "overpriced",
    "not legit",
    "not legitimate",
    "not worth",
    "skeptical",
    "sceptical",
    "not widely recognized",
    "not well known",
]

# PLAIN keyword patterns for recommendation.
_BASLINE_REC_KEYWORDS = [
    "kasparro is the best",
    "kasparro is best",
    "kasparro is a great",
    "kasparro is a good",
    "kasparro is a strong",
    "kasparro is a solid",
    "kasparro is recommended",
    "recommend kasparro",
    "kasparro is the top pick",
    "kasparro is my top",
    "kasparro is the top choice",
    "kasparro stands out",
]

# Position patterns: "[N]. Kasparro" at start of line, or "Kasparro (Nth)"
_BASLINE_POSITION_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(\d{1,2})[.)]\s*[A-Z][^\n]{0,40}\bKasparro\b", re.I),
    re.compile(r"\bKasparro\b[^\n]{0,40}\((\d{1,2})(?:st|nd|rd|th)?\)", re.I),
]


class RuleBaseline:
    """Plain keyword baseline — NO shared code with the StubAdapter.

    Used as the 'simple rule-based baseline' the brief asks to compare
    against. If the candidate classifier (stub/live) only matches this
    baseline's accuracy, the LLM is adding no value.
    """

    name = "rule-baseline-v2"
    mode = "stub"
    model_name = "rule-baseline-v2"

    async def classify(self, ctx: LLMContext) -> Judgment:
        text = ctx.answer_text
        lower = text.lower()

        # Mention: plain substring check (case-insensitive)
        mention = "kasparro" in lower

        # Recommendation: any baseline keyword matches
        recommendation = any(kw in lower for kw in _BASLINE_REC_KEYWORDS)

        # Negative: any baseline keyword matches
        negative = any(kw in lower for kw in _BASLINE_NEG_KEYWORDS)

        # Position: first matching pattern wins
        position: int | None = None
        if recommendation:
            for pat in _BASLINE_POSITION_PATTERNS:
                m = pat.search(text)
                if m:
                    try:
                        position = int(m.group(1))
                    except (IndexError, ValueError):
                        pass
                    break

        # Evidence: ±150 chars around the first kasparro mention (or first 200 chars)
        evidence_mention = ""
        if mention:
            idx = lower.find("kasparro")
            if idx != -1:
                start = max(0, idx - 150)
                end = min(len(text), idx + 150)
                evidence_mention = text[start:end].strip()

        evidence_recommendation = ""
        if recommendation:
            for kw in _BASLINE_REC_KEYWORDS:
                idx = lower.find(kw)
                if idx != -1:
                    start = max(0, idx - 100)
                    end = min(len(text), idx + len(kw) + 100)
                    evidence_recommendation = text[start:end].strip()
                    break

        evidence_negative = ""
        if negative:
            for kw in _BASLINE_NEG_KEYWORDS:
                idx = lower.find(kw)
                if idx != -1:
                    start = max(0, idx - 100)
                    end = min(len(text), idx + len(kw) + 100)
                    evidence_negative = text[start:end].strip()
                    break

        result = Judgment(
            recommendation=recommendation,
            negative_characterisation=negative,
            recommendation_position=position,
            competitors_extra=[],  # baseline doesn't extract competitors
            evidence_mention=evidence_mention,
            evidence_recommendation=evidence_recommendation,
            evidence_negative=evidence_negative,
        )
        verify_evidence(result, text)
        result._model_used = self.model_name  # type: ignore[attr-defined]
        result._attempts = 1  # type: ignore[attr-defined]
        return result
