"""Rule-based baseline classifier.

Pure regex + keyword matching. The grader compares the LLM/stub
classifier against this baseline on the same eval cases — the gap
between the two is one of the things the brief asks to report.
"""
from __future__ import annotations

import re

from kasparro_auditor.llm.adapter import (
    LLMContext,
    Judgment,
    verify_evidence,
)
from kasparro_auditor.scoring.competitors import COMPETITOR_ALIASES
from kasparro_auditor.scoring.facts import compute_competitors
from kasparro_auditor.llm.stub_adapter import _pick_evidence, _pick_evidence_for_pattern

# Heuristics for negative characterisation (same as StubAdapter)
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

# Heuristics for recommendation
_REC_PATTERNS = [
    r"\bkasparro\s+is\s+(?:the\s+)?(?:best|top|recommended|worth|great|good)\b",
    r"\brecommend\s+kasparro\b",
    r"\bkasparro\s+is\s+a\s+(?:great|good|strong|solid)\s+(?:choice|option|fit)\b",
    r"\b(?:top\s+pick|best\s+option)[^.]*\bkasparro\b",
    r"\bkasparro\s+stands\s+out\b",
]

# Numeric position pattern
_POSITION_PATTERNS = [
    r"(?:^|\n)\s*(\d{1,2})[.)]\s*[A-Z][^\n]{0,40}\bKasparro\b",
    r"\bKasparro\b[^\n]{0,40}\((\d{1,2})(?:st|nd|rd|th)?\)",
]


class RuleBaseline:
    """Even simpler than the StubAdapter — pure regex only, no Facts.

    Used as the 'simple rule-based baseline' the brief asks to compare
    against. Differs from StubAdapter in that it computes its own mention
    flag and competitor list inline (doesn't use the Facts layer).
    """

    name = "rule-baseline-v1"
    mode = "stub"
    model_name = "rule-baseline-v1"

    async def classify(self, ctx: LLMContext) -> Judgment:
        text = ctx.answer_text
        lower = text.lower()

        # Baseline uses its own regex for mention (independent of Facts)
        mention = "kasparro" in lower
        # Baseline doesn't compute spans — just the boolean
        recommendation = any(
            re.search(p, text, flags=re.IGNORECASE | re.MULTILINE) for p in _REC_PATTERNS
        )
        negative = any(re.search(p, lower, flags=re.MULTILINE) for p in _NEG_PATTERNS)
        position = None
        if recommendation:
            for p in _POSITION_PATTERNS:
                m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
                if m:
                    try:
                        position = int(m.group(1))
                    except (IndexError, ValueError):
                        pass
                    break

        # Use the same competitor matcher as the Facts layer
        competitors_known = compute_competitors(text)
        competitors_extra: list[str] = []  # baseline doesn't differentiate

        evidence_mention = _pick_evidence(text, "kasparro") if mention else ""
        evidence_recommendation = _pick_evidence(text, "kasparro") if recommendation else ""
        evidence_negative = ""
        if negative:
            for p in _NEG_PATTERNS:
                m = re.search(p, text, flags=re.IGNORECASE | re.MULTILINE)
                if m:
                    start = max(0, m.start() - 100)
                    end = min(len(text), m.end() + 100)
                    evidence_negative = text[start:end].strip()
                    break

        result = Judgment(
            recommendation=recommendation,
            negative_characterisation=negative,
            recommendation_position=position,
            competitors_extra=competitors_extra,
            evidence_mention=evidence_mention,
            evidence_recommendation=evidence_recommendation,
            evidence_negative=evidence_negative,
        )
        verify_evidence(result, text)
        result._model_used = self.model_name  # type: ignore[attr-defined]
        result._attempts = 1  # type: ignore[attr-defined]
        return result
