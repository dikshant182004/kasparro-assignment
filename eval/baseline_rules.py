"""Rule-based baseline classifier.

Pure regex + keyword matching. The grader compares the LLM/stub
classifier against this baseline on the same eval cases — the gap
between the two is one of the things the brief asks to report.
"""
from __future__ import annotations

import re

from kasparro_auditor.llm.adapter import LLMContext, verify_evidence
from kasparro_auditor.llm.stub_adapter import (
    KNOWN_COMPETITORS,
    _NEG_PATTERNS,
    _POSITION_PATTERNS,
    _REC_PATTERNS,
    _canonical_competitor,
    _pick_evidence,
)
from kasparro_auditor.schemas.results import ClassificationResult


class RuleBaseline:
    """Even simpler than the StubAdapter — no LLM, pure regex only.

    Used as the 'simple rule-based baseline' the brief asks to compare
    against. Differs from StubAdapter in that it does NOT pick
    evidence via sentence-boundary heuristics — it uses the first
    200 chars around the brand mention.
    """

    name = "rule-baseline-v1"

    async def classify(self, ctx: LLMContext) -> ClassificationResult:
        text = ctx.answer_text
        lower = text.lower()

        mention = "kasparro" in lower
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

        competitors: list[str] = []
        seen = set()
        for name in KNOWN_COMPETITORS:
            if name in lower:
                canon = _canonical_competitor(name)
                if canon not in seen:
                    competitors.append(canon)
                    seen.add(canon)

        # Baseline evidence: ±100 chars around first Kasparro mention
        evidence = _pick_evidence(text, "kasparro")
        if not evidence:
            evidence = text[:200]

        start = text.find(evidence)
        end = start + len(evidence)
        if start == -1:
            evidence = text[:200]
            start = 0
            end = min(200, len(text))

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
        verify_evidence(result, text)
        return result
