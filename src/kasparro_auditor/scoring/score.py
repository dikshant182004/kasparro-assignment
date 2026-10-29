"""Composite visibility score with components and bootstrap uncertainty.

Formula (transparent and disclosed):
  composite = 100 * (
      0.30 * mention_rate
    + 0.25 * recommendation_rate
    + 0.15 * (1 - negative_rate)
    + 0.15 * citation_rate
    + 0.15 * position_score
  )

Each rate uses `valid_responses` as the denominator (excluding failed).
Failed responses are reported separately as `failed_count / total`.

A mention is NOT a recommendation. A failed analysis is NOT a successful
zero-score observation. These rules are enforced structurally.
"""
from __future__ import annotations

import random
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.db.models import ResponseResult
from kasparro_auditor.schemas.results import VisibilityScore

SCORE_WEIGHTS: dict[str, float] = {
    "mention_rate": 0.30,
    "recommendation_rate": 0.25,
    "negative_penalty": 0.15,
    "citation_rate": 0.15,
    "position_score": 0.15,
}


def _position_score(positions: Iterable[int | None]) -> float:
    """Convert ranked positions to a 0..1 score.

    Position 1 → 1.0, position 2 → 0.8, position 3 → 0.6, decaying by 0.2
    per rank. Beyond 5 (or no position observed), score = 0.2 (participation
    floor). No recommendation means no position → 0.0 (but mention_rate
    already captures presence, so this is fine).
    """
    ranked = [p for p in positions if p is not None and p >= 1]
    if not ranked:
        return 0.0
    scores = [max(0.2, 1.0 - 0.2 * (p - 1)) for p in ranked]
    return sum(scores) / len(scores)


async def compute_visibility_score(
    session: AsyncSession, run_id: str, bootstrap_samples: int = 1000
) -> VisibilityScore:
    """Compute the composite score + 95% bootstrap CI over the prompt grid."""
    rows = (
        await session.scalars(
            select(ResponseResult).where(ResponseResult.run_id == run_id)
        )
    ).all()

    total = len(rows)
    failed = sum(1 for r in rows if r.failed)
    skipped = sum(1 for r in rows if r.skipped)
    valid_rows = [r for r in rows if not r.failed and not r.skipped]
    valid = len(valid_rows)
    if valid == 0:
        return VisibilityScore(
            mention_rate=0.0,
            recommendation_rate=0.0,
            negative_rate=0.0,
            citation_rate=0.0,
            avg_position=None,
            composite=0.0,
            composite_ci_low=0.0,
            composite_ci_high=0.0,
            weights=SCORE_WEIGHTS,
            denominator_valid=0,
            denominator_total=total,
            failed_count=failed,
            skipped_count=skipped,
        )

    mentions = sum(1 for r in valid_rows if r.mention)
    recommendations = sum(1 for r in valid_rows if r.recommendation)
    negatives = sum(1 for r in valid_rows if r.negative_characterisation)
    citations = sum(1 for r in valid_rows if r.has_kasparro_citation)
    positions = [
        r.recommendation_position
        for r in valid_rows
        if r.recommendation and r.recommendation_position is not None
    ]

    mention_rate = mentions / valid
    recommendation_rate = recommendations / valid
    negative_rate = negatives / valid
    citation_rate = citations / valid
    pos_score = _position_score([r.recommendation_position for r in valid_rows])
    avg_position = (sum(positions) / len(positions)) if positions else None

    composite = (
        SCORE_WEIGHTS["mention_rate"] * mention_rate
        + SCORE_WEIGHTS["recommendation_rate"] * recommendation_rate
        + SCORE_WEIGHTS["negative_penalty"] * (1.0 - negative_rate)
        + SCORE_WEIGHTS["citation_rate"] * citation_rate
        + SCORE_WEIGHTS["position_score"] * pos_score
    ) * 100.0

    # Bootstrap CI over the per-response composite contributions
    rng = random.Random(42)  # deterministic seed
    per_row_scores = [
        (
            SCORE_WEIGHTS["mention_rate"] * (1.0 if r.mention else 0.0)
            + SCORE_WEIGHTS["recommendation_rate"] * (1.0 if r.recommendation else 0.0)
            + SCORE_WEIGHTS["negative_penalty"] * (0.0 if r.negative_characterisation else 1.0)
            + SCORE_WEIGHTS["citation_rate"] * (1.0 if r.has_kasparro_citation else 0.0)
            + SCORE_WEIGHTS["position_score"]
            * (
                max(0.2, 1.0 - 0.2 * (r.recommendation_position - 1))
                if r.recommendation_position is not None and r.recommendation
                else 0.0
            )
        )
        * 100.0
        for r in valid_rows
    ]
    if bootstrap_samples > 0 and valid > 1:
        boots = []
        for _ in range(bootstrap_samples):
            sample = [
                per_row_scores[rng.randrange(valid)] for _ in range(valid)
            ]
            boots.append(sum(sample) / valid)
        boots.sort()
        lo = boots[int(0.025 * bootstrap_samples)]
        hi = boots[int(0.975 * bootstrap_samples) - 1]
    else:
        lo = hi = composite

    # Persist score components back to each result row for traceability.
    for r in valid_rows:
        r.mention_score = 1.0 if r.mention else 0.0
        r.recommendation_score = 1.0 if r.recommendation else 0.0
        r.negative_penalty = 0.0 if r.negative_characterisation else 1.0
        r.citation_score = 1.0 if r.has_kasparro_citation else 0.0
        r.position_score = (
            max(0.2, 1.0 - 0.2 * (r.recommendation_position - 1))
            if (r.recommendation_position is not None and r.recommendation)
            else 0.0
        )
        r.composite_score = (
            SCORE_WEIGHTS["mention_rate"] * r.mention_score
            + SCORE_WEIGHTS["recommendation_rate"] * r.recommendation_score
            + SCORE_WEIGHTS["negative_penalty"] * r.negative_penalty
            + SCORE_WEIGHTS["citation_rate"] * r.citation_score
            + SCORE_WEIGHTS["position_score"] * r.position_score
        ) * 100.0
    await session.flush()

    return VisibilityScore(
        mention_rate=mention_rate,
        recommendation_rate=recommendation_rate,
        negative_rate=negative_rate,
        citation_rate=citation_rate,
        avg_position=avg_position,
        composite=composite,
        composite_ci_low=lo,
        composite_ci_high=hi,
        weights=SCORE_WEIGHTS,
        denominator_valid=valid,
        denominator_total=total,
        failed_count=failed,
        skipped_count=skipped,
    )
