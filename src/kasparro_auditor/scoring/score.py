"""Composite visibility score — v2 (rewritten per reviewer's fix spec).

Key changes from v1:
  - Absent responses score 0 (no free 15 points from negative_penalty).
  - Negative is a 0.5 multiplier, not a +15 component.
  - Position is renormalised: when unobservable, its weight is dropped
    and the denominator shrinks.
  - Headline = mean of per-row scores (always consistent with its parts).
  - All-failed runs return composite=None + insufficient_data=True
    (NEVER a confident 0.0).
  - Bootstrap CI is cluster-bootstrap over PROMPTS (not responses),
    because each prompt appears 3 times (once per engine).
  - Per-engine and per-category slices with denominators.
  - Weight sensitivity table (3 alternative weight sets).

Formula per successful row:
  raw = w_mention*mention + w_recommended*recommended + w_position*pos + w_citation*cite
  row_score = 100 * (raw / active_weight) * (0.5 if negative else 1.0)

Where active_weight = w_mention + w_recommended + w_citation + (w_position if position observable)
                      else w_mention + w_recommended + w_citation

Default weights: mention=0.30, recommended=0.40, position=0.15, citation=0.15.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.db.models import ResponseResult
from kasparro_auditor.schemas.results import CategorySlice, EngineSlice, VisibilityScore

# ---- Weights --------------------------------------------------------------
# Recommendation gets the largest weight (it's the actual prize).
# Mention is the baseline presence gate. Position is "ranking quality".
# Citation is third-party validation. Negative is a 0.5 multiplier on the
# row score (not a separate additive component — this kills the "absent
# response gets 15 free points" bug).
SCORE_WEIGHTS_V2: dict[str, float] = {
    "mention": 0.30,
    "recommended": 0.40,
    "position": 0.15,
    "own_citation": 0.15,
}

# Three alternative weight sets for the sensitivity table.
WEIGHT_SENSITIVITY_SETS: list[tuple[str, dict[str, float]]] = [
    ("balanced", SCORE_WEIGHTS_V2),
    ("mention_heavy", {"mention": 0.50, "recommended": 0.30, "position": 0.10, "own_citation": 0.10}),
    ("recommendation_heavy", {"mention": 0.20, "recommended": 0.55, "position": 0.15, "own_citation": 0.10}),
]

NEGATIVE_MULTIPLIER = 0.5  # negative characterisation halves the row score
POSITION_FLOOR = 0.2       # rank #1 = 1.0, decays by 0.2/rank, floor at 0.2

SCORING_VERSION = "v2"
CATEGORY_VERSION = "v1"


@dataclass(slots=True)
class _RowInputs:
    """Inputs needed to score one row."""

    response_id: str
    prompt_id: str
    engine: str
    mention: bool
    recommendation: bool
    negative: bool
    position: int | None
    has_kasparro_citation: bool


def _position_factor(position: int | None, recommended: bool) -> tuple[float, bool]:
    """Return (position_factor, is_observable).

    Position is observable ONLY when recommended=True AND position is not None.
    When unobservable, the caller drops the position weight and renormalises.
    """
    if not recommended or position is None:
        return 0.0, False
    return max(POSITION_FLOOR, 1.0 - 0.2 * (position - 1)), True


def _row_score(row: _RowInputs, weights: dict[str, float] = SCORE_WEIGHTS_V2) -> float:
    """Compute the score for one successful row. Always in [0, 100].

    An absent response (mention=False, recommended=False, citation=False,
    position unobservable) scores 0 — no free points.
    """
    pos_factor, pos_observable = _position_factor(row.position, row.recommendation)
    active_weight = (
        weights["mention"]
        + weights["recommended"]
        + weights["own_citation"]
        + (weights["position"] if pos_observable else 0.0)
    )
    if active_weight == 0:
        return 0.0
    raw = (
        weights["mention"] * (1.0 if row.mention else 0.0)
        + weights["recommended"] * (1.0 if row.recommendation else 0.0)
        + weights["position"] * pos_factor
        + weights["own_citation"] * (1.0 if row.has_kasparro_citation else 0.0)
    )
    multiplier = NEGATIVE_MULTIPLIER if row.negative else 1.0
    return 100.0 * (raw / active_weight) * multiplier


def _cluster_bootstrap_ci(
    rows: list[_RowInputs],
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    """Cluster bootstrap over prompts (each prompt has up to 3 engine rows).

    Returns (ci_low, ci_high). Returns (None, None) if there are <2 unique
    prompts or no valid rows. The interval covers PROMPT-SAMPLING ONLY —
    not classifier error.
    """
    valid_rows = [r for r in rows]
    if not valid_rows:
        return None, None

    # Group row scores by prompt_id (cluster unit)
    by_prompt: dict[str, list[float]] = {}
    for r in valid_rows:
        by_prompt.setdefault(r.prompt_id, []).append(_row_score(r))
    prompt_ids = list(by_prompt.keys())
    if len(prompt_ids) < 2:
        return None, None

    rng = random.Random(seed)
    n_prompts = len(prompt_ids)
    boots: list[float] = []
    for _ in range(n_samples):
        # Resample prompts with replacement
        sampled_prompt_scores: list[float] = []
        for _ in range(n_prompts):
            pid = prompt_ids[rng.randrange(n_prompts)]
            # Pick a random engine row from this prompt's cluster
            cluster = by_prompt[pid]
            sampled_prompt_scores.append(cluster[rng.randrange(len(cluster))])
        boots.append(sum(sampled_prompt_scores) / len(sampled_prompt_scores))
    boots.sort()
    lo = boots[int(0.025 * n_samples)]
    hi = boots[int(0.975 * n_samples) - 1]
    return lo, hi


def _build_slice(rows: list[_RowInputs]) -> tuple[float | None, int, int, int, int] | None:
    """Build a score slice for a list of rows.

    Returns (composite, denominator_valid, denominator_total, failed, skipped)
    or None if there are no rows at all in this slice.
    """
    if not rows:
        return None
    valid_rows = rows  # all rows here are valid (caller filters)
    if not valid_rows:
        return (None, 0, 0, 0, 0)
    per_row = [_row_score(r) for r in valid_rows]
    composite = sum(per_row) / len(per_row)
    lo, hi = _cluster_bootstrap_ci(valid_rows)
    # Return composite + the CI as part of the slice
    return (composite, len(valid_rows), len(valid_rows), 0, 0, lo, hi)  # type: ignore[return-value]


async def compute_visibility_score(
    session: AsyncSession,
    run_id: str,
    bootstrap_samples: int = 1000,
) -> VisibilityScore:
    """Compute the composite score + per-engine + per-category slices.

    Returns None composite + insufficient_data=True if no valid rows.
    """
    rows_orm = (
        await session.scalars(
            select(ResponseResult).where(ResponseResult.run_id == run_id)
        )
    ).all()

    total = len(rows_orm)
    failed = sum(1 for r in rows_orm if r.failed)
    skipped = sum(1 for r in rows_orm if r.skipped)
    valid_rows_orm = [r for r in rows_orm if not r.failed and not r.skipped]
    valid = len(valid_rows_orm)

    if valid == 0:
        # All-failed run — return None composite, NEVER 0.0
        return VisibilityScore(
            composite=None,
            composite_ci_low=None,
            composite_ci_high=None,
            insufficient_data=True,
            weights=SCORE_WEIGHTS_V2,
            denominator_valid=0,
            denominator_total=total,
            failed_count=failed,
            skipped_count=skipped,
            per_engine={},
            per_category={},
            weight_sensitivity={},
            scoring_version=SCORING_VERSION,
            category_version=CATEGORY_VERSION,
        )

    # Build _RowInputs for each valid row
    rows: list[_RowInputs] = [
        _RowInputs(
            response_id=r.response_id,
            prompt_id=r.prompt_id,
            engine=r.engine.value,
            mention=r.mention,
            recommendation=r.recommendation,
            negative=r.negative_characterisation,
            position=r.recommendation_position,
            has_kasparro_citation=r.has_kasparro_citation,
        )
        for r in valid_rows_orm
    ]

    # Per-row scores — persist back to each ResponseResult for traceability
    per_row_scores = [_row_score(r) for r in rows]
    for r_orm, score in zip(valid_rows_orm, per_row_scores, strict=True):
        # Persist component scores per row
        pos_factor, pos_observable = _position_factor(r_orm.recommendation_position, r_orm.recommendation)
        r_orm.mention_score = 1.0 if r_orm.mention else 0.0
        r_orm.recommendation_score = 1.0 if r_orm.recommendation else 0.0
        r_orm.negative_penalty = NEGATIVE_MULTIPLIER if r_orm.negative_characterisation else 1.0
        r_orm.citation_score = 1.0 if r_orm.has_kasparro_citation else 0.0
        r_orm.position_score = pos_factor
        r_orm.composite_score = score
    await session.flush()

    # Headline = mean of per-row scores (ALWAYS consistent with parts)
    composite = sum(per_row_scores) / len(per_row_scores)
    ci_low, ci_high = _cluster_bootstrap_ci(rows, n_samples=bootstrap_samples)

    # Per-engine slice
    per_engine: dict[str, EngineSlice] = {}
    for engine in ("aimode", "chatgpt", "gemini"):
        engine_rows = [r for r in rows if r.engine == engine]
        if not engine_rows:
            continue
        engine_scores = [_row_score(r) for r in engine_rows]
        engine_composite = sum(engine_scores) / len(engine_scores)
        lo, hi = _cluster_bootstrap_ci(engine_rows, n_samples=bootstrap_samples)
        # Failed/skipped counts for this engine
        engine_total = sum(1 for r in rows_orm if r.engine.value == engine)
        engine_failed = sum(1 for r in rows_orm if r.engine.value == engine and r.failed)
        engine_skipped = sum(1 for r in rows_orm if r.engine.value == engine and r.skipped)
        per_engine[engine] = EngineSlice(
            composite=engine_composite,
            composite_ci_low=lo,
            composite_ci_high=hi,
            denominator_valid=len(engine_rows),
            denominator_total=engine_total,
            failed_count=engine_failed,
            skipped_count=engine_skipped,
        )

    # Per-category slice — branded vs generic is the lead split
    per_category: dict[str, CategorySlice] = {}
    # Pull prompt_names_brand from the ORM rows (it's a column on ResponseResult)
    for slice_key, predicate in [
        ("branded", lambda r: r.prompt_names_brand),
        ("generic", lambda r: not r.prompt_names_brand),
    ]:
        slice_rows_orm = [r for r in valid_rows_orm if predicate(r)]
        if not slice_rows_orm:
            continue
        slice_rows = [
            _RowInputs(
                response_id=r.response_id,
                prompt_id=r.prompt_id,
                engine=r.engine.value,
                mention=r.mention,
                recommendation=r.recommendation,
                negative=r.negative_characterisation,
                position=r.recommendation_position,
                has_kasparro_citation=r.has_kasparro_citation,
            )
            for r in slice_rows_orm
        ]
        slice_scores = [_row_score(r) for r in slice_rows]
        slice_composite = sum(slice_scores) / len(slice_scores)
        lo, hi = _cluster_bootstrap_ci(slice_rows, n_samples=bootstrap_samples)
        slice_total = sum(1 for r in rows_orm if predicate(r))
        slice_failed = sum(1 for r in rows_orm if predicate(r) and r.failed)
        slice_skipped = sum(1 for r in rows_orm if predicate(r) and r.skipped)
        per_category[slice_key] = CategorySlice(
            composite=slice_composite,
            composite_ci_low=lo,
            composite_ci_high=hi,
            denominator_valid=len(slice_rows),
            denominator_total=slice_total,
            failed_count=slice_failed,
            skipped_count=slice_skipped,
        )

    # Weight sensitivity table
    weight_sensitivity: dict[str, float] = {}
    for name, weights in WEIGHT_SENSITIVITY_SETS:
        alt_scores = [_row_score(r, weights) for r in rows]
        weight_sensitivity[name] = sum(alt_scores) / len(alt_scores)

    # Also compute mention/recommendation/negative/citation rates for the
    # headline (kept for diagnostics backward-compat)
    mention_rate = sum(1 for r in rows if r.mention) / valid
    recommendation_rate = sum(1 for r in rows if r.recommendation) / valid
    negative_rate = sum(1 for r in rows if r.negative) / valid
    citation_rate = sum(1 for r in rows if r.has_kasparro_citation) / valid
    positions = [r.position for r in rows if r.recommendation and r.position is not None]
    avg_position = (sum(positions) / len(positions)) if positions else None

    return VisibilityScore(
        composite=composite,
        composite_ci_low=ci_low,
        composite_ci_high=ci_high,
        insufficient_data=False,
        weights=SCORE_WEIGHTS_V2,
        denominator_valid=valid,
        denominator_total=total,
        failed_count=failed,
        skipped_count=skipped,
        mention_rate=mention_rate,
        recommendation_rate=recommendation_rate,
        negative_rate=negative_rate,
        citation_rate=citation_rate,
        avg_position=avg_position,
        per_engine=per_engine,
        per_category=per_category,
        weight_sensitivity=weight_sensitivity,
        scoring_version=SCORING_VERSION,
        category_version=CATEGORY_VERSION,
    )
