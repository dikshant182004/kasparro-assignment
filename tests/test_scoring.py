"""Scoring v2 tests — new architecture per reviewer's fix spec.

Tests:
  1. Absent response scores exactly 0 (no free 15 points from negative_penalty)
  2. Headline = mean of row scores, tight tolerance
  3. Unobservable position renormalises; ranked recommendation changes only its own row
  4. All-failed run gives None, never 0
  5. Per-engine denominators add up to totals
  6. Weight sensitivity table is populated
  7. Branded vs generic slices are populated
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from kasparro_auditor.db.models import (
    AnalysisRun,
    Engine,
    ErrorCode,
    ExecutionMode,
    ResponseResult,
)
from kasparro_auditor.scoring.score import (
    CATEGORY_VERSION,
    SCORE_WEIGHTS_V2,
    SCORING_VERSION,
    _position_factor,
    _row_score,
    _RowInputs,
    compute_visibility_score,
)


def _row(
    response_id: str,
    *,
    mention: bool = False,
    recommendation: bool = False,
    negative: bool = False,
    position: int | None = None,
    has_kasparro_citation: bool = False,
    prompt_id: str = "p1",
    engine: Engine = Engine.CHATGPT,
) -> _RowInputs:
    """Helper to build a _RowInputs."""
    return _RowInputs(
        response_id=response_id,
        prompt_id=prompt_id,
        engine=engine.value,
        mention=mention,
        recommendation=recommendation,
        negative=negative,
        position=position,
        has_kasparro_citation=has_kasparro_citation,
    )


# ---------------------------------------------------------------------------
# Test 1: Absent response scores exactly 0
# ---------------------------------------------------------------------------


def test_absent_response_scores_zero():
    """An absent response (no mention, no rec, no citation) MUST score 0.

    This is the bug the reviewer caught: v1 gave absent responses 15 free
    points from negative_penalty. v2 uses negative as a 0.5 multiplier
    on a row score that's already 0, so absent → 0.
    """
    row = _row("r1", mention=False, recommendation=False, negative=False, has_kasparro_citation=False)
    assert _row_score(row) == 0.0


def test_absent_with_negative_still_zero():
    """Even if an absent response were flagged negative, score stays 0.

    0 × 0.5 = 0. No free points.
    """
    row = _row("r1", mention=False, recommendation=False, negative=True, has_kasparro_citation=False)
    assert _row_score(row) == 0.0


def test_mention_only_scores_above_zero():
    """A mention with no recommendation/citation scores positive."""
    row = _row("r1", mention=True, recommendation=False, negative=False, has_kasparro_citation=False)
    # active_weight = 0.30 + 0.40 + 0.15 = 0.85 (position dropped)
    # raw = 0.30 * 1 = 0.30
    # score = 100 * (0.30 / 0.85) = 35.29...
    assert _row_score(row) == pytest.approx(35.29, abs=0.5)


def test_full_recommendation_with_citation_scores_100():
    """A recommended + cited + #1-ranked response scores 100."""
    row = _row(
        "r1",
        mention=True,
        recommendation=True,
        negative=False,
        position=1,
        has_kasparro_citation=True,
    )
    assert _row_score(row) == pytest.approx(100.0, abs=0.5)


def test_negative_halves_score():
    """A negative characterisation multiplies the row score by 0.5."""
    row_pos = _row("r1", mention=True, recommendation=False, negative=False)
    row_neg = _row("r2", mention=True, recommendation=False, negative=True)
    assert _row_score(row_neg) == pytest.approx(_row_score(row_pos) * 0.5)


# ---------------------------------------------------------------------------
# Test 2: Headline = mean of row scores
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_headline_equals_mean_of_row_scores(session):
    """The headline composite MUST equal the mean of per-row composite_score.

    This is the bug the reviewer caught: v1's position_score was averaged
    only over rows that had a position, then applied as a flat 15% weight,
    causing the headline to diverge from the mean of its parts.
    """
    run = AnalysisRun(
        dataset_id="ds-test",
        execution_mode=ExecutionMode.STUB,
        model_name="stub-v1",
        status="completed",
        total=4,
        successful=4,
        failed=0,
        skipped=0,
    )
    session.add(run)
    await session.flush()

    # 4 rows with varied scores: 0, 35, 50, 100
    rows_data = [
        _row("r1", mention=False, recommendation=False, prompt_id="p1"),  # 0
        _row("r2", mention=True, prompt_id="p2"),  # ~35
        _row("r3", mention=True, recommendation=True, has_kasparro_citation=True, prompt_id="p3"),  # high
        _row("r4", mention=True, recommendation=True, position=1, has_kasparro_citation=True, prompt_id="p4"),  # 100
    ]
    for rd in rows_data:
        session.add(
            ResponseResult(
                run_id=run.id,
                response_id=rd.response_id,
                prompt_id=rd.prompt_id,
                engine=Engine.CHATGPT,
                mention=rd.mention,
                prompt_names_brand=True,
                recommendation=rd.recommendation,
                negative_characterisation=rd.negative,
                recommendation_position=rd.position,
                has_kasparro_citation=rd.has_kasparro_citation,
                execution_mode=ExecutionMode.STUB,
                model_name="stub-v1",
            )
        )
    await session.flush()

    score = await compute_visibility_score(session, run.id)

    # Pull per-row composite_score from DB
    db_rows = (
        await session.scalars(
            select(ResponseResult).where(ResponseResult.run_id == run.id)
        )
    ).all()
    row_mean = sum(r.composite_score for r in db_rows) / len(db_rows)

    assert score.composite is not None
    assert score.composite == pytest.approx(row_mean, abs=0.01)


# ---------------------------------------------------------------------------
# Test 3: Unobservable position renormalises
# ---------------------------------------------------------------------------


def test_position_unobservable_renormalises_weight():
    """When position is unobservable, its weight is dropped and the others renormalise.

    A mention-only row should score the same regardless of whether position
    weight is in the formula, because position is dropped.
    """
    # mention=True, no recommendation, no citation, no position
    row = _row("r1", mention=True, recommendation=False, position=None)
    # active_weight = 0.30 + 0.40 + 0.15 = 0.85 (position dropped)
    # raw = 0.30
    # score = 100 * 0.30 / 0.85 = 35.29
    assert _row_score(row) == pytest.approx(35.29, abs=0.5)


def test_position_observable_includes_its_weight():
    """When position is observable (recommended + has position), its weight is included."""
    row = _row(
        "r1",
        mention=True,
        recommendation=True,
        position=1,
        has_kasparro_citation=True,
    )
    # active_weight = 0.30 + 0.40 + 0.15 + 0.15 = 1.0 (full)
    # raw = 0.30 + 0.40 + 0.15*1.0 + 0.15 = 1.0
    # score = 100
    assert _row_score(row) == pytest.approx(100.0, abs=0.5)


def test_position_floor_at_0_2():
    """Position #5+ should give the 0.2 floor, not 0."""
    pos5_factor, observable = _position_factor(5, recommended=True)
    assert observable is True
    assert pos5_factor == pytest.approx(0.2)


def test_position_unobservable_when_not_recommended():
    """Position is unobservable when recommended=False, even if position is set."""
    _, observable = _position_factor(1, recommended=False)
    assert observable is False


# ---------------------------------------------------------------------------
# Test 4: All-failed run gives None, never 0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_failed_run_returns_none_not_zero(session):
    """If every row failed, composite MUST be None + insufficient_data=True.

    v1 returned composite=0.0 with CI (0,0) — the brief's "failed analysis
    as a successful zero-score observation" forbidden pattern.
    """
    run = AnalysisRun(
        dataset_id="ds-test",
        execution_mode=ExecutionMode.STUB,
        status="completed",
        total=3,
        successful=0,
        failed=3,
        skipped=0,
    )
    session.add(run)
    await session.flush()

    for rid in ("r1", "r2", "r3"):
        session.add(
            ResponseResult(
                run_id=run.id,
                response_id=rid,
                prompt_id="p1",
                engine=Engine.GEMINI,
                failed=True,
                failure_reason="simulated timeout",
                error_code=ErrorCode.TIMEOUT,
                execution_mode=ExecutionMode.STUB,
            )
        )
    await session.flush()

    score = await compute_visibility_score(session, run.id)
    assert score.composite is None
    assert score.composite_ci_low is None
    assert score.composite_ci_high is None
    assert score.insufficient_data is True
    assert score.denominator_valid == 0
    assert score.denominator_total == 3
    assert score.failed_count == 3


# ---------------------------------------------------------------------------
# Test 5: Per-engine denominators add up to totals
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_engine_denominators_add_up(session):
    """Sum of per-engine denominator_valid = headline denominator_valid."""
    run = AnalysisRun(
        dataset_id="ds-test",
        execution_mode=ExecutionMode.STUB,
        status="completed",
        total=6,
        successful=6,
        failed=0,
        skipped=0,
    )
    session.add(run)
    await session.flush()

    # 6 rows: 2 per engine
    for engine in (Engine.AIMODE, Engine.CHATGPT, Engine.GEMINI):
        for i in (1, 2):
            session.add(
                ResponseResult(
                    run_id=run.id,
                    response_id=f"r-{engine.value}-{i}",
                    prompt_id=f"p-{i}",
                    engine=engine,
                    mention=True,
                    prompt_names_brand=True,
                    execution_mode=ExecutionMode.STUB,
                )
            )
    await session.flush()

    score = await compute_visibility_score(session, run.id)
    engine_valid_sum = sum(s.denominator_valid for s in score.per_engine.values())
    assert engine_valid_sum == score.denominator_valid
    assert score.denominator_valid == 6


@pytest.mark.asyncio
async def test_per_category_denominators_add_up(session):
    """Sum of per-category denominator_valid = headline denominator_valid."""
    run = AnalysisRun(
        dataset_id="ds-test",
        execution_mode=ExecutionMode.STUB,
        status="completed",
        total=4,
        successful=4,
        failed=0,
        skipped=0,
    )
    session.add(run)
    await session.flush()

    # 4 rows: 2 branded + 2 generic
    for i, branded in [(1, True), (2, True), (3, False), (4, False)]:
        session.add(
            ResponseResult(
                run_id=run.id,
                response_id=f"r{i}",
                prompt_id=f"p{i}",
                engine=Engine.CHATGPT,
                mention=True,
                prompt_names_brand=branded,
                execution_mode=ExecutionMode.STUB,
            )
        )
    await session.flush()

    score = await compute_visibility_score(session, run.id)
    cat_valid_sum = sum(s.denominator_valid for s in score.per_category.values())
    assert cat_valid_sum == score.denominator_valid
    assert "branded" in score.per_category
    assert "generic" in score.per_category


# ---------------------------------------------------------------------------
# Test 6: Weight sensitivity table populated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_weight_sensitivity_table_populated(session):
    """The weight sensitivity table contains 3 alternative weight sets."""
    run = AnalysisRun(
        dataset_id="ds-test",
        execution_mode=ExecutionMode.STUB,
        status="completed",
        total=2,
        successful=2,
        failed=0,
        skipped=0,
    )
    session.add(run)
    await session.flush()

    for i in (1, 2):
        session.add(
            ResponseResult(
                run_id=run.id,
                response_id=f"r{i}",
                prompt_id=f"p{i}",
                engine=Engine.CHATGPT,
                mention=True,
                recommendation=True,
                prompt_names_brand=True,
                execution_mode=ExecutionMode.STUB,
            )
        )
    await session.flush()

    score = await compute_visibility_score(session, run.id)
    assert "balanced" in score.weight_sensitivity
    assert "mention_heavy" in score.weight_sensitivity
    assert "recommendation_heavy" in score.weight_sensitivity


# ---------------------------------------------------------------------------
# Test 7: Version tracking on the score object
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_records_version(session):
    """VisibilityScore records scoring_version + category_version."""
    run = AnalysisRun(
        dataset_id="ds-test",
        execution_mode=ExecutionMode.STUB,
        status="completed",
        total=1,
        successful=1,
        failed=0,
        skipped=0,
    )
    session.add(run)
    await session.flush()

    session.add(
        ResponseResult(
            run_id=run.id,
            response_id="r1",
            prompt_id="p1",
            engine=Engine.CHATGPT,
            mention=True,
            prompt_names_brand=True,
            execution_mode=ExecutionMode.STUB,
        )
    )
    await session.flush()

    score = await compute_visibility_score(session, run.id)
    assert score.scoring_version == SCORING_VERSION
    assert score.category_version == CATEGORY_VERSION


# ---------------------------------------------------------------------------
# Weights + position factor sanity tests
# ---------------------------------------------------------------------------


def test_weights_v2_sum_to_one():
    """v2 weights sum to 1.0."""
    assert sum(SCORE_WEIGHTS_V2.values()) == pytest.approx(1.0)


def test_position_factor_decay():
    """Position #1 = 1.0, #2 = 0.8, #5 = 0.2 (floor)."""
    assert _position_factor(1, True)[0] == 1.0
    assert _position_factor(2, True)[0] == pytest.approx(0.8)
    assert _position_factor(5, True)[0] == pytest.approx(0.2)
    assert _position_factor(10, True)[0] == pytest.approx(0.2)  # floor


# ---------------------------------------------------------------------------
# Failed rows excluded from denominator (kept from v1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_results_excluded_from_denominator(session):
    """A failed analysis is NOT a successful zero-score observation."""
    run = AnalysisRun(
        dataset_id="ds-test",
        execution_mode=ExecutionMode.STUB,
        status="completed",
        total=3,
        successful=1,
        failed=2,
        skipped=0,
    )
    session.add(run)
    await session.flush()

    session.add(
        ResponseResult(
            run_id=run.id,
            response_id="r1",
            prompt_id="p1",
            engine=Engine.CHATGPT,
            mention=True,
            recommendation=True,
            negative_characterisation=False,
            recommendation_position=1,
            competitors_known=[],
            evidence_mention="Kasparro is mentioned.",
            evidence_recommendation="Kasparro is recommended",
            has_kasparro_citation=True,
            citation_count_for_response=1,
            unique_citation_count=1,
            execution_mode=ExecutionMode.STUB,
        )
    )
    for rid in ("r2", "r3"):
        session.add(
            ResponseResult(
                run_id=run.id,
                response_id=rid,
                prompt_id="p2",
                engine=Engine.GEMINI,
                failed=True,
                failure_reason="simulated timeout",
                error_code=ErrorCode.TIMEOUT,
                execution_mode=ExecutionMode.STUB,
            )
        )
    await session.flush()

    score = await compute_visibility_score(session, run.id)
    assert score.denominator_valid == 1
    assert score.denominator_total == 3
    assert score.failed_count == 2
    assert score.mention_rate == pytest.approx(1.0)
