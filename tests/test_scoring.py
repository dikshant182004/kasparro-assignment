"""Scoring math tests (deterministic; no LLM)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from kasparro_auditor.db.models import (
    AnalysisRun,
    Engine,
    ExecutionMode,
    ResponseResult,
)
from kasparro_auditor.scoring.score import (
    SCORE_WEIGHTS,
    _position_score,
    compute_visibility_score,
)


def test_position_score_basic():
    assert _position_score([1]) == 1.0
    assert _position_score([2]) == pytest.approx(0.8)
    assert _position_score([5]) == pytest.approx(0.2)
    assert _position_score([None, None]) == 0.0
    assert _position_score([]) == 0.0
    # Average of multiple positions
    assert _position_score([1, 3]) == pytest.approx((1.0 + 0.6) / 2)


def test_score_weights_sum_to_one():
    """The five weights must sum to 1.0 — disclosed in the report."""
    assert sum(SCORE_WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_compute_visibility_score_with_synthetic_data(session):
    """Verify the composite score matches hand-computed values."""
    # Insert a synthetic run + results
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

    rows = [
        ResponseResult(
            run_id=run.id,
            response_id="r1",
            prompt_id="p1",
            engine=Engine.CHATGPT,
            mention=True,
            recommendation=True,
            negative_characterisation=False,
            recommendation_position=1,
            competitors_named=["profound"],
            evidence_excerpt="Kasparro is recommended as a top option.",
            char_offset_start=0,
            char_offset_end=37,
            has_kasparro_citation=True,
            citation_count_for_response=1,
            unique_citation_count=1,
            execution_mode=ExecutionMode.STUB,
            model_name="stub-v1",
        ),
        ResponseResult(
            run_id=run.id,
            response_id="r2",
            prompt_id="p1",
            engine=Engine.GEMINI,
            mention=True,
            recommendation=False,
            negative_characterisation=True,
            recommendation_position=None,
            competitors_named=[],
            evidence_excerpt="Kasparro is mentioned but is just hype.",
            char_offset_start=0,
            char_offset_end=35,
            has_kasparro_citation=False,
            citation_count_for_response=0,
            unique_citation_count=0,
            execution_mode=ExecutionMode.STUB,
            model_name="stub-v1",
        ),
        ResponseResult(
            run_id=run.id,
            response_id="r3",
            prompt_id="p2",
            engine=Engine.AIMODE,
            mention=False,
            recommendation=False,
            negative_characterisation=False,
            recommendation_position=None,
            competitors_named=[],
            evidence_excerpt="The answer does not name Kasparro.",
            char_offset_start=0,
            char_offset_end=33,
            has_kasparro_citation=False,
            citation_count_for_response=0,
            unique_citation_count=0,
            execution_mode=ExecutionMode.STUB,
            model_name="stub-v1",
        ),
        ResponseResult(
            run_id=run.id,
            response_id="r4",
            prompt_id="p2",
            engine=Engine.CHATGPT,
            mention=True,
            recommendation=True,
            negative_characterisation=False,
            recommendation_position=3,
            competitors_named=["otterly ai", "peec ai"],
            evidence_excerpt="Kasparro is third on our list.",
            char_offset_start=0,
            char_offset_end=27,
            has_kasparro_citation=False,
            citation_count_for_response=0,
            unique_citation_count=0,
            execution_mode=ExecutionMode.STUB,
            model_name="stub-v1",
        ),
    ]
    session.add_all(rows)
    await session.flush()

    score = await compute_visibility_score(session, run.id)

    # Hand-computed:
    # mention_rate = 3/4 = 0.75
    # recommendation_rate = 2/4 = 0.50
    # negative_rate = 1/4 = 0.25 → (1 - 0.25) = 0.75
    # citation_rate = 1/4 = 0.25
    # position_score: positions = [1, None, None, 3] → for recommendation=True only → [1, 3]
    #   _position_score([1, 3]) = (1.0 + 0.6) / 2 = 0.8
    # composite = 100 * (0.30*0.75 + 0.25*0.50 + 0.15*0.75 + 0.15*0.25 + 0.15*0.8)
    #           = 100 * (0.225 + 0.125 + 0.1125 + 0.0375 + 0.12) = 100 * 0.62 = 62.0
    assert score.mention_rate == pytest.approx(0.75)
    assert score.recommendation_rate == pytest.approx(0.50)
    assert score.negative_rate == pytest.approx(0.25)
    assert score.citation_rate == pytest.approx(0.25)
    assert score.composite == pytest.approx(62.0, abs=0.5)
    assert score.composite_ci_low <= score.composite <= score.composite_ci_high
    assert score.denominator_valid == 4
    assert score.denominator_total == 4
    assert score.failed_count == 0


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

    # One valid mention
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
            competitors_named=[],
            evidence_excerpt="Kasparro is recommended",
            char_offset_start=0,
            char_offset_end=23,
            has_kasparro_citation=True,
            citation_count_for_response=1,
            unique_citation_count=1,
            execution_mode=ExecutionMode.STUB,
        )
    )
    # Two failed — must not count toward denominators
    for rid in ("r2", "r3"):
        session.add(
            ResponseResult(
                run_id=run.id,
                response_id=rid,
                prompt_id="p2",
                engine=Engine.GEMINI,
                failed=True,
                failure_reason="simulated timeout",
                execution_mode=ExecutionMode.STUB,
            )
        )
    await session.flush()

    score = await compute_visibility_score(session, run.id)
    # Valid denominator = 1, total = 3, failed = 2
    assert score.denominator_valid == 1
    assert score.denominator_total == 3
    assert score.failed_count == 2
    # Mention rate over valid = 1/1 = 1.0 (failed NOT counted as zero-mention)
    assert score.mention_rate == pytest.approx(1.0)
