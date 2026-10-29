"""Tests for the new submit gate: remedy→diagnostic id validator + citation cases."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import select

from kasparro_auditor.db.models import (
    AnalysisRun,
    Engine,
    ExecutionMode,
    ResponseResult,
    ResponseRow,
    PromptRow,
    Dataset,
)


async def _setup_run_with_responses(session, n_responses: int = 5) -> tuple[str, str]:
    """Create a Dataset + AnalysisRun + n successful ResponseResults.

    Each response uses a different (prompt_id, engine) pair to satisfy the
    UNIQUE constraint on responses.
    """
    ds = Dataset(run_id="test-run", source_dir="/tmp")
    session.add(ds)
    await session.flush()
    # Add n prompts
    for i in range(n_responses):
        p = PromptRow(prompt_id=f"p{i}", run_id="test-run", text=f"does kasparro work? ({i})",
                      content_hash=f"ph{i}", raw_json="{}")
        session.add(p)
    await session.flush()
    # Add n responses — each with a unique (prompt_id, engine) pair
    engines = [Engine.CHATGPT, Engine.GEMINI, Engine.AIMODE]
    for i in range(n_responses):
        r = ResponseRow(
            response_id=f"r{i}",
            run_id="test-run",
            prompt_id=f"p{i}",
            engine=engines[i % 3],
            answer_text=f"Kasparro is mentioned here. Some answer text {i}.",
            capture_status="captured",
            archive_stored_at="2026-07-31T04:06:40+00:00",
            content_hash=f"hash{i}",
            raw_json="{}",
            dataset_id=ds.id,
        )
        session.add(r)
    await session.flush()

    run = AnalysisRun(
        dataset_id=ds.id,
        execution_mode=ExecutionMode.STUB,
        model_name="stub-v1",
        scoring_version="v2",
        category_version="v1",
        status="completed",
        total=n_responses,
        successful=n_responses,
        failed=0,
        skipped=0,
    )
    session.add(run)
    await session.flush()

    # Add ResponseResults for each response
    for i in range(n_responses):
        rr = ResponseResult(
            run_id=run.id,
            response_id=f"r{i}",
            prompt_id=f"p{i}",
            engine=engines[i % 3],
            mention=True,
            recommendation=False,
            negative_characterisation=False,
            execution_mode=ExecutionMode.STUB,
            model_name="stub-v1",
        )
        session.add(rr)
    await session.flush()
    return run.id, ds.id


@pytest.mark.asyncio
async def test_remedy_citing_nonexistent_diagnostic_is_rejected(session):
    """Submit gate rejects a remedy that cites a non-existent diagnostic id."""
    from kasparro_auditor.config import ActorRole
    from kasparro_auditor.reports.state import (
        create_draft,
        submit_version,
    )
    from kasparro_auditor.api.errors import UnsupportedClaimsError

    run_id, _ = await _setup_run_with_responses(session)

    version = await create_draft(
        session,
        analysis_run_id=run_id,
        actor_id="analyst-001",
        title="Test Report",
        findings={"summary": "test"},
        scores={"composite": 50.0},
        diagnostics=[
            {
                "id": "diag_real_1",
                "title": "Real diagnostic 1",
                "pattern": "Pattern 1",
                "hypothesis": "Hypothesis 1",
                "evidence_response_ids": ["r0"],
                "evidence_excerpts": ["Kasparro is mentioned here."],
                "would_change_conclusion": "Something",
            },
            {
                "id": "diag_real_2",
                "title": "Real diagnostic 2",
                "pattern": "Pattern 2",
                "hypothesis": "Hypothesis 2",
                "evidence_response_ids": ["r1"],
                "evidence_excerpts": ["Kasparro is mentioned here."],
                "would_change_conclusion": "Something",
            },
            {
                "id": "diag_real_3",
                "title": "Real diagnostic 3",
                "pattern": "Pattern 3",
                "hypothesis": "Hypothesis 3",
                "evidence_response_ids": ["r2"],
                "evidence_excerpts": ["Kasparro is mentioned here."],
                "would_change_conclusion": "Something",
            },
        ],
        remedies=[
            {
                "id": "remedy_1",
                "priority": 1,
                "problem": "diag_real_1",
                "supporting_evidence": ["diag_real_1"],  # OK — exists
                "action": "Do something",
                "expected_benefit": "Benefit",
                "effort": "low",
                "uncertainty": "low",
                "validation_step": "Step",
            },
            {
                "id": "remedy_2",
                "priority": 2,
                "problem": "diag_real_2",
                "supporting_evidence": ["diag_real_2"],  # OK — exists
                "action": "Do something",
                "expected_benefit": "Benefit",
                "effort": "low",
                "uncertainty": "low",
                "validation_step": "Step",
            },
            {
                "id": "remedy_3_bad",
                "priority": 3,
                "problem": "diag_nonexistent",  # BAD — doesn't exist
                "supporting_evidence": ["diag_nonexistent"],  # BAD — doesn't exist
                "action": "Do something",
                "expected_benefit": "Benefit",
                "effort": "low",
                "uncertainty": "low",
                "validation_step": "Step",
            },
        ],
    )
    await session.flush()

    # Submit should be REJECTED with unsupported_claims because remedy_3_bad
    # cites a diagnostic id that doesn't exist
    with pytest.raises(UnsupportedClaimsError, match="not one of the 3 diagnostic ids"):
        await submit_version(session, version.id, "analyst-001", ActorRole.ANALYST)


@pytest.mark.asyncio
async def test_remedy_with_valid_diagnostic_ids_passes(session):
    """Submit gate accepts remedies that cite existing diagnostic ids."""
    from kasparro_auditor.config import ActorRole
    from kasparro_auditor.reports.state import (
        create_draft,
        submit_version,
    )

    run_id, _ = await _setup_run_with_responses(session)

    version = await create_draft(
        session,
        analysis_run_id=run_id,
        actor_id="analyst-001",
        title="Test Report",
        findings={"summary": "test"},
        scores={"composite": 50.0},
        diagnostics=[
            {
                "id": "diag_1",
                "title": "Diagnostic 1",
                "pattern": "Pattern 1",
                "hypothesis": "Hypothesis 1",
                "evidence_response_ids": ["r0"],
                "evidence_excerpts": ["Kasparro is mentioned here."],
                "would_change_conclusion": "Something",
            },
        ] * 3,  # 3 identical diagnostics for simplicity
        remedies=[
            {
                "id": f"remedy_{i}",
                "priority": i,
                "problem": "diag_1",
                "supporting_evidence": ["diag_1"],  # Valid
                "action": "Do something",
                "expected_benefit": "Benefit",
                "effort": "low",
                "uncertainty": "low",
                "validation_step": "Step",
            }
            for i in (1, 2, 3)
        ],
    )
    await session.flush()

    # Submit should succeed — all remedies cite existing diagnostics
    submitted = await submit_version(session, version.id, "analyst-001", ActorRole.ANALYST)
    assert submitted.status.value == "submitted"
    assert submitted.content_hash is not None
