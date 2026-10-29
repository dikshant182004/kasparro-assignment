"""Report workflow endpoints: PATCH /reports/{id}, submit, decision, export."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.api.deps import (
    Actor,
    get_actor,
    require_analyst,
    require_reviewer,
)
from kasparro_auditor.api.errors import api_error
from kasparro_auditor.db.base import get_session
from kasparro_auditor.db.models import AnalysisRun, ReportVersion, ReviewDecision
from kasparro_auditor.reports.diagnostics import build_diagnostics_and_remedies
from kasparro_auditor.reports.state import (
    assert_exportable,
    create_draft,
    decide,
    fork_from_approved,
    patch_draft,
    submit_version,
)
from kasparro_auditor.schemas.api import (
    DecisionCreate,
    DecisionRead,
    ExportRead,
    ReportPatch,
    ReportRead,
)

router = APIRouter(tags=["reports"])


# ---------------------------------------------------------------------------
# Create draft from analysis run
# ---------------------------------------------------------------------------


@router.post("/analyses/{analysis_id}/reports", response_model=ReportRead, status_code=201)
async def create_report_draft(
    analysis_id: str = Path(...),
    actor: Actor = Depends(require_analyst),
    session: AsyncSession = Depends(get_session),
) -> ReportRead:
    """Analyst creates a draft report from a completed analysis run.

    Auto-populates findings/diagnostics/remedies from the analysis.
    """
    run = await session.get(AnalysisRun, analysis_id)
    if run is None:
        raise api_error("not_found", f"AnalysisRun {analysis_id} not found.")
    if run.status != "completed":
        raise api_error(
            "not_found",
            f"AnalysisRun {analysis_id} is not completed (status={run.status}).",
        )

    diagnostics, remedies, score = await build_diagnostics_and_remedies(session, run.id)
    scores_dict = score.model_dump()
    engine_summary = await _group_by_engine_summary(session, run.id)
    findings = {
        "summary": f"Composite visibility score: {score.composite:.2f}/100 "
        f"(95% CI: {score.composite_ci_low:.2f}–{score.composite_ci_high:.2f}). "
        f"Based on {score.denominator_valid}/{score.denominator_total} valid responses "
        f"({score.failed_count} failed, {score.skipped_count} skipped).",
        "by_engine": engine_summary,
    }

    version = await create_draft(
        session,
        analysis_run_id=run.id,
        actor_id=actor.actor_id,
        findings=findings,
        scores=scores_dict,
        diagnostics=diagnostics,
        remedies=remedies,
    )
    await session.commit()
    await session.refresh(version)
    return ReportRead.model_validate(version)


async def _group_by_engine_summary(session: AsyncSession, run_id: str) -> dict[str, dict]:
    """Return per-engine counts."""
    from kasparro_auditor.db.models import ResponseResult

    rows = (
        await session.scalars(
            select(ResponseResult).where(ResponseResult.run_id == run_id)
        )
    ).all()
    out: dict[str, dict] = {}
    for r in rows:
        eng = r.engine.value
        if eng not in out:
            out[eng] = {"total": 0, "mention": 0, "recommendation": 0, "negative": 0, "failed": 0}
        out[eng]["total"] += 1
        if r.failed:
            out[eng]["failed"] += 1
            continue
        if r.mention:
            out[eng]["mention"] += 1
        if r.recommendation:
            out[eng]["recommendation"] += 1
        if r.negative_characterisation:
            out[eng]["negative"] += 1
    return out


# ---------------------------------------------------------------------------
# Patch / submit / decide / export
# ---------------------------------------------------------------------------


@router.patch("/reports/{version_id}", response_model=ReportRead)
async def patch_report(
    body: ReportPatch,
    version_id: str = Path(...),
    actor: Actor = Depends(require_analyst),
    session: AsyncSession = Depends(get_session),
) -> ReportRead:
    """Analyst edits a DRAFT version. Non-draft versions cannot be patched."""
    version = await patch_draft(session, version_id, actor.actor_id, actor.role, body)
    await session.commit()
    await session.refresh(version)
    return ReportRead.model_validate(version)


@router.post("/reports/{version_id}/submit", response_model=ReportRead)
async def submit_report(
    version_id: str = Path(...),
    actor: Actor = Depends(require_analyst),
    session: AsyncSession = Depends(get_session),
) -> ReportRead:
    """Analyst submits the version for review. Version becomes immutable."""
    version = await submit_version(session, version_id, actor.actor_id, actor.role)
    await session.commit()
    await session.refresh(version)
    return ReportRead.model_validate(version)


@router.post("/reports/{version_id}/decision", response_model=DecisionRead)
async def decide_report(
    body: DecisionCreate,
    version_id: str = Path(...),
    actor: Actor = Depends(require_reviewer),
    session: AsyncSession = Depends(get_session),
) -> DecisionRead:
    """Reviewer approves or rejects a submitted version with a comment."""
    decision_row = await decide(session, version_id, actor.actor_id, actor.role, body)
    await session.commit()
    await session.refresh(decision_row)
    return DecisionRead.model_validate(decision_row)


@router.get("/reports/{version_id}/export", response_model=ExportRead)
async def export_report(
    version_id: str = Path(...),
    actor: Actor = Depends(get_actor),  # either role may export
    session: AsyncSession = Depends(get_session),
) -> ExportRead:
    """Export an APPROVED version. Refuses draft/submitted/rejected/stale."""
    version = await session.get(ReportVersion, version_id)
    if version is None:
        raise api_error("not_found", f"ReportVersion {version_id} not found.")
    await assert_exportable(version)

    # Find the most recent decision on this version.
    decision = (
        await session.scalars(
            select(ReviewDecision)
            .where(ReviewDecision.version_id == version_id)
            .order_by(ReviewDecision.decided_at.desc())
        )
    ).first()

    return ExportRead(
        version_id=version.id,
        version_number=version.version_number,
        exported_at=datetime.now(timezone.utc),
        approved_by=decision.actor_id if decision else "unknown",
        approved_at=decision.decided_at if decision else version.submitted_at or datetime.now(timezone.utc),
        title=version.title,
        findings=version.findings,
        diagnostics=version.diagnostics,
        remedies=version.remedies,
        scores=version.scores,
    )


# ---------------------------------------------------------------------------
# Fork (edit-after-approval flow)
# ---------------------------------------------------------------------------


@router.post("/reports/{version_id}/fork", response_model=ReportRead, status_code=201)
async def fork_report(
    version_id: str = Path(...),
    actor: Actor = Depends(require_analyst),
    session: AsyncSession = Depends(get_session),
) -> ReportRead:
    """Fork a new draft from a submitted/approved/rejected version.

    Implements: 'Editing a submitted or approved report creates a new
    draft and requires a new review.'
    """
    new_draft = await fork_from_approved(session, version_id, actor.actor_id, actor.role)
    await session.commit()
    await session.refresh(new_draft)
    return ReportRead.model_validate(new_draft)


@router.get("/reports/{version_id}", response_model=ReportRead)
async def get_report(
    version_id: str = Path(...),
    actor: Actor = Depends(get_actor),
    session: AsyncSession = Depends(get_session),
) -> ReportRead:
    version = await session.get(ReportVersion, version_id)
    if version is None:
        raise api_error("not_found", f"ReportVersion {version_id} not found.")
    return ReportRead.model_validate(version)
