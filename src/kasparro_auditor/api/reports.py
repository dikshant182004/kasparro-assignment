"""Report workflow endpoints: PATCH, submit, decision, fork, export.

Plus the new paged results endpoint and the dedicated score endpoint.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.api.deps import (
    Actor,
    get_actor,
    require_analyst,
    require_reviewer,
)
from kasparro_auditor.api.errors import (
    InsufficientDataError,
    NotFoundError,
    WrongRoleError,
)
from kasparro_auditor.db.base import get_session
from kasparro_auditor.db.models import (
    AnalysisRun,
    ResponseResult,
    ReviewDecision,
    ReportVersion,
)
from kasparro_auditor.reports.diagnostics import build_diagnostics_and_remedies
from kasparro_auditor.reports.state import (
    assert_exportable,
    create_draft,
    decide,
    find_latest_version,
    find_superseding_version,
    fork_from_approved,
    patch_draft,
    submit_version,
)
from kasparro_auditor.schemas.api import (
    DecisionCreate,
    DecisionRead,
    ExportRead,
    PaginatedResults,
    ReportPatch,
    ReportRead,
    ResponseResultRead,
)
from kasparro_auditor.scoring.score import compute_visibility_score

router = APIRouter(tags=["reports"])


# ---------------------------------------------------------------------------
# New endpoints: GET /analyses/{id}/results (paged) + GET /analyses/{id}/score
# ---------------------------------------------------------------------------


@router.get("/analyses/{analysis_id}/results", response_model=PaginatedResults)
async def list_results(
    analysis_id: str = Path(...),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    actor: Actor = Depends(get_actor),
    session: AsyncSession = Depends(get_session),
) -> PaginatedResults:
    """List per-response results for an analysis run, paged.

    Lets the reviewer inspect failures, evidence, and traceability per
    response_id without paging through the whole DB.
    """
    run = await session.get(AnalysisRun, analysis_id)
    if run is None:
        raise NotFoundError(f"AnalysisRun {analysis_id} not found.")

    total = await session.scalar(
        select(func.count()).select_from(ResponseResult).where(
            ResponseResult.run_id == analysis_id
        )
    ) or 0

    rows = (
        await session.scalars(
            select(ResponseResult)
            .where(ResponseResult.run_id == analysis_id)
            .order_by(ResponseResult.prompt_id, ResponseResult.engine)
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return PaginatedResults(
        items=[ResponseResultRead.model_validate(r) for r in rows],
        total=int(total),
        limit=limit,
        offset=offset,
    )


@router.get("/analyses/{analysis_id}/score")
async def get_score(
    analysis_id: str = Path(...),
    actor: Actor = Depends(get_actor),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Return the VisibilityScore dict for this analysis run."""
    run = await session.get(AnalysisRun, analysis_id)
    if run is None:
        raise NotFoundError(f"AnalysisRun {analysis_id} not found.")
    if run.status != "completed":
        raise NotFoundError(
            f"AnalysisRun {analysis_id} is not completed (status={run.status})."
        )
    score = await compute_visibility_score(session, run.id)
    return score.model_dump()


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

    Auto-populates findings/diagnostics/remedies/scores from the analysis.
    Refuses to create a draft from a run with 0 successful results
    (insufficient_data → 409).
    """
    run = await session.get(AnalysisRun, analysis_id)
    if run is None:
        raise NotFoundError(f"AnalysisRun {analysis_id} not found.")

    # Only attempt LLM diagnostics synthesis in live mode. Stub/replay
    # modes have no LLM client, so they'd fail and fall back to rule-based
    # with a misleading 'rule_llm_failed' flag.
    use_llm = run.execution_mode.value == "live"
    diagnostics, remedies, score, diag_source = await build_diagnostics_and_remedies(
        session, run.id, use_llm=use_llm
    )
    scores_dict = score.model_dump()
    engine_summary = await _group_by_engine_summary(session, run.id)
    # Convert per_category CategorySlice objects to plain dicts (JSON-serializable)
    per_category_plain = {
        k: v.model_dump() if hasattr(v, "model_dump") else v
        for k, v in score.per_category.items()
    }
    findings = {
        "summary": (
            f"Composite visibility score: {score.composite if score.composite is not None else 'N/A'}/100 "
            f"(95% CI: {score.composite_ci_low if score.composite_ci_low is not None else 'N/A'}–"
            f"{score.composite_ci_high if score.composite_ci_high is not None else 'N/A'}). "
            f"Based on {score.denominator_valid}/{score.denominator_total} valid responses "
            f"({score.failed_count} failed, {score.skipped_count} skipped)."
        ),
        "by_engine": engine_summary,
        "per_category": per_category_plain,
        "diagnostics_source": diag_source,
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
    """Analyst edits a DRAFT version.

    Only title, diagnostics (typed), remedies (typed), and notes are editable.
    Scores are server-derived and read-only — attempts to PATCH scores
    are silently ignored (the field isn't in ReportPatch).
    """
    version = await patch_draft(session, version_id, actor.actor_id, actor.role, body)
    await session.commit()
    await session.refresh(version)
    return ReportRead.model_validate(version)


@router.post("/reports/{version_id}/submit", response_model=ReportRead)
async def submit_report(
    version_id: str = Path(...),
    caller_lock_version: int | None = None,
    actor: Actor = Depends(require_analyst),
    session: AsyncSession = Depends(get_session),
) -> ReportRead:
    """Analyst submits the version for review. Version becomes immutable.

    Submit gate:
      - Exactly 3 diagnostics + 3 remedies
      - Every evidence response_id must exist in the run
      - Every excerpt must be verbatim in the answer text
      - content_hash is computed and stored
    """
    version = await submit_version(
        session, version_id, actor.actor_id, actor.role, caller_lock_version
    )
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
    """Reviewer approves or rejects a submitted version with a comment.

    Optimistic locking: caller must send their observed lock_version.
    Mismatch → 409 concurrent_modification (catches two reviewers at once).
    """
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
    """Export an APPROVED version. Refuses draft/submitted/rejected/stale.

    Verifies content_hash to detect tampering between submit and export.
    Refuses if no ReviewDecision row exists (inconsistent approval state).
    Returns identity + traceability fields so the reviewer can verify
    what they're exporting.
    """
    version = await session.get(ReportVersion, version_id)
    if version is None:
        raise NotFoundError(f"ReportVersion {version_id} not found.")
    await assert_exportable(session, version)

    decision = (
        await session.scalars(
            select(ReviewDecision)
            .where(ReviewDecision.version_id == version_id)
            .order_by(ReviewDecision.decided_at.desc())
        )
    ).first()

    run = await session.get(AnalysisRun, version.analysis_run_id)

    # Determine is_latest + superseded_by
    latest = await find_latest_version(session, version.analysis_run_id)
    is_latest = (latest is not None and latest.id == version.id)
    superseded_by = await find_superseding_version(session, version)

    # model_quality_verified: True only for live calls. Stub/replay outputs
    # don't establish live-model correctness (per brief).
    actual_mode = version.scores.get("scoring_version", "v2")  # placeholder
    model_quality_verified = (
        run is not None and run.execution_mode.value == "live"
    )

    return ExportRead(
        version_id=version.id,
        version_number=version.version_number,
        exported_at=datetime.now(timezone.utc),
        approved_by=decision.actor_id if decision else "unknown",
        approved_at=decision.decided_at if decision else (
            version.submitted_at or datetime.now(timezone.utc)
        ),
        title=version.title,
        findings=version.findings,
        diagnostics=version.diagnostics,
        remedies=version.remedies,
        scores=version.scores,
        notes=version.notes or "",
        analysis_id=version.analysis_run_id,
        execution_mode=run.execution_mode.value if run else "unknown",
        model_name=run.model_name if run else None,
        scoring_version=run.scoring_version if run else None,
        category_version=run.category_version if run else None,
        model_quality_verified=model_quality_verified,
        content_hash=version.content_hash or "",
        is_latest=is_latest,
        superseded_by=superseded_by,
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

    Forking from a DRAFT is forbidden — that would create parallel drafts.
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
        raise NotFoundError(f"ReportVersion {version_id} not found.")
    return ReportRead.model_validate(version)
