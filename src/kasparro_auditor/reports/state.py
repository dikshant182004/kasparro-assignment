"""Report version state machine.

States:  draft → submitted → approved | rejected

Rules enforced:
  * PATCH on a non-draft version is forbidden — must fork a new draft.
  * Only the reviewer can submit a decision.
  * Submitting freezes the version (immutable).
  * Edit-after-approval forks a new draft with parent_version_id pointing back.
  * Old approval authorises export only of that specific immutable version.
  * Rejection blocks export.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.config import ActorRole
from kasparro_auditor.db.models import (
    AnalysisRun,
    DecisionType,
    ReportStatus,
    ReportVersion,
    ReviewDecision,
)
from kasparro_auditor.schemas.api import DecisionCreate, ReportPatch


class ReportStateError(Exception):
    """Raised on an illegal state transition or permission violation."""


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


async def create_draft(
    session: AsyncSession,
    analysis_run_id: str,
    actor_id: str,
    title: str = "Kasparro AI Visibility Report",
    findings: dict | None = None,
    scores: dict | None = None,
    diagnostics: list | None = None,
    remedies: list | None = None,
) -> ReportVersion:
    """Analyst creates a new draft version."""
    # Confirm the analysis run exists.
    run = await session.get(AnalysisRun, analysis_run_id)
    if run is None:
        raise ReportStateError(f"AnalysisRun {analysis_run_id} not found")

    # Next version_number for this analysis run.
    existing = (
        await session.scalars(
            select(ReportVersion)
            .where(ReportVersion.analysis_run_id == analysis_run_id)
            .order_by(ReportVersion.version_number.desc())
        )
    ).first()
    next_num = (existing.version_number + 1) if existing else 1

    version = ReportVersion(
        analysis_run_id=analysis_run_id,
        version_number=next_num,
        status=ReportStatus.DRAFT,
        title=title,
        findings=findings or {},
        scores=scores or {},
        diagnostics=diagnostics or [],
        remedies=remedies or [],
        created_by=actor_id,
    )
    session.add(version)
    await session.flush()
    return version


async def patch_draft(
    session: AsyncSession,
    version_id: str,
    actor_id: str,
    actor_role: ActorRole,
    patch: ReportPatch,
) -> ReportVersion:
    """Analyst edits a DRAFT version. Submitted/approved/rejected are immutable."""
    if actor_role != ActorRole.ANALYST:
        raise ReportStateError("Only the analyst role may edit a draft")

    version = await session.get(ReportVersion, version_id)
    if version is None:
        raise ReportStateError(f"ReportVersion {version_id} not found")

    if version.status != ReportStatus.DRAFT:
        raise ReportStateError(
            f"Cannot PATCH a {version.status.value} version. "
            "Fork a new draft with fork_from_approved() instead.",
        )

    if patch.title is not None:
        version.title = patch.title
    if patch.findings is not None:
        version.findings = patch.findings
    if patch.diagnostics is not None:
        version.diagnostics = patch.diagnostics
    if patch.remedies is not None:
        version.remedies = patch.remedies
    if patch.scores is not None:
        version.scores = patch.scores
    await session.flush()
    return version


async def submit_version(
    session: AsyncSession,
    version_id: str,
    actor_id: str,
    actor_role: ActorRole,
) -> ReportVersion:
    """Analyst freezes a draft → submitted. Becomes immutable for edits."""
    if actor_role != ActorRole.ANALYST:
        raise ReportStateError("Only the analyst role may submit")

    version = await session.get(ReportVersion, version_id)
    if version is None:
        raise ReportStateError(f"ReportVersion {version_id} not found")
    if version.status != ReportStatus.DRAFT:
        raise ReportStateError(
            f"Cannot submit a {version.status.value} version (must be draft)",
        )
    version.status = ReportStatus.SUBMITTED
    version.submitted_at = datetime.now(timezone.utc)
    version.submitted_by = actor_id
    await session.flush()
    return version


async def decide(
    session: AsyncSession,
    version_id: str,
    actor_id: str,
    actor_role: ActorRole,
    decision: DecisionCreate,
) -> ReviewDecision:
    """Reviewer approves or rejects a submitted version with a comment."""
    if actor_role != ActorRole.REVIEWER:
        raise ReportStateError("Only the reviewer role may approve or reject")

    version = await session.get(ReportVersion, version_id)
    if version is None:
        raise ReportStateError(f"ReportVersion {version_id} not found")
    if version.status != ReportStatus.SUBMITTED:
        raise ReportStateError(
            f"Cannot decide a {version.status.value} version (must be submitted)",
        )

    dec_type = DecisionType(decision.decision)
    decision_row = ReviewDecision(
        version_id=version_id,
        decision=dec_type,
        actor_id=actor_id,
        actor_role="reviewer",
        comment=decision.comment,
    )
    session.add(decision_row)
    version.status = (
        ReportStatus.APPROVED if dec_type == DecisionType.APPROVE else ReportStatus.REJECTED
    )
    await session.flush()
    return decision_row


async def fork_from_approved(
    session: AsyncSession,
    version_id: str,
    actor_id: str,
    actor_role: ActorRole,
) -> ReportVersion:
    """Create a new draft from an existing (submitted/approved/rejected) version.

    Implements the brief rule: "Editing a submitted or approved report
    creates a new draft and requires a new review."
    """
    if actor_role != ActorRole.ANALYST:
        raise ReportStateError("Only the analyst may fork a new draft")

    src = await session.get(ReportVersion, version_id)
    if src is None:
        raise ReportStateError(f"ReportVersion {version_id} not found")

    # Confirm the analysis run exists.
    run = await session.get(AnalysisRun, src.analysis_run_id)
    if run is None:
        raise ReportStateError(f"AnalysisRun {src.analysis_run_id} not found")

    # Next version_number for this analysis run.
    existing = (
        await session.scalars(
            select(ReportVersion)
            .where(ReportVersion.analysis_run_id == src.analysis_run_id)
            .order_by(ReportVersion.version_number.desc())
        )
    ).first()
    next_num = (existing.version_number + 1) if existing else 1

    version = ReportVersion(
        analysis_run_id=src.analysis_run_id,
        parent_version_id=src.id,  # link back to the source version
        version_number=next_num,
        status=ReportStatus.DRAFT,
        title=src.title,
        findings=dict(src.findings or {}),
        scores=dict(src.scores or {}),
        diagnostics=list(src.diagnostics or []),
        remedies=list(src.remedies or []),
        created_by=actor_id,
    )
    session.add(version)
    await session.flush()
    return version


# ---------------------------------------------------------------------------
# Export gate
# ---------------------------------------------------------------------------


async def assert_exportable(version: ReportVersion) -> None:
    """Raise if this version is not approved and therefore not exportable."""
    if version.status != ReportStatus.APPROVED:
        raise ReportStateError(
            f"Cannot export a {version.status.value} version. "
            "Only approved versions can be exported.",
        )
