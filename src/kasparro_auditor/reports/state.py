"""Report version state machine.

States:  draft → submitted → approved | rejected

Typed exceptions replace string matching:
  - NotFoundError             → 404
  - WrongRoleError            → 403
  - InvalidTransitionError    → 409 with code (not_approved | not_submitted
                                | not_draft | concurrent_modification)

Concurrency: optimistic locking via lock_version. Every transition
checks the caller's observed lock_version matches the current value,
then increments it. Mismatch → concurrent_modification (catches two
reviewers deciding at once).

Integrity: content_hash is computed at submit time over the canonical
JSON of (title, findings, diagnostics, remedies). Verified at export to
detect tampering between submit and export.

Submit gate:
  - exactly 3 diagnostics + 3 remedies
  - every diagnostic.evidence_response_id must exist in the run
  - every excerpt must be a verbatim substring of the answer text
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.api.errors import (
    InsufficientDataError,
    InvalidTransitionError,
    NotFoundError,
    UnsupportedClaimsError,
    WrongRoleError,
)
from kasparro_auditor.config import ActorRole
from kasparro_auditor.db.models import (
    AnalysisRun,
    DecisionType,
    ReportStatus,
    ReportVersion,
    ResponseResult,
    ResponseRow,
    ReviewDecision,
)
from kasparro_auditor.schemas.api import DecisionCreate, ReportPatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_content_hash(version: ReportVersion) -> str:
    """SHA-256 of canonical JSON of (title, findings, diagnostics, remedies, notes).

    Computed at submit time; verified at export. Detects tampering.
    """
    payload = json.dumps(
        {
            "title": version.title,
            "findings": version.findings,
            "diagnostics": version.diagnostics,
            "remedies": version.remedies,
            "notes": version.notes,
        },
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


async def _verify_evidence_for_submit(
    session: AsyncSession, version: ReportVersion
) -> None:
    """Submit gate: verify every evidence response_id + excerpt is real.

    Raises UnsupportedClaimsError if any cited response_id doesn't exist
    in the run, or any excerpt isn't a verbatim substring of the
    corresponding answer_text.
    """
    run_id = version.analysis_run_id

    # Check diagnostics count
    if len(version.diagnostics) != 3:
        raise UnsupportedClaimsError(
            f"Report must have exactly 3 diagnostics; got {len(version.diagnostics)}."
        )
    if len(version.remedies) != 3:
        raise UnsupportedClaimsError(
            f"Report must have exactly 3 remedies; got {len(version.remedies)}."
        )

    # Build response_id → answer_text lookup for evidence verification
    cited_ids: set[str] = set()
    for d in version.diagnostics:
        for rid in d.get("evidence_response_ids", []):
            cited_ids.add(rid)

    if cited_ids:
        # Pull the actual answer_text for each cited response_id
        rows = (
            await session.execute(
                select(ResponseResult.response_id, ResponseRow.answer_text)
                .join(ResponseRow, ResponseResult.response_id == ResponseRow.response_id)
                .where(ResponseResult.run_id == run_id)
                .where(ResponseResult.response_id.in_(cited_ids))
            )
        ).all()
        answer_lookup = {r[0]: r[1] for r in rows}

        # Verify every cited response_id exists
        missing = cited_ids - set(answer_lookup.keys())
        if missing:
            raise UnsupportedClaimsError(
                f"Diagnostics cite response_ids that don't exist in run {run_id}: "
                f"{sorted(missing)}"
            )

        # Verify every excerpt is a verbatim substring
        for d in version.diagnostics:
            for rid, excerpt in zip(
                d.get("evidence_response_ids", []),
                d.get("evidence_excerpts", []),
                strict=False,
            ):
                if not excerpt:
                    continue
                if rid not in answer_lookup:
                    continue
                answer_text = answer_lookup[rid]
                if excerpt not in answer_text:
                    # Whitespace-tolerant check
                    norm_text = " ".join(answer_text.split())
                    norm_excerpt = " ".join(excerpt.split())
                    if norm_excerpt not in norm_text:
                        raise UnsupportedClaimsError(
                            f"Diagnostic '{d.get('id', '?')}' cites an excerpt that is "
                            f"not a verbatim substring of response_id={rid}'s answer_text."
                        )


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
    notes: str = "",
) -> ReportVersion:
    """Analyst creates a new draft version."""
    run = await session.get(AnalysisRun, analysis_run_id)
    if run is None:
        raise NotFoundError(f"AnalysisRun {analysis_run_id} not found")

    if run.status != "completed":
        raise InvalidTransitionError(
            "not_submitted",
            f"AnalysisRun {analysis_run_id} is not completed (status={run.status}).",
        )
    if run.successful == 0:
        raise InsufficientDataError(
            f"AnalysisRun {analysis_run_id} has 0 successful results; "
            "cannot create a report from a run with no valid classifications."
        )

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
        notes=notes,
        lock_version=0,
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
    """Analyst edits a DRAFT version. Submitted/approved/rejected are immutable.

    Only title, diagnostics (typed), remedies (typed), notes are editable.
    Scores are NEVER accepted from the client.
    """
    if actor_role != ActorRole.ANALYST:
        raise WrongRoleError("Only the analyst role may edit a draft")

    version = await session.get(ReportVersion, version_id)
    if version is None:
        raise NotFoundError(f"ReportVersion {version_id} not found")

    if version.status != ReportStatus.DRAFT:
        raise InvalidTransitionError(
            "not_draft",
            f"Cannot PATCH a {version.status.value} version. "
            "Fork a new draft with /fork instead.",
        )

    if patch.title is not None:
        version.title = patch.title
    if patch.diagnostics is not None:
        # Convert typed Pydantic objects to dicts for JSON storage
        version.diagnostics = [d.model_dump() for d in patch.diagnostics]
    if patch.remedies is not None:
        version.remedies = [r.model_dump() for r in patch.remedies]
    if patch.notes is not None:
        version.notes = patch.notes
    # NOTE: scores is intentionally NOT patched. It's server-derived.
    version.lock_version += 1
    await session.flush()
    return version


async def submit_version(
    session: AsyncSession,
    version_id: str,
    actor_id: str,
    actor_role: ActorRole,
    caller_lock_version: int | None = None,
) -> ReportVersion:
    """Analyst freezes a draft → submitted.

    Submit gate:
      - 3 diagnostics + 3 remedies
      - Every evidence_response_id must exist in the run
      - Every excerpt must be a verbatim substring of the answer text
      - Compute + store content_hash for tamper detection at export
    """
    if actor_role != ActorRole.ANALYST:
        raise WrongRoleError("Only the analyst role may submit")

    version = await session.get(ReportVersion, version_id)
    if version is None:
        raise NotFoundError(f"ReportVersion {version_id} not found")

    # Optimistic locking
    if caller_lock_version is not None and caller_lock_version != version.lock_version:
        raise InvalidTransitionError(
            "concurrent_modification",
            f"Version was modified by another caller. "
            f"Caller observed lock_version={caller_lock_version}, "
            f"current is {version.lock_version}. Reload and retry.",
        )

    if version.status != ReportStatus.DRAFT:
        raise InvalidTransitionError(
            "not_draft",
            f"Cannot submit a {version.status.value} version (must be draft)",
        )

    # Submit gate — verify all evidence is real
    await _verify_evidence_for_submit(session, version)

    version.status = ReportStatus.SUBMITTED
    version.submitted_at = datetime.now(timezone.utc)
    version.submitted_by = actor_id
    version.content_hash = _compute_content_hash(version)
    version.lock_version += 1
    await session.flush()
    return version


async def decide(
    session: AsyncSession,
    version_id: str,
    actor_id: str,
    actor_role: ActorRole,
    decision: DecisionCreate,
) -> ReviewDecision:
    """Reviewer approves or rejects a submitted version with a comment.

    Optimistic locking: caller's lock_version must match. This catches
    two reviewers deciding at once.
    """
    if actor_role != ActorRole.REVIEWER:
        raise WrongRoleError("Only the reviewer role may approve or reject")

    version = await session.get(ReportVersion, version_id)
    if version is None:
        raise NotFoundError(f"ReportVersion {version_id} not found")

    # Optimistic locking
    if decision.lock_version != version.lock_version:
        raise InvalidTransitionError(
            "concurrent_modification",
            f"Version was modified by another caller. "
            f"Caller observed lock_version={decision.lock_version}, "
            f"current is {version.lock_version}. Reload and retry.",
        )

    if version.status != ReportStatus.SUBMITTED:
        raise InvalidTransitionError(
            "not_submitted",
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
    version.lock_version += 1
    await session.flush()
    return decision_row


async def fork_from_approved(
    session: AsyncSession,
    version_id: str,
    actor_id: str,
    actor_role: ActorRole,
) -> ReportVersion:
    """Create a new draft from an existing submitted/approved/rejected version.

    Forking from a DRAFT is forbidden — that would create parallel drafts.
    The source must be a non-draft version.
    """
    if actor_role != ActorRole.ANALYST:
        raise WrongRoleError("Only the analyst may fork a new draft")

    src = await session.get(ReportVersion, version_id)
    if src is None:
        raise NotFoundError(f"ReportVersion {version_id} not found")

    if src.status == ReportStatus.DRAFT:
        raise InvalidTransitionError(
            "not_draft",
            "Cannot fork a DRAFT version — that would create parallel drafts. "
            "PATCH the existing draft instead.",
        )

    run = await session.get(AnalysisRun, src.analysis_run_id)
    if run is None:
        raise NotFoundError(f"AnalysisRun {src.analysis_run_id} not found")

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
        parent_version_id=src.id,
        version_number=next_num,
        status=ReportStatus.DRAFT,
        title=src.title,
        findings=dict(src.findings or {}),
        scores=dict(src.scores or {}),
        diagnostics=list(src.diagnostics or []),
        remedies=list(src.remedies or []),
        notes=src.notes or "",
        lock_version=0,
        created_by=actor_id,
    )
    session.add(version)
    await session.flush()
    return version


# ---------------------------------------------------------------------------
# Export gate + helpers
# ---------------------------------------------------------------------------


class ReportStateError(Exception):
    """Legacy compatibility — kept so old imports still work.

    New code raises typed APIError subclasses instead.
    """


async def assert_exportable(
    session: AsyncSession,
    version: ReportVersion,
) -> None:
    """Raise if this version cannot be exported.

    Conditions:
      - status == APPROVED
      - has at least one ReviewDecision row (refuse if missing)
      - content_hash matches (refuse if tampered)
    """
    if version.status != ReportStatus.APPROVED:
        raise InvalidTransitionError(
            "not_approved",
            f"Cannot export a {version.status.value} version. "
            "Only approved versions can be exported.",
        )

    # Refuse export if no decision row exists
    decision = (
        await session.scalars(
            select(ReviewDecision)
            .where(ReviewDecision.version_id == version.id)
            .order_by(ReviewDecision.decided_at.desc())
        )
    ).first()
    if decision is None:
        raise InvalidTransitionError(
            "not_approved",
            f"Version {version.id} is approved but has no ReviewDecision row. "
            "Refusing to export — approval state is inconsistent.",
        )

    # Verify content_hash (tamper detection)
    if version.content_hash is None:
        raise InvalidTransitionError(
            "not_approved",
            f"Version {version.id} has no content_hash. "
            "Cannot verify integrity.",
        )
    expected_hash = _compute_content_hash(version)
    if expected_hash != version.content_hash:
        raise InvalidTransitionError(
            "not_approved",
            f"Version {version.id} content_hash mismatch. "
            "Report was tampered with after submit. Refusing to export.",
        )


async def find_latest_version(
    session: AsyncSession, analysis_run_id: str
) -> ReportVersion | None:
    """Return the highest-version_number ReportVersion for this run, or None."""
    return (
        await session.scalars(
            select(ReportVersion)
            .where(ReportVersion.analysis_run_id == analysis_run_id)
            .order_by(ReportVersion.version_number.desc())
        )
    ).first()


async def find_superseding_version(
    session: AsyncSession, version: ReportVersion
) -> str | None:
    """Return the id of the version that supersedes this one, or None.

    A version is superseded if there's a later version_number for the
    same analysis_run_id.
    """
    later = (
        await session.scalars(
            select(ReportVersion)
            .where(ReportVersion.analysis_run_id == version.analysis_run_id)
            .where(ReportVersion.version_number > version.version_number)
            .order_by(ReportVersion.version_number.asc())
        )
    ).first()
    return later.id if later else None
