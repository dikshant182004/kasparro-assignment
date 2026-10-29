"""API request/response schemas.

ReportPatch is RESTRICTED: only title, typed diagnostics, typed remedies,
and notes can be edited. Scores are server-derived and read-only — analysts
cannot PATCH composite to 99.0.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: dict[str, Any]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class DatasetCreate(BaseModel):
    """POST /datasets body. No source_dir — always imports from DATA_PACK_DIR."""

    # No fields: import always uses DATA_PACK_DIR from .env
    # This prevents path traversal and keeps the caller honest.
    pass


class RejectedRecordRead(BaseModel):
    """One rejected row with full context."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    source_file: str
    line_no: int | None
    record_id: str | None
    raw_payload: str
    error: str
    rejection_type: str
    rejected_at: datetime


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    source_dir: str
    prompt_count: int
    response_count: int
    citation_count: int
    rejected_count: int
    imported_at: datetime


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class AnalysisCreate(BaseModel):
    dataset_id: str
    execution_mode: Literal["live", "local", "replay", "stub"] | None = Field(
        default=None,
        description="Override EXECUTION_MODE for this run.",
    )


class AnalysisRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    execution_mode: str
    model_name: str | None
    scoring_version: str | None = None
    category_version: str | None = None
    status: str
    total: int
    successful: int
    failed: int
    skipped: int
    started_at: datetime
    finished_at: datetime | None
    error_summary: str | None
    # The visibility score is populated when status=completed
    score: dict[str, Any] | None = None


class ResponseResultRead(BaseModel):
    """One response result for the paged /analyses/{id}/results endpoint."""

    model_config = ConfigDict(from_attributes=True)

    response_id: str
    prompt_id: str
    engine: str
    mention: bool
    mention_disagreement: bool
    prompt_names_brand: bool
    prompt_category: str | None
    competitors_known: list[str]
    recommendation: bool
    negative_characterisation: bool
    recommendation_position: int | None
    competitors_extra: list[str]
    evidence_mention: str
    evidence_recommendation: str
    evidence_negative: str
    citation_count_for_response: int
    has_kasparro_citation: bool
    composite_score: float
    failed: bool
    skipped: bool
    error_code: str | None
    failure_reason: str | None
    skip_reason: str | None
    execution_mode: str
    model_name: str | None
    attempts: int
    latency_ms: int


class PaginatedResults(BaseModel):
    """Paged response for GET /analyses/{id}/results."""

    items: list[ResponseResultRead]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Reports — TYPED diagnostics + remedies (no more arbitrary dicts)
# ---------------------------------------------------------------------------


class DiagnosticInput(BaseModel):
    """Typed diagnostic for ReportPatch.diagnostics.

    Required fields + every evidence_response_id will be verified against
    the run's results at submit time.
    """

    id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=255)
    pattern: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    evidence_response_ids: list[str] = Field(default_factory=list)
    evidence_excerpts: list[str] = Field(default_factory=list)
    would_change_conclusion: str = Field(min_length=1)


class RemedyInput(BaseModel):
    """Typed remedy for ReportPatch.remedies."""

    id: str = Field(min_length=1, max_length=128)
    priority: int = Field(ge=1, le=10)
    problem: str = Field(min_length=1)
    supporting_evidence: list[str] = Field(default_factory=list)
    action: str = Field(min_length=1)
    expected_benefit: str = Field(min_length=1)
    effort: Literal["low", "medium", "high"]
    uncertainty: Literal["low", "medium", "high"]
    validation_step: str = Field(min_length=1)


class ReportPatch(BaseModel):
    """Analyst edits on a draft version.

    RESTRICTED — only these fields are editable:
      - title
      - diagnostics (typed, validated)
      - remedies (typed, validated)
      - notes

    Scores are NOT editable. They are server-derived from the run's
    ResponseResults and any attempt to PATCH them is silently ignored
    (not stored). This prevents the "PATCH scores to composite: 99.0" bug.
    """

    title: str | None = Field(default=None, min_length=1, max_length=255)
    diagnostics: list[DiagnosticInput] | None = None
    remedies: list[RemedyInput] | None = None
    notes: str | None = Field(default=None, max_length=10000)


class ReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    analysis_run_id: str
    parent_version_id: str | None
    version_number: int
    status: str
    title: str
    findings: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    remedies: list[dict[str, Any]]
    scores: dict[str, Any]
    notes: str
    lock_version: int
    content_hash: str | None
    created_by: str
    created_at: datetime
    submitted_at: datetime | None
    submitted_by: str | None


class DecisionCreate(BaseModel):
    """Reviewer's decision on a submitted version. Comment is required."""

    decision: Literal["approve", "reject"]
    comment: str = Field(min_length=1, max_length=2000)
    # Caller's observed lock_version for optimistic concurrency.
    # Mismatch → 409 concurrent_modification.
    lock_version: int = Field(ge=0)


class DecisionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    version_id: str
    decision: str
    actor_id: str
    actor_role: str
    comment: str
    decided_at: datetime


class ExportRead(BaseModel):
    """The exported approved report (simulated client-report export).

    Includes identity + traceability fields so a reviewer can verify
    what they're exporting:
      - analysis_id: which AnalysisRun produced the data
      - execution_mode + model_name + scoring_version: how it was made
      - model_quality_verified: False for stub/replay, True only for live
      - content_hash: tamper-detection (verified against stored hash)
      - is_latest + superseded_by: is there a newer version?
    """

    version_id: str
    version_number: int
    exported_at: datetime
    approved_by: str
    approved_at: datetime
    title: str
    findings: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    remedies: list[dict[str, Any]]
    scores: dict[str, Any]
    notes: str
    # Identity + traceability
    analysis_id: str
    execution_mode: str
    model_name: str | None
    scoring_version: str | None
    category_version: str | None
    model_quality_verified: bool
    content_hash: str
    is_latest: bool
    superseded_by: str | None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthRead(BaseModel):
    status: str
    version: str
    execution_mode: str
    database_healthy: bool
    # NOTE: database_url is intentionally NOT exposed. URLs may contain
    # credentials and the brief says to keep secrets out of submissions.
