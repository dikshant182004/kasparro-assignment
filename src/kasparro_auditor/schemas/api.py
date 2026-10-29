"""API request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kasparro_auditor.schemas.results import Diagnostic, Remedy, VisibilityScore


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error: dict[str, Any]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class DatasetCreate(BaseModel):
    """Optional request body for POST /datasets. Defaults to the local pack."""

    source_dir: str | None = Field(
        default=None,
        description="Path to the extracted assignment pack. Defaults to DATA_PACK_DIR.",
    )


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
    execution_mode: str | None = Field(
        default=None,
        description="Override the EXECUTION_MODE env var for this run (live/replay/stub).",
    )


class AnalysisRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    execution_mode: str
    model_name: str | None
    status: str
    total: int
    successful: int
    failed: int
    skipped: int
    started_at: datetime
    finished_at: datetime | None
    error_summary: str | None


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


class ReportPatch(BaseModel):
    """Analyst edits on a draft version. Only fields the analyst may change."""

    title: str | None = None
    findings: dict[str, Any] | None = None
    diagnostics: list[dict[str, Any]] | None = None
    remedies: list[dict[str, Any]] | None = None
    scores: dict[str, Any] | None = None


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
    created_by: str
    created_at: datetime
    submitted_at: datetime | None
    submitted_by: str | None


class DecisionCreate(BaseModel):
    """Reviewer's decision on a submitted version."""

    decision: str = Field(pattern="^(approve|reject)$")
    comment: str = Field(default="", max_length=2000)


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
    """The exported approved report (simulated client-report export)."""

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


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthRead(BaseModel):
    status: str
    version: str
    execution_mode: str
    database_url: str  # safe to expose (sqlite path)
