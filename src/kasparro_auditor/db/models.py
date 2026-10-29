"""SQLAlchemy ORM entities.

Mapping to the assignment's suggested entities (class names are illustrative
per the brief — the *contract* matters, not the names):

    Dataset           — an imported pack (one row per import)
    PromptRow         — one prompt (50 expected)
    ResponseRow       — one archived answer (150 expected; unique by response_id)
    CitationRow       — one citation occurrence (2439 expected)
    AnalysisRun       — one execution of the pipeline over a Dataset
    ResponseResult    — classification + evidence for one ResponseRow
    ReportVersion     — immutable-on-submit report state (draft/submitted/approved/rejected)
    ReviewDecision    — reviewer's approve/reject on a submitted version
    RejectedRecord    — invalid input rows preserved (one bad row never kills the batch)
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.sqlite import JSON

from kasparro_auditor.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Engine(str, enum.Enum):
    AIMODE = "aimode"
    CHATGPT = "chatgpt"
    GEMINI = "gemini"


class ReportStatus(str, enum.Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"


class DecisionType(str, enum.Enum):
    APPROVE = "approve"
    REJECT = "reject"


class ExecutionMode(str, enum.Enum):
    LIVE = "live"
    REPLAY = "replay"
    STUB = "stub"
    LOCAL = "local"  # reserved for a local Ollama-style adapter


# ---------------------------------------------------------------------------
# Ingestion tables
# ---------------------------------------------------------------------------


class Dataset(Base):
    """One successful import of the JSONL pack."""

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    source_dir: Mapped[str] = mapped_column(Text)
    prompt_count: Mapped[int] = mapped_column(Integer, default=0)
    response_count: Mapped[int] = mapped_column(Integer, default=0)
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    rejected_count: Mapped[int] = mapped_column(Integer, default=0)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    responses: Mapped[list[ResponseRow]] = relationship(back_populates="dataset")


class PromptRow(Base):
    __tablename__ = "prompts"
    __table_args__ = (UniqueConstraint("prompt_id", name="uq_prompts_prompt_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    prompt_id: Mapped[str] = mapped_column(String(128), index=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    text: Mapped[str] = mapped_column(Text)

    responses: Mapped[list[ResponseRow]] = relationship(back_populates="prompt")


class ResponseRow(Base):
    __tablename__ = "responses"
    __table_args__ = (
        UniqueConstraint("response_id", name="uq_responses_response_id"),
        UniqueConstraint("prompt_id", "engine", name="uq_responses_prompt_engine"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    response_id: Mapped[str] = mapped_column(String(64), index=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    prompt_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("prompts.prompt_id"), index=True
    )
    engine: Mapped[Engine] = mapped_column(SAEnum(Engine, values_callable=lambda e: [x.value for x in e]))
    engine_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    answer_text: Mapped[str] = mapped_column(Text)
    answer_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    capture_status: Mapped[str] = mapped_column(String(32), default="captured")
    provider_timestamp: Mapped[str | None] = mapped_column(String(64), nullable=True)
    archive_stored_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    settings: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    capture_route: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider_api_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider_task_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id"), index=True
    )

    prompt: Mapped[PromptRow] = relationship(back_populates="responses")
    dataset: Mapped[Dataset] = relationship(back_populates="responses")
    citations: Mapped[list[CitationRow]] = relationship(
        back_populates="response", cascade="all, delete-orphan"
    )
    result: Mapped[ResponseResult | None] = relationship(
        back_populates="response", uselist=False, cascade="all, delete-orphan"
    )


class CitationRow(Base):
    __tablename__ = "citations"
    __table_args__ = (
        UniqueConstraint("citation_id", name="uq_citations_citation_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    citation_id: Mapped[str] = mapped_column(String(64), index=True)
    response_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("responses.response_id"), index=True
    )
    prompt_id: Mapped[str] = mapped_column(String(128), index=True)
    engine: Mapped[Engine] = mapped_column(SAEnum(Engine, values_callable=lambda e: [x.value for x in e]))
    order: Mapped[int] = mapped_column(Integer)
    citation_type: Mapped[str] = mapped_column(String(32))
    url: Mapped[str] = mapped_column(Text)
    normalized_url: Mapped[str] = mapped_column(Text, index=True)
    domain: Mapped[str | None] = mapped_column(String(255), nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_date: Mapped[str | None] = mapped_column(String(64), nullable=True)

    response: Mapped[ResponseRow] = relationship(back_populates="citations")


class RejectedRecord(Base):
    """Rows that failed Pydantic validation during ingestion.

    Stored so one bad record never erases valid results (brief requirement).
    """

    __tablename__ = "rejected_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id"), index=True
    )
    source_file: Mapped[str] = mapped_column(Text)
    raw_payload: Mapped[str] = mapped_column(Text)
    error: Mapped[str] = mapped_column(Text)
    rejected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# Analysis tables
# ---------------------------------------------------------------------------


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id"), index=True
    )
    execution_mode: Mapped[ExecutionMode] = mapped_column(SAEnum(ExecutionMode, values_callable=lambda e: [x.value for x in e]))
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending/running/completed/failed
    total: Mapped[int] = mapped_column(Integer, default=0)
    successful: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    results: Mapped[list[ResponseResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class ResponseResult(Base):
    """Classification + evidence for one ResponseRow.

    `failed=True` means the LLM produced no usable classification —
    this is NOT a successful zero-score observation.
    """

    __tablename__ = "response_results"
    __table_args__ = (
        UniqueConstraint("run_id", "response_id", name="uq_results_run_response"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_runs.id"), index=True
    )
    response_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("responses.response_id"), index=True
    )
    prompt_id: Mapped[str] = mapped_column(String(128), index=True)
    engine: Mapped[Engine] = mapped_column(SAEnum(Engine, values_callable=lambda e: [x.value for x in e]))

    # Labels (the LLM-produced judgment)
    mention: Mapped[bool] = mapped_column(Boolean, default=False)
    recommendation: Mapped[bool] = mapped_column(Boolean, default=False)
    negative_characterisation: Mapped[bool] = mapped_column(Boolean, default=False)
    recommendation_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    competitors_named: Mapped[list[str]] = mapped_column(JSON, default=list)
    competitors_observed_count: Mapped[int] = mapped_column(Integer, default=0)

    # Evidence — exact excerpt + char offset (grader verifies it exists)
    evidence_excerpt: Mapped[str] = mapped_column(Text, default="")
    char_offset_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_offset_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Deterministic citation facts (computed before LLM)
    citation_count_for_response: Mapped[int] = mapped_column(Integer, default=0)
    unique_citation_count: Mapped[int] = mapped_column(Integer, default=0)
    has_kasparro_citation: Mapped[bool] = mapped_column(Boolean, default=False)
    kasparro_domain_in_citations: Mapped[bool] = mapped_column(Boolean, default=False)

    # Score components (filled by scoring step)
    mention_score: Mapped[float] = mapped_column(Float, default=0.0)
    recommendation_score: Mapped[float] = mapped_column(Float, default=0.0)
    negative_penalty: Mapped[float] = mapped_column(Float, default=0.0)
    citation_score: Mapped[float] = mapped_column(Float, default=0.0)
    position_score: Mapped[float] = mapped_column(Float, default=0.0)
    composite_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Failure handling — never manufacture confident results
    failed: Mapped[bool] = mapped_column(Boolean, default=False)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    skip_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Traceability — every output labeled with execution mode
    execution_mode: Mapped[ExecutionMode] = mapped_column(SAEnum(ExecutionMode, values_callable=lambda e: [x.value for x in e]))
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    raw_llm_output: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    prompt_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped[AnalysisRun] = relationship(back_populates="results")
    response: Mapped[ResponseRow] = relationship(back_populates="result")


# ---------------------------------------------------------------------------
# Report review workflow
# ---------------------------------------------------------------------------


class ReportVersion(Base):
    """A versioned report. Once submitted, becomes immutable.

    Editing a submitted/approved version spawns a new draft with
    `parent_version_id` pointing back. Old approvals authorise export
    only of that specific immutable older version.
    """

    __tablename__ = "report_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    analysis_run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analysis_runs.id"), index=True
    )
    parent_version_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("report_versions.id"), nullable=True
    )
    version_number: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[ReportStatus] = mapped_column(
        SAEnum(ReportStatus, values_callable=lambda e: [x.value for x in e]), default=ReportStatus.DRAFT, index=True
    )
    title: Mapped[str] = mapped_column(String(255), default="Kasparro AI Visibility Report")
    findings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    diagnostics: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    remedies: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_by: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    submitted_by: Mapped[str | None] = mapped_column(String(64), nullable=True)

    decisions: Mapped[list[ReviewDecision]] = relationship(
        back_populates="version", cascade="all, delete-orphan"
    )
    parent: Mapped[ReportVersion | None] = relationship(
        remote_side="ReportVersion.id", uselist=False
    )


class ReviewDecision(Base):
    """A reviewer's approve/reject on a submitted version, with comment."""

    __tablename__ = "review_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    version_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("report_versions.id"), index=True
    )
    decision: Mapped[DecisionType] = mapped_column(SAEnum(DecisionType, values_callable=lambda e: [x.value for x in e]))
    actor_id: Mapped[str] = mapped_column(String(64))
    actor_role: Mapped[str] = mapped_column(String(32))  # always "reviewer"
    comment: Mapped[str] = mapped_column(Text, default="")
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    version: Mapped[ReportVersion] = relationship(back_populates="decisions")
