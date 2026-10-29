"""Pydantic result schemas.

`ClassificationResult` is the structured-output contract the LLM must
return. `ResponseResultRead` is the full DB row projection used by the
API and grader.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ClassificationResult(BaseModel):
    """Structured output contract enforced via Instructor + Groq JSON mode."""

    model_config = ConfigDict(extra="forbid")

    mention: bool = Field(
        description="True if the answer names Kasparro anywhere in the text."
    )
    recommendation: bool = Field(
        description="True only if the answer presents Kasparro as a suitable option "
        "for the question. A passing mention is NOT a recommendation."
    )
    recommendation_position: int | None = Field(
        default=None,
        description="1-based position if Kasparro is explicitly ranked or listed "
        "among comparable options. Null if not ranked/listed (prose mention only).",
        ge=1,
        le=20,
    )
    negative_characterisation: bool = Field(
        description="True if the answer criticises Kasparro, expresses scepticism, "
        "calls it a gimmick/hype/scam, or otherwise characterises it negatively."
    )
    competitors_named: list[str] = Field(
        default_factory=list,
        description="Lowercased names of competing AI-visibility tools explicitly "
        "named in the answer (e.g. profound, otterly ai, peec ai, athenahq, dageno ai).",
    )
    evidence_excerpt: str = Field(
        description="An exact substring copied verbatim from the answer that supports "
        "the labels above. Must be >= 10 characters and present verbatim."
    )
    char_offset_start: int = Field(
        description="Character offset in answer_text where evidence_excerpt begins.",
        ge=0,
    )
    char_offset_end: int = Field(
        description="Character offset in answer_text where evidence_excerpt ends.",
        ge=0,
    )


class ResponseResultRead(BaseModel):
    """Projection of a ResponseResult row for API/grader consumption."""

    model_config = ConfigDict(from_attributes=True)

    response_id: str
    prompt_id: str
    engine: str
    mention: bool
    recommendation: bool
    negative_characterisation: bool
    recommendation_position: int | None
    competitors_named: list[str]
    evidence_excerpt: str
    char_offset_start: int | None
    char_offset_end: int | None
    citation_count_for_response: int
    has_kasparro_citation: bool
    composite_score: float
    failed: bool
    failure_reason: str | None
    skipped: bool
    skip_reason: str | None
    execution_mode: str
    model_name: str | None
    latency_ms: int


class VisibilityScore(BaseModel):
    """Composite visibility score with components and uncertainty."""

    mention_rate: float
    recommendation_rate: float
    negative_rate: float
    citation_rate: float
    avg_position: float | None
    composite: float
    composite_ci_low: float
    composite_ci_high: float
    weights: dict[str, float]
    denominator_valid: int
    denominator_total: int
    failed_count: int
    skipped_count: int


class Diagnostic(BaseModel):
    """One evidence-backed diagnostic finding."""

    id: str
    title: str
    pattern: str  # recorded observation
    hypothesis: str  # possible explanation
    evidence_response_ids: list[str]
    evidence_excerpts: list[str]
    would_change_conclusion: str


class Remedy(BaseModel):
    """One prioritised remedy."""

    id: str
    priority: int  # 1 = highest
    problem: str
    supporting_evidence: list[str]  # diagnostic ids
    action: str
    expected_benefit: str
    effort: str  # low/medium/high
    uncertainty: str  # low/medium/high
    validation_step: str
