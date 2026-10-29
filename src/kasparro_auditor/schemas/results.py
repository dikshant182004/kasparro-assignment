"""Pydantic result schemas.

`Judgment` is the structured-output contract the LLM must return —
only judgment fields, NOT facts (mention/competitors/citations are
computed deterministically in scoring/facts.py).

`ResponseResultRead` is the full DB row projection used by the API and grader.

`VisibilityScore` (v2) — composite is Optional, with per-engine + per-category
slices, weight sensitivity table, and an insufficient_data flag for all-failed runs.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Judgment(BaseModel):
    """LLM structured-output schema. Facts are NOT here — only judgment.

    The LLM does not return mention (computed by regex), competitors_known
    (computed by alias matching), or citation counts (computed from the DB).
    Cross-field validators enforce evidence-required-when-true.
    """

    model_config = ConfigDict(extra="forbid")

    recommendation: bool = Field(
        description="True if the answer presents Kasparro as an option the user "
        "should consider or choose. Conditional/hypothetical mentions and "
        "'I couldn't find evidence' are NOT recommendations."
    )
    negative_characterisation: bool = Field(
        description="True if the answer itself criticises or dismisses Kasparro. "
        "Negated wording is NOT negative."
    )
    recommendation_position: int | None = Field(
        default=None,
        description="1-based rank ONLY if Kasparro is explicitly ranked/listed.",
        ge=1,
        le=20,
    )
    competitors_extra: list[str] = Field(
        default_factory=list,
        description="Other provider names named verbatim in the answer, excluding "
        "those already in competitors_known.",
    )
    evidence_mention: str = Field(
        default="",
        description="Verbatim excerpt (<=300 chars) supporting the mention label. "
        "Required when mention=True (caller checks).",
    )
    evidence_recommendation: str = Field(
        default="",
        description="Verbatim excerpt (<=300 chars) supporting recommendation=True.",
    )
    evidence_negative: str = Field(
        default="",
        description="Verbatim excerpt (<=300 chars) supporting negative=True.",
    )

    @model_validator(mode="after")
    def _evidence_required_when_true(self) -> "Judgment":
        if self.recommendation and not self.evidence_recommendation.strip():
            raise ValueError("evidence_recommendation is required when recommendation=True")
        if self.negative_characterisation and not self.evidence_negative.strip():
            raise ValueError("evidence_negative is required when negative_characterisation=True")
        return self


class ResponseResultRead(BaseModel):
    """Projection of a ResponseResult row for API/grader consumption."""

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
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class EngineSlice(BaseModel):
    """Per-engine score slice with denominators + CI."""

    composite: float
    composite_ci_low: float | None
    composite_ci_high: float | None
    denominator_valid: int
    denominator_total: int
    failed_count: int
    skipped_count: int


class CategorySlice(BaseModel):
    """Per-category score slice (e.g. branded vs generic)."""

    composite: float
    composite_ci_low: float | None
    composite_ci_high: float | None
    denominator_valid: int
    denominator_total: int
    failed_count: int
    skipped_count: int


class VisibilityScore(BaseModel):
    """Composite visibility score (v2) — with per-engine + per-category slices.

    composite is Optional: None + insufficient_data=True when no valid rows
    (all-failed run). This prevents the "failed analysis as a successful
    zero-score observation" the brief forbids.
    """

    composite: float | None = None
    composite_ci_low: float | None = None
    composite_ci_high: float | None = None
    insufficient_data: bool = False
    weights: dict[str, float]
    denominator_valid: int
    denominator_total: int
    failed_count: int
    skipped_count: int

    # Rate components (kept for diagnostics + comparison)
    mention_rate: float = 0.0
    recommendation_rate: float = 0.0
    negative_rate: float = 0.0
    citation_rate: float = 0.0
    avg_position: float | None = None

    # Per-engine + per-category slices with denominators
    per_engine: dict[str, EngineSlice] = Field(default_factory=dict)
    per_category: dict[str, CategorySlice] = Field(default_factory=dict)

    # Weight sensitivity table — composite under 3 alternative weight sets
    weight_sensitivity: dict[str, float] = Field(default_factory=dict)

    # Version tracking
    scoring_version: str = "v2"
    category_version: str = "v1"


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
