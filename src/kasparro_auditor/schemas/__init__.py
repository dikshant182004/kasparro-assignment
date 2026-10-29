"""Schemas package."""
from kasparro_auditor.schemas.inputs import CitationRecord, PromptRecord, ResponseRecord
from kasparro_auditor.schemas.results import (
    Diagnostic,
    Judgment,
    Remedy,
    ResponseResultRead,
    VisibilityScore,
)
from kasparro_auditor.schemas.api import (
    AnalysisCreate,
    AnalysisRead,
    DatasetCreate,
    DatasetRead,
    DecisionCreate,
    DecisionRead,
    ErrorResponse,
    ExportRead,
    HealthRead,
    RejectedRecordRead,
    ReportPatch,
    ReportRead,
)

__all__ = [
    "CitationRecord",
    "PromptRecord",
    "ResponseRecord",
    "Diagnostic",
    "Judgment",
    "Remedy",
    "ResponseResultRead",
    "VisibilityScore",
    "AnalysisCreate",
    "AnalysisRead",
    "DatasetCreate",
    "DatasetRead",
    "DecisionCreate",
    "DecisionRead",
    "ErrorResponse",
    "ExportRead",
    "HealthRead",
    "RejectedRecordRead",
    "ReportPatch",
    "ReportRead",
]
