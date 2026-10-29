"""Typed API errors with structured envelope.

Every error returns:
    {"error": {"code": str, "message": str, "trace_id": str}}

Replaces the previous string-matching approach with a typed exception
hierarchy so error codes are deterministic and testable.

Hierarchy:
  APIError (base)
    ├── NotFoundError            (404, not_found)
    ├── WrongRoleError           (403, wrong_role)
    ├── InvalidTransitionError   (409, not_approved | not_submitted | not_draft | concurrent_modification)
    ├── ContentConflictError     (409, content_conflict)
    ├── InsufficientDataError    (409, insufficient_data)
    ├── EmptyDatasetError        (422, empty_dataset)
    ├── UnsupportedClaimsError   (422, unsupported_claims)
    ├── ValidationError          (422, validation_error)
    ├── ModelUnavailableError    (503, model_unavailable)
    └── AnalysisTimeoutError     (504, analysis_timeout)
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import HTTPException


def _trace_id() -> str:
    return uuid.uuid4().hex[:16]


class APIError(HTTPException):
    """Base class for all typed API errors."""

    code: str = "api_error"
    default_status: int = 400
    default_message: str = "API error."

    def __init__(
        self,
        message: str | None = None,
        *,
        status_code: int | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code or self.code
        msg = message or self.default_message
        status = status_code or self.default_status
        body = {
            "error": {
                "code": self.code,
                "message": msg,
                "trace_id": _trace_id(),
            }
        }
        if details:
            body["error"]["details"] = details
        super().__init__(status_code=status, detail=body)


class NotFoundError(APIError):
    code = "not_found"
    default_status = 404
    default_message = "Resource not found."


class WrongRoleError(APIError):
    code = "wrong_role"
    default_status = 403
    default_message = (
        "Caller role is not authorised for this action. Identity is derived "
        "from the bearer token, not the request body."
    )


class InvalidTransitionError(APIError):
    """Raised on illegal state machine transitions.

    transition_code is one of: not_approved, not_submitted, not_draft,
    concurrent_modification.
    """

    default_status = 409

    def __init__(self, transition_code: str, message: str | None = None) -> None:
        self.code = transition_code
        super().__init__(message)


class ContentConflictError(APIError):
    code = "content_conflict"
    default_status = 409
    default_message = (
        "Same ID already imported with different content. The new row was "
        "rejected; the existing row was NOT overwritten."
    )


class InsufficientDataError(APIError):
    code = "insufficient_data"
    default_status = 409
    default_message = (
        "Analysis run has 0 valid results. Cannot create a report from a "
        "run with no successful classifications."
    )


class EmptyDatasetError(APIError):
    code = "empty_dataset"
    default_status = 422
    default_message = "Dataset has 0 responses. Cannot run analysis on an empty dataset."


class UnsupportedClaimsError(APIError):
    code = "unsupported_claims"
    default_status = 422
    default_message = (
        "Report contains claims that are not backed by evidence in the run. "
        "Every cited response_id must exist and every excerpt must be a "
        "verbatim substring of the corresponding answer_text."
    )


class ModelUnavailableError(APIError):
    code = "model_unavailable"
    default_status = 503
    default_message = (
        "The configured LLM model is unavailable. Check that GROQ_API_KEY is "
        "set and the model name is correct (Groq retired llama-3.3-70b-versatile "
        "and llama-3.1-8b-instant on 16 Aug 2026; use openai/gpt-oss-120b)."
    )


class AnalysisTimeoutError(APIError):
    code = "analysis_timeout"
    default_status = 504
    default_message = "Analysis timed out. The run was marked failed."


class DuplicateRequestError(APIError):
    code = "duplicate_request"
    default_status = 409
    default_message = (
        "This request conflicts with an in-progress resource. "
        "Wait for the existing one to finish, or poll its status."
    )


# ---------------------------------------------------------------------------
# Convenience: code -> default (status, message) for non-exception paths
# ---------------------------------------------------------------------------


ERROR_TEMPLATES: dict[str, tuple[int, str, str]] = {
    "missing_token": (401, "missing_token",
        "Authorization header missing or malformed."),
    "unknown_token": (401, "unknown_token",
        "Bearer token did not match any configured actor."),
    "wrong_role": (403, "wrong_role",
        "Caller role is not authorised for this action."),
    "not_found": (404, "not_found", "Resource not found."),
    "not_approved": (409, "not_approved",
        "Cannot export a version that is not in the approved state."),
    "not_submitted": (409, "not_submitted",
        "Cannot decide a version that is not in the submitted state."),
    "not_draft": (409, "not_draft",
        "Cannot edit a non-draft version. Fork a new draft from the existing version."),
    "concurrent_modification": (409, "concurrent_modification",
        "Version was modified by another caller. Reload and retry."),
    "content_conflict": (409, "content_conflict",
        "Same ID already imported with different content."),
    "insufficient_data": (409, "insufficient_data",
        "Analysis run has 0 valid results."),
    "empty_dataset": (422, "empty_dataset",
        "Dataset has 0 responses."),
    "unsupported_claims": (422, "unsupported_claims",
        "Report contains claims not backed by evidence in the run."),
    "validation_error": (422, "validation_error",
        "Request body failed schema validation."),
    "model_unavailable": (503, "model_unavailable",
        "The configured LLM model is unavailable."),
    "analysis_timeout": (504, "analysis_timeout",
        "Analysis timed out."),
    "invalid_source": (400, "invalid_source",
        "source_dir is not accepted. Imports always use DATA_PACK_DIR from .env."),
    "duplicate_request": (409, "duplicate_request",
        "This request has already been processed."),
}


def api_error(key: str, message: str | None = None, status: int | None = None) -> APIError:
    """Build an APIError from a known template, with optional overrides."""
    code_default = ERROR_TEMPLATES.get(key)
    if code_default is None:
        return APIError(message or "Unknown error.", status_code=status or 400, code=key)
    s, c, m = code_default
    return APIError(
        message or m,
        status_code=status or s,
        code=c,
    )


# ---------------------------------------------------------------------------
# Exception handlers for FastAPI
# ---------------------------------------------------------------------------


async def http_exception_handler(request, exc):
    """Handle StarletteHTTPException (e.g. 404 from FastAPI itself)."""
    from fastapi.responses import JSONResponse

    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "http_error", "message": str(exc.detail), "trace_id": _trace_id()}},
    )


async def validation_exception_handler(request, exc):
    """Handle Pydantic RequestValidationError → 422 with structured body."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request body failed schema validation.",
                "trace_id": _trace_id(),
                "details": exc.errors(),
            }
        },
    )


async def unhandled_exception_handler(request, exc):
    """Catch-all for unexpected exceptions → 500."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unhandled exception occurred.",
                "trace_id": _trace_id(),
                "exception_type": type(exc).__name__,
            }
        },
    )
