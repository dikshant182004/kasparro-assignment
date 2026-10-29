"""Structured error helpers.

Every 4xx/5xx returns `{"error": {"code", "message", "trace_id"}}`.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


def _trace_id() -> str:
    return uuid.uuid4().hex[:16]


class APIError(HTTPException):
    """HTTP exception with a structured error body."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(
            status_code=status_code,
            detail={"error": {"code": code, "message": message, "trace_id": _trace_id()}},
        )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "http_error", "message": str(exc.detail), "trace_id": _trace_id()}},
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
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


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
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


ERROR_TEMPLATES: dict[str, tuple[int, str, str]] = {
    # (status_code, code, default_message)
    "duplicate_response": (409, "duplicate_response",
        "A response with this response_id already exists. Use the existing dataset or fork a new one."),
    "wrong_role": (403, "wrong_role",
        "Caller role is not authorised for this action. Identity is derived from the bearer token, not the request body."),
    "rejected_blocks_export": (409, "rejected_blocks_export",
        "Cannot export a rejected version. Fork a new draft and obtain fresh review."),
    "not_submitted": (409, "not_submitted",
        "Cannot decide a version that is not in the submitted state."),
    "not_approved": (409, "not_approved",
        "Cannot export a version that is not in the approved state."),
    "not_draft": (409, "not_draft",
        "Cannot edit a non-draft version. Fork a new draft from the existing version."),
    "unknown_token": (401, "unknown_token",
        "Bearer token did not match any configured actor."),
    "missing_token": (401, "missing_token",
        "Authorization header missing or malformed."),
    "not_found": (404, "not_found", "Resource not found."),
    "llm_timeout": (504, "llm_timeout",
        "LLM call timed out. The result row is recorded as failed; export is blocked."),
    "malformed_llm_output": (422, "malformed_llm_output",
        "LLM output failed schema or evidence verification. The result row is recorded as failed."),
    "duplicate_request": (409, "duplicate_request",
        "This request has already been processed. Use the existing resource."),
}


def api_error(key: str, message: str | None = None, status: int | None = None) -> APIError:
    """Build an APIError from a known template, with optional overrides."""
    code_default = ERROR_TEMPLATES.get(key)
    if code_default is None:
        return APIError(status or 400, key, message or "Unknown error.")
    s, c, m = code_default
    return APIError(status or s, c, message or m)
