"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from kasparro_auditor import __version__
from kasparro_auditor.api.analyses import router as analyses_router
from kasparro_auditor.api.datasets import router as datasets_router
from kasparro_auditor.api.errors import (
    APIError,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from kasparro_auditor.api.health import router as health_router
from kasparro_auditor.api.reports import router as reports_router
from kasparro_auditor.config import get_settings
from kasparro_auditor.reports.state import ReportStateError


@asynccontextmanager
async def lifespan(app: FastSession):  # type: ignore[name-defined]
    """App startup + shutdown lifecycle."""
    # Startup: refuse to run with default tokens
    settings = get_settings()
    settings.validate_startup()

    # Create DB tables if not present (dev convenience; production
    # would use Alembic migrations).
    from kasparro_auditor.db.base import Base, engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown: dispose engine
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Kasparro Auditor",
        description="AI-visibility analysis backend — Kasparro Fresher AI Engineer assignment",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Exception handlers
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(APIError, _api_error_handler)
    app.add_exception_handler(ReportStateError, _report_state_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Routers
    app.include_router(health_router)
    app.include_router(datasets_router)
    app.include_router(analyses_router)
    app.include_router(reports_router)

    return app


async def _api_error_handler(request: Request, exc: APIError):
    """APIError carries its own structured body in .detail."""
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=exc.status_code, content=exc.detail)


async def _report_state_error_handler(request: Request, exc: ReportStateError):
    """Map ReportStateError to the right HTTP status by transition_code."""
    from fastapi.responses import JSONResponse

    transition_code = getattr(exc, "transition_code", None)
    if transition_code:
        # Already a typed transition code (not_approved, not_submitted, not_draft, concurrent_modification)
        from kasparro_auditor.api.errors import ERROR_TEMPLATES

        status, code, _ = ERROR_TEMPLATES.get(
            transition_code, (409, transition_code, str(exc))
        )
        body = {"error": {"code": code, "message": str(exc)}}
    else:
        # Legacy fallback: keep message-based mapping for safety
        msg = str(exc)
        if "Cannot export" in msg or "rejected" in msg.lower():
            code, status = "not_approved", 409
        elif "Cannot decide" in msg:
            code, status = "not_submitted", 409
        elif "Cannot PATCH" in msg or "Cannot edit" in msg:
            code, status = "not_draft", 409
        elif "role" in msg.lower():
            code, status = "wrong_role", 403
        else:
            code, status = "not_found", 404
        body = {"error": {"code": code, "message": msg}}
    import uuid

    body["error"]["trace_id"] = uuid.uuid4().hex[:16]
    return JSONResponse(status_code=status, content=body)


app = create_app()
