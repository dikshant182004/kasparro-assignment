"""FastAPI application entry point."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from kasparro_auditor import __version__
from kasparro_auditor.api.analyses import router as analyses_router
from kasparro_auditor.api.datasets import router as datasets_router
from kasparro_auditor.api.errors import (
    api_error,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from kasparro_auditor.api.health import router as health_router
from kasparro_auditor.api.reports import router as reports_router
from kasparro_auditor.config import get_settings
from kasparro_auditor.reports.state import ReportStateError


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Kasparro Auditor",
        description="AI-visibility analysis backend — Kasparro Fresher AI Engineer assignment",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Exception handlers
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ReportStateError, _report_state_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Routers
    app.include_router(health_router)
    app.include_router(datasets_router)
    app.include_router(analyses_router)
    app.include_router(reports_router)

    @app.on_event("startup")
    async def _startup() -> None:
        # Create DB tables if not present (dev convenience; production
        # would use Alembic migrations).
        from kasparro_auditor.db.base import Base, engine

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    return app


async def _report_state_error_handler(request: Request, exc: ReportStateError):
    """Map ReportStateError to the right HTTP status based on its message."""
    msg = str(exc)
    if "Cannot export" in msg or "rejected" in msg.lower():
        err = api_error("rejected_blocks_export", msg)
    elif "Cannot decide" in msg:
        err = api_error("not_submitted", msg)
    elif "Cannot PATCH" in msg or "Cannot edit" in msg:
        err = api_error("not_draft", msg)
    elif "role" in msg.lower():
        err = api_error("wrong_role", msg)
    else:
        err = api_error("not_found", msg, status=409)
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=err.status_code, content=err.detail)


app = create_app()
