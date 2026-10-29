"""Health endpoint.

Returns version + execution_mode + DB liveness (SELECT 1).
Does NOT leak database_url — credentials in URLs are private.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor import __version__
from kasparro_auditor.config import get_settings
from kasparro_auditor.db.base import get_session
from kasparro_auditor.schemas.api import HealthRead

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthRead)
async def health(session: AsyncSession = Depends(get_session)) -> HealthRead:
    settings = get_settings()
    # Run SELECT 1 to verify DB connectivity. If this fails, FastAPI
    # will return 500 (the dep rolls back + re-raises).
    await session.execute(text("SELECT 1"))
    return HealthRead(
        status="ok",
        version=__version__,
        execution_mode=settings.execution_mode.value,
        database_healthy=True,
    )
