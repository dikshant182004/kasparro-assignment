"""Health endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from kasparro_auditor import __version__
from kasparro_auditor.config import get_settings
from kasparro_auditor.schemas.api import HealthRead

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthRead)
async def health() -> HealthRead:
    settings = get_settings()
    return HealthRead(
        status="ok",
        version=__version__,
        execution_mode=settings.execution_mode.value,
        # Strip credentials from the URL if any
        database_url=settings.database_url,
    )
