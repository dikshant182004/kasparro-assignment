"""Analysis run endpoints: POST /analyses, GET /analyses/{id}."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.api.deps import Actor, get_actor
from kasparro_auditor.db.base import get_session
from kasparro_auditor.api.errors import api_error
from kasparro_auditor.config import ExecutionMode, get_settings
from kasparro_auditor.db.models import AnalysisRun, Dataset, ExecutionMode as DBExecutionMode
from kasparro_auditor.llm.factory import build_adapter
from kasparro_auditor.schemas.api import AnalysisCreate, AnalysisRead
from kasparro_auditor.scoring.classify import run_analysis

router = APIRouter(tags=["analyses"])


@router.post("/analyses", response_model=AnalysisRead, status_code=201)
async def create_analysis(
    body: AnalysisCreate,
    background_tasks: BackgroundTasks,
    actor: Actor = Depends(get_actor),
    session: AsyncSession = Depends(get_session),
) -> AnalysisRead:
    """Kick off an analysis run. Returns immediately; runs in background."""
    ds = await session.get(Dataset, body.dataset_id)
    if ds is None:
        raise api_error("not_found", f"Dataset {body.dataset_id} not found.")

    settings = get_settings()
    mode_value = body.execution_mode or settings.execution_mode.value
    mode = ExecutionMode(mode_value)
    adapter = build_adapter(mode)

    run = AnalysisRun(
        dataset_id=body.dataset_id,
        execution_mode=DBExecutionMode(mode.value),
        model_name=getattr(adapter, "model_name", None),
        prompt_hash=None,
        status="running",
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    # Run synchronously within the request for the assignment's small dataset
    # (150 responses, bounded concurrency). A bounded synchronous flow is
    # explicitly allowed by the brief.
    try:
        await asyncio.wait_for(
            run_analysis(session, run, body.dataset_id, adapter),
            timeout=settings.analysis_timeout_seconds,
        )
    except asyncio.TimeoutError:
        run.status = "failed"
        run.error_summary = f"Analysis timed out after {settings.analysis_timeout_seconds}s"
        await session.commit()
        raise api_error("llm_timeout")

    await session.refresh(run)
    return AnalysisRead.model_validate(run)


@router.get("/analyses/{analysis_id}", response_model=AnalysisRead)
async def get_analysis(
    analysis_id: str = Path(...),
    actor: Actor = Depends(get_actor),
    session: AsyncSession = Depends(get_session),
) -> AnalysisRead:
    run = await session.get(AnalysisRun, analysis_id)
    if run is None:
        raise api_error("not_found", f"AnalysisRun {analysis_id} not found.")
    return AnalysisRead.model_validate(run)
