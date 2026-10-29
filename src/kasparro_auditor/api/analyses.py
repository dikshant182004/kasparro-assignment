"""Analysis run endpoints: POST /analyses, GET /analyses/{id}.

POST /analyses is analyst-only. Runs synchronously (within the request)
for the small assignment dataset — the brief explicitly allows a bounded
synchronous flow. Adds:
  - Empty dataset gate (422 empty_dataset)
  - One-running-analysis-per-dataset (409 duplicate_request if a run is active)
  - Preflight model check when mode=live (503 model_unavailable instead
    of failing 150 times quietly)
  - Timeout renamed to analysis_timeout (was llm_timeout, misleading)
"""
from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, Header, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.api.deps import Actor, require_analyst
from kasparro_auditor.api.errors import (
    AnalysisTimeoutError,
    APIError,
    DuplicateRequestError,
    EmptyDatasetError,
    ModelUnavailableError,
    NotFoundError,
)
from kasparro_auditor.config import ExecutionMode, get_settings
from kasparro_auditor.db.base import get_session
from kasparro_auditor.db.models import (
    AnalysisRun,
    Dataset,
    ExecutionMode as DBExecutionMode,
    ResponseRow,
)
from kasparro_auditor.llm.factory import build_adapter, preflight
from kasparro_auditor.schemas.api import AnalysisCreate, AnalysisRead
from kasparro_auditor.scoring.classify import run_analysis
from kasparro_auditor.scoring.score import (
    CATEGORY_VERSION,
    SCORING_VERSION,
    compute_visibility_score,
)

router = APIRouter(tags=["analyses"])


async def _compute_score_dict(session: AsyncSession, run: AnalysisRun) -> dict | None:
    """Compute the visibility score for a completed run, return as dict."""
    if run.status != "completed":
        return None
    score = await compute_visibility_score(session, run.id)
    return score.model_dump()


@router.post("/analyses", response_model=AnalysisRead, status_code=201)
async def create_analysis(
    body: AnalysisCreate,
    actor: Actor = Depends(require_analyst),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: AsyncSession = Depends(get_session),
) -> AnalysisRead:
    """Kick off an analysis run.

    Runs synchronously within the request (the brief allows a bounded
    synchronous flow for the small 150-response dataset).

    Gates:
      - Dataset must exist (404)
      - Dataset must have responses (422 empty_dataset)
      - Only one running analysis per dataset (409 duplicate_request)
      - Preflight model check when mode=live (503 model_unavailable)
    """
    ds = await session.get(Dataset, body.dataset_id)
    if ds is None:
        raise NotFoundError(f"Dataset {body.dataset_id} not found.")

    # Empty dataset gate
    response_count = await session.scalar(
        select(ResponseRow).where(ResponseRow.dataset_id == body.dataset_id)
    )
    if response_count is None:
        raise EmptyDatasetError(
            f"Dataset {body.dataset_id} has 0 responses. "
            "Cannot run analysis on an empty dataset."
        )

    # One running analysis per dataset
    existing_running = (
        await session.scalars(
            select(AnalysisRun)
            .where(AnalysisRun.dataset_id == body.dataset_id)
            .where(AnalysisRun.status.in_(["pending", "running"]))
        )
    ).first()
    if existing_running is not None:
        raise DuplicateRequestError(
            f"Analysis run {existing_running.id} is already in progress "
            f"on dataset {body.dataset_id}. Wait for it to finish, or use "
            f"GET /analyses/{existing_running.id} to check status."
        )

    settings = get_settings()
    mode_value = body.execution_mode or settings.execution_mode.value
    mode = ExecutionMode(mode_value)
    adapter = build_adapter(mode)

    # Preflight: if live mode, make one trivial call to verify the model works
    # before committing to 150 calls. Fail loudly here instead of 150 times quietly.
    if mode == ExecutionMode.LIVE:
        try:
            await preflight(adapter)
        except Exception:
            raise ModelUnavailableError(
                "Preflight check failed. The configured Groq model is unavailable. "
                "Verify GROQ_API_KEY is set and the model name is current. "
                "Groq retired llama-3.3-70b-versatile on 16 Aug 2026; "
                "use openai/gpt-oss-120b."
            )

    run = AnalysisRun(
        dataset_id=body.dataset_id,
        execution_mode=DBExecutionMode(mode.value),
        model_name=getattr(adapter, "model_name", None),
        prompt_hash=None,
        scoring_version=SCORING_VERSION,
        category_version=CATEGORY_VERSION,
        status="running",
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    # Run synchronously within the request for the assignment's small dataset.
    try:
        await asyncio.wait_for(
            run_analysis(session, run, body.dataset_id, adapter),
            timeout=settings.analysis_timeout_seconds,
        )
    except asyncio.TimeoutError:
        run.status = "failed"
        run.error_summary = f"Analysis timed out after {settings.analysis_timeout_seconds}s"
        await session.commit()
        raise AnalysisTimeoutError(
            f"Analysis timed out after {settings.analysis_timeout_seconds}s. "
            f"The run was marked failed; see GET /analyses/{run.id}/results."
        )

    await session.refresh(run)
    score_dict = await _compute_score_dict(session, run)
    return AnalysisRead.model_validate(run).model_copy(update={"score": score_dict})


@router.get("/analyses/{analysis_id}", response_model=AnalysisRead)
async def get_analysis(
    analysis_id: str = Path(...),
    actor: Actor = Depends(require_analyst),
    session: AsyncSession = Depends(get_session),
) -> AnalysisRead:
    run = await session.get(AnalysisRun, analysis_id)
    if run is None:
        raise NotFoundError(f"AnalysisRun {analysis_id} not found.")
    score_dict = await _compute_score_dict(session, run)
    return AnalysisRead.model_validate(run).model_copy(update={"score": score_dict})
