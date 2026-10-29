"""Dataset import endpoints: POST /datasets, GET /datasets/{id}."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.api.deps import Actor, get_actor
from kasparro_auditor.db.base import get_session
from kasparro_auditor.api.errors import api_error
from kasparro_auditor.db.models import Dataset
from kasparro_auditor.ingest.loader import import_pack
from kasparro_auditor.schemas.api import DatasetCreate, DatasetRead

router = APIRouter(tags=["datasets"])


@router.post("/datasets", response_model=DatasetRead, status_code=201)
async def create_dataset(
    body: DatasetCreate,
    actor: Actor = Depends(get_actor),
    session: AsyncSession = Depends(get_session),
) -> DatasetRead:
    """Import the assignment pack. Idempotent on response_id."""
    report = await import_pack(session, body.source_dir)
    ds = await session.get(Dataset, report.dataset_id)
    if ds is None:  # pragma: no cover — defensive
        raise api_error("not_found", "Dataset not created.")
    return DatasetRead.model_validate(ds)


@router.get("/datasets/{dataset_id}", response_model=DatasetRead)
async def get_dataset(
    dataset_id: str = Path(...),
    actor: Actor = Depends(get_actor),
    session: AsyncSession = Depends(get_session),
) -> DatasetRead:
    ds = await session.get(Dataset, dataset_id)
    if ds is None:
        raise api_error("not_found", f"Dataset {dataset_id} not found.")
    return DatasetRead.model_validate(ds)
