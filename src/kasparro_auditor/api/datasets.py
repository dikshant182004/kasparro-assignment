"""Dataset import endpoints: POST /datasets, GET /datasets/{id}, GET /datasets/{id}/rejected.

POST /datasets is analyst-only. No source_dir accepted — always imports
from DATA_PACK_DIR configured via .env.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.api.deps import Actor, get_actor, require_analyst
from kasparro_auditor.api.errors import EmptyDatasetError, NotFoundError
from kasparro_auditor.db.base import get_session
from kasparro_auditor.db.models import Dataset, RejectedRecord
from kasparro_auditor.ingest.loader import import_pack
from kasparro_auditor.schemas.api import (
    DatasetCreate,
    DatasetRead,
    RejectedRecordRead,
)

router = APIRouter(tags=["datasets"])


@router.post("/datasets", response_model=DatasetRead, status_code=201)
async def create_dataset(
    body: DatasetCreate | None = None,
    actor: Actor = Depends(require_analyst),
    session: AsyncSession = Depends(get_session),
) -> DatasetRead:
    """Import the assignment pack from DATA_PACK_DIR. Analyst-only.

    Idempotent: same response_id + same content → reused;
    same response_id + different content → conflict (rejected, not overwritten).
    """
    # No source_dir from caller — always use DATA_PACK_DIR from .env
    report = await import_pack(session, source_dir=None)
    ds = await session.get(Dataset, report.dataset_id)
    if ds is None:  # pragma: no cover — defensive
        raise NotFoundError("Dataset not created.")
    return DatasetRead.model_validate(ds)


@router.get("/datasets/{dataset_id}", response_model=DatasetRead)
async def get_dataset(
    dataset_id: str = Path(...),
    actor: Actor = Depends(get_actor),
    session: AsyncSession = Depends(get_session),
) -> DatasetRead:
    ds = await session.get(Dataset, dataset_id)
    if ds is None:
        raise NotFoundError(f"Dataset {dataset_id} not found.")
    return DatasetRead.model_validate(ds)


@router.get("/datasets/{dataset_id}/rejected", response_model=list[RejectedRecordRead])
async def list_rejected(
    dataset_id: str = Path(...),
    actor: Actor = Depends(get_actor),
    session: AsyncSession = Depends(get_session),
) -> list[RejectedRecordRead]:
    """List all rejected records for a dataset, with line numbers + errors."""
    ds = await session.get(Dataset, dataset_id)
    if ds is None:
        raise NotFoundError(f"Dataset {dataset_id} not found.")
    rows = (
        await session.scalars(
            select(RejectedRecord)
            .where(RejectedRecord.dataset_id == dataset_id)
            .order_by(RejectedRecord.source_file, RejectedRecord.line_no)
        )
    ).all()
    return [RejectedRecordRead.model_validate(r) for r in rows]
