"""JSONL → SQLite import with idempotent upserts.

A repeated import of the same `response_id` is rejected with HTTP 409
at the API layer (see api/datasets.py). At the loader level, we use
`ON CONFLICT DO NOTHING` so re-imports are safe and idempotent — the
caller is informed of how many were new vs. reused.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.db.models import (
    CitationRow,
    Dataset,
    Engine,
    PromptRow,
    RejectedRecord,
    ResponseRow,
)
from kasparro_auditor.ingest.normalize import domain_of, normalize_url
from kasparro_auditor.schemas.inputs import CitationRecord, PromptRecord, ResponseRecord
from kasparro_auditor.config import get_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one JSON object per non-blank line."""
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path.name}:{line_no} invalid JSON: {err}") from err


@dataclass
class ImportReport:
    dataset_id: str
    prompts_new: int = 0
    responses_new: int = 0
    responses_reused: int = 0
    citations_new: int = 0
    rejected: list[dict[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


async def import_pack(
    session: AsyncSession,
    source_dir: str | Path | None = None,
) -> ImportReport:
    """Import prompts/responses/citations from the pack directory.

    Idempotent: a response_id that already exists in the DB is NOT
    duplicated. The caller is informed via `responses_reused`.
    """
    settings = get_settings()
    src = Path(source_dir) if source_dir else settings.data_pack_dir
    data_dir = src / "data" if (src / "data").is_dir() else src

    required = ["prompts.jsonl", "responses.jsonl", "citations.jsonl"]
    missing = [r for r in required if not (data_dir / r).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Pack directory {data_dir} is missing required files: {missing}"
        )

    # Read run_id from data/run.json for traceability
    run_meta: dict[str, Any] = {}
    run_path = data_dir / "run.json"
    if run_path.is_file():
        run_meta = json.loads(run_path.read_text(encoding="utf-8"))
    run_id = run_meta.get("run_id", "unknown")

    # Create the Dataset row first
    dataset = Dataset(
        run_id=run_id,
        source_dir=str(src.resolve()),
    )
    session.add(dataset)
    await session.flush()
    report = ImportReport(dataset_id=dataset.id)

    # 1. Prompts
    await _import_prompts(session, dataset, data_dir / "prompts.jsonl", report)

    # 2. Responses (idempotent upsert)
    await _import_responses(session, dataset, data_dir / "responses.jsonl", report)

    # 3. Citations
    await _import_citations(session, data_dir / "citations.jsonl", report)

    # Update dataset counters
    dataset.prompt_count = report.prompts_new
    dataset.response_count = report.responses_new + report.responses_reused
    dataset.citation_count = report.citations_new
    dataset.rejected_count = len(report.rejected)

    await session.commit()
    return report


# ---------------------------------------------------------------------------
# Per-file importers
# ---------------------------------------------------------------------------


async def _import_prompts(
    session: AsyncSession, dataset: Dataset, path: Path, report: ImportReport
) -> None:
    for raw in _read_jsonl(path):
        try:
            rec = PromptRecord.model_validate(raw)
        except ValidationError as err:
            await _reject(session, dataset.id, path.name, raw, str(err))
            report.rejected.append({"file": path.name, "error": str(err)})
            continue
        stmt = sqlite_insert(PromptRow).values(
            prompt_id=rec.prompt_id,
            run_id=rec.run_id,
            text=rec.text,
        ).on_conflict_do_nothing(index_elements=["prompt_id"])
        result = await session.execute(stmt)
        report.prompts_new += result.rowcount or 0


async def _import_responses(
    session: AsyncSession, dataset: Dataset, path: Path, report: ImportReport
) -> None:
    for raw in _read_jsonl(path):
        try:
            rec = ResponseRecord.model_validate(raw)
        except ValidationError as err:
            await _reject(session, dataset.id, path.name, raw, str(err))
            report.rejected.append({"file": path.name, "error": str(err)})
            continue

        # Idempotency: existing response_id is reused, NOT duplicated.
        existing = await session.scalar(
            select(ResponseRow).where(ResponseRow.response_id == rec.response_id)
        )
        if existing is not None:
            report.responses_reused += 1
            continue

        row = ResponseRow(
            response_id=rec.response_id,
            run_id=rec.run_id,
            prompt_id=rec.prompt_id,
            engine=Engine(rec.engine),
            engine_name=rec.engine_name,
            model_version=rec.model_version,
            answer_text=rec.answer_text,
            answer_markdown=rec.answer_markdown,
            capture_status=rec.capture_status,
            provider_timestamp=rec.provider_timestamp,
            archive_stored_at=rec.archive_stored_at,
            settings=rec.settings,
            capture_route=rec.capture_route,
            provider_api_version=rec.provider_api_version,
            provider_task_id=rec.provider_task_id,
            dataset_id=dataset.id,
        )
        session.add(row)
        report.responses_new += 1
    await session.flush()


async def _import_citations(
    session: AsyncSession, path: Path, report: ImportReport
) -> None:
    for raw in _read_jsonl(path):
        try:
            rec = CitationRecord.model_validate(raw)
        except ValidationError as err:
            await _reject(session, "", path.name, raw, str(err))
            report.rejected.append({"file": path.name, "error": str(err)})
            continue
        norm = normalize_url(rec.url)
        dom = rec.domain or domain_of(rec.url)
        # Idempotent upsert on citation_id — re-importing the same pack
        # is a no-op rather than a constraint violation.
        stmt = sqlite_insert(CitationRow).values(
            citation_id=rec.citation_id,
            response_id=rec.response_id,
            prompt_id=rec.prompt_id,
            engine=Engine(rec.engine),
            order=rec.order,
            citation_type=rec.citation_type,
            url=rec.url,
            normalized_url=norm,
            domain=dom,
            title=rec.title,
            publication_date=rec.publication_date,
        ).on_conflict_do_nothing(index_elements=["citation_id"])
        result = await session.execute(stmt)
        report.citations_new += result.rowcount or 0
    await session.flush()


async def _reject(
    session: AsyncSession, dataset_id: str, source_file: str, raw: Any, error: str
) -> None:
    session.add(
        RejectedRecord(
            dataset_id=dataset_id or "",
            source_file=source_file,
            raw_payload=json.dumps(raw, default=str),
            error=error[:2000],
        )
    )
