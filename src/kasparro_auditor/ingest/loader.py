"""JSONL → SQLite import with content hashing, reuse-vs-conflict detection,
and referential integrity checks.

Policy (disclosed in README):
  - Same response_id + same content_hash → REUSE (idempotent, counted as reused)
  - Same response_id + different content_hash → CONFLICT (rejected, NOT overwritten)
  - Same citation_id + same content_hash → REUSE
  - Same citation_id + different content_hash → CONFLICT (rejected)
  - Malformed JSON line → caught per line, stored in rejected_records with line_no
  - Schema validation failure → rejected_records with line_no + record_id
  - Orphan citation (response_id not found) → rejected_records
  - Engine/prompt_id mismatch (citation vs response) → rejected_records
  - (prompt_id, engine) collision on responses → treated as conflict if content differs

One bad record NEVER erases valid results (brief requirement).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.config import get_settings
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_hash(data: dict[str, Any], fields: list[str]) -> str:
    """SHA-256 of the JSON-serialized selected fields (sorted keys)."""
    payload = json.dumps(
        {f: data.get(f) for f in fields},
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _read_jsonl_with_lines(path: Path) -> Iterator[tuple[int, str, dict[str, Any] | None, str | None]]:
    """Yield (line_no, raw_line, parsed_dict, error).

    Never raises on a bad line — the caller decides what to do.
    One malformed line does NOT abort the whole import.
    """
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Line {line_no}: expected JSON object, got {type(parsed).__name__}")
                yield line_no, stripped, parsed, None
            except (json.JSONDecodeError, ValueError) as err:
                yield line_no, stripped, None, str(err)


@dataclass
class ImportReport:
    dataset_id: str
    run_id: str
    prompts_new: int = 0
    prompts_reused: int = 0
    responses_new: int = 0
    responses_reused: int = 0
    citations_new: int = 0
    citations_reused: int = 0
    conflicts: list[dict[str, str]] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    expected_counts: dict[str, int] = field(default_factory=dict)
    count_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


async def import_pack(
    session: AsyncSession,
    source_dir: str | Path | None = None,
) -> ImportReport:
    """Import prompts/responses/citations from the pack directory.

    Idempotent: same content_hash → reused; different content_hash → conflict.
    One bad line → rejected_records, import continues.
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

    # Read run.json for provenance + expected counts
    run_meta: dict[str, Any] = {}
    run_path = data_dir / "run.json"
    if run_path.is_file():
        run_meta = json.loads(run_path.read_text(encoding="utf-8"))
    run_id = run_meta.get("run_id", "unknown")
    expected = run_meta.get("original_run_counts", {})

    # Create the Dataset row
    dataset = Dataset(run_id=run_id, source_dir=str(src.resolve()))
    session.add(dataset)
    await session.flush()
    report = ImportReport(dataset_id=dataset.id, run_id=run_id, expected_counts=expected)

    # 1. Prompts (with content hash + reuse/conflict)
    await _import_prompts(session, dataset, data_dir / "prompts.jsonl", report)

    # 2. Responses (with content hash + reuse/conflict + prompt existence check)
    await _import_responses(session, dataset, data_dir / "responses.jsonl", report)

    # 3. Citations (with content hash + reuse/conflict + referential checks)
    await _import_citations(session, dataset, data_dir / "citations.jsonl", report)

    # 4. Count ACTUAL linked rows (not deltas) and update Dataset
    await _update_dataset_counts(session, dataset, report)

    # 5. Compare to run.json expected counts
    _check_expected_counts(report)

    await session.commit()
    return report


# ---------------------------------------------------------------------------
# Per-file importers
# ---------------------------------------------------------------------------


async def _import_prompts(
    session: AsyncSession, dataset: Dataset, path: Path, report: ImportReport
) -> None:
    for line_no, raw, parsed, json_err in _read_jsonl_with_lines(path):
        if json_err is not None:
            await _reject(
                session, dataset.id, path.name, line_no, None, raw, json_err, "json_error"
            )
            report.rejected.append({"file": path.name, "line": str(line_no), "error": json_err})
            continue

        try:
            rec = PromptRecord.model_validate(parsed)
        except ValidationError as err:
            record_id = parsed.get("prompt_id")
            await _reject(
                session, dataset.id, path.name, line_no, record_id, raw, str(err), "validation"
            )
            report.rejected.append({"file": path.name, "line": str(line_no), "error": str(err)})
            continue

        chash = _content_hash(parsed, ["prompt_id", "run_id", "text"])

        # Check if prompt_id already exists
        existing = await session.scalar(
            select(PromptRow).where(PromptRow.prompt_id == rec.prompt_id)
        )
        if existing is not None:
            if existing.content_hash == chash:
                report.prompts_reused += 1
            else:
                # Conflict: same ID, different content
                conflict_msg = (
                    f"prompt_id={rec.prompt_id} already imported with different content "
                    f"(existing_hash={existing.content_hash}, new_hash={chash})"
                )
                await _reject(
                    session, dataset.id, path.name, line_no, rec.prompt_id, raw,
                    conflict_msg, "conflict"
                )
                report.conflicts.append({"file": path.name, "line": str(line_no), "id": rec.prompt_id})
            continue

        row = PromptRow(
            prompt_id=rec.prompt_id,
            run_id=rec.run_id,
            text=rec.text,
            content_hash=chash,
            raw_json=raw,
        )
        session.add(row)
        report.prompts_new += 1
    await session.flush()


async def _import_responses(
    session: AsyncSession, dataset: Dataset, path: Path, report: ImportReport
) -> None:
    # Build a set of known prompt_ids for referential checks
    known_prompt_ids = {
        row[0] for row in (await session.execute(select(PromptRow.prompt_id))).all()
    }

    for line_no, raw, parsed, json_err in _read_jsonl_with_lines(path):
        if json_err is not None:
            await _reject(
                session, dataset.id, path.name, line_no, None, raw, json_err, "json_error"
            )
            report.rejected.append({"file": path.name, "line": str(line_no), "error": json_err})
            continue

        try:
            rec = ResponseRecord.model_validate(parsed)
        except ValidationError as err:
            record_id = parsed.get("response_id")
            await _reject(
                session, dataset.id, path.name, line_no, record_id, raw, str(err), "validation"
            )
            report.rejected.append({"file": path.name, "line": str(line_no), "error": str(err)})
            continue

        # Referential check: prompt_id must exist
        if rec.prompt_id not in known_prompt_ids:
            ref_msg = f"response_id={rec.response_id} references unknown prompt_id={rec.prompt_id}"
            await _reject(
                session, dataset.id, path.name, line_no, rec.response_id, raw, ref_msg, "orphan_reference"
            )
            report.rejected.append({"file": path.name, "line": str(line_no), "error": ref_msg})
            continue

        chash = _content_hash(
            parsed,
            ["response_id", "prompt_id", "engine", "answer_text", "model_version"],
        )

        # Check if response_id already exists
        existing = await session.scalar(
            select(ResponseRow).where(ResponseRow.response_id == rec.response_id)
        )
        if existing is not None:
            if existing.content_hash == chash:
                report.responses_reused += 1
            else:
                conflict_msg = (
                    f"response_id={rec.response_id} already imported with different content "
                    f"(existing_hash={existing.content_hash}, new_hash={chash})"
                )
                await _reject(
                    session, dataset.id, path.name, line_no, rec.response_id, raw,
                    conflict_msg, "conflict"
                )
                report.conflicts.append({"file": path.name, "line": str(line_no), "id": rec.response_id})
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
            content_hash=chash,
            raw_json=raw,
            dataset_id=dataset.id,
        )
        session.add(row)
        report.responses_new += 1
    await session.flush()


async def _import_citations(
    session: AsyncSession, dataset: Dataset, path: Path, report: ImportReport
) -> None:
    # Build lookup of response_id → (engine, prompt_id) for referential checks
    resp_rows = (
        await session.execute(
            select(ResponseRow.response_id, ResponseRow.engine, ResponseRow.prompt_id)
        )
    ).all()
    response_lookup = {
        r[0]: {"engine": r[1].value if hasattr(r[1], "value") else str(r[1]), "prompt_id": r[2]}
        for r in resp_rows
    }

    for line_no, raw, parsed, json_err in _read_jsonl_with_lines(path):
        if json_err is not None:
            await _reject(
                session, dataset.id, path.name, line_no, None, raw, json_err, "json_error"
            )
            report.rejected.append({"file": path.name, "line": str(line_no), "error": json_err})
            continue

        try:
            rec = CitationRecord.model_validate(parsed)
        except ValidationError as err:
            record_id = parsed.get("citation_id")
            await _reject(
                session, dataset.id, path.name, line_no, record_id, raw, str(err), "validation"
            )
            report.rejected.append({"file": path.name, "line": str(line_no), "error": str(err)})
            continue

        # Referential check: response_id must exist
        parent = response_lookup.get(rec.response_id)
        if parent is None:
            ref_msg = f"citation_id={rec.citation_id} references unknown response_id={rec.response_id}"
            await _reject(
                session, dataset.id, path.name, line_no, rec.citation_id, raw,
                ref_msg, "orphan_reference"
            )
            report.rejected.append({"file": path.name, "line": str(line_no), "error": ref_msg})
            continue

        # Referential check: engine must match the response's engine
        if rec.engine != parent["engine"]:
            mismatch_msg = (
                f"citation_id={rec.citation_id} engine={rec.engine} does not match "
                f"response_id={rec.response_id} engine={parent['engine']}"
            )
            await _reject(
                session, dataset.id, path.name, line_no, rec.citation_id, raw,
                mismatch_msg, "engine_mismatch"
            )
            report.rejected.append({"file": path.name, "line": str(line_no), "error": mismatch_msg})
            continue

        # Referential check: prompt_id must match the response's prompt_id
        if rec.prompt_id != parent["prompt_id"]:
            mismatch_msg = (
                f"citation_id={rec.citation_id} prompt_id={rec.prompt_id} does not match "
                f"response_id={rec.response_id} prompt_id={parent['prompt_id']}"
            )
            await _reject(
                session, dataset.id, path.name, line_no, rec.citation_id, raw,
                mismatch_msg, "prompt_mismatch"
            )
            report.rejected.append({"file": path.name, "line": str(line_no), "error": mismatch_msg})
            continue

        chash = _content_hash(
            parsed,
            ["citation_id", "response_id", "url", "order", "citation_type"],
        )

        norm = normalize_url(rec.url)
        dom = domain_of(rec.url)  # always normalize at ingest

        # Check if citation_id already exists
        existing = await session.scalar(
            select(CitationRow).where(CitationRow.citation_id == rec.citation_id)
        )
        if existing is not None:
            if existing.content_hash == chash:
                report.citations_reused += 1
            else:
                conflict_msg = (
                    f"citation_id={rec.citation_id} already imported with different content "
                    f"(existing_hash={existing.content_hash}, new_hash={chash})"
                )
                await _reject(
                    session, dataset.id, path.name, line_no, rec.citation_id, raw,
                    conflict_msg, "conflict"
                )
                report.conflicts.append({"file": path.name, "line": str(line_no), "id": rec.citation_id})
            continue

        row = CitationRow(
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
            content_hash=chash,
            raw_json=raw,
        )
        session.add(row)
        report.citations_new += 1
    await session.flush()


# ---------------------------------------------------------------------------
# Post-import: count actual linked rows + compare to expected
# ---------------------------------------------------------------------------


async def _update_dataset_counts(
    session: AsyncSession, dataset: Dataset, report: ImportReport
) -> None:
    """Count actual rows linked to this dataset — NOT deltas.

    This fixes the bug where re-import showed count=0 because all rows
    were reused. The Dataset always reflects the true linked totals.
    """
    from sqlalchemy import func

    prompt_count = await session.scalar(
        select(func.count()).select_from(PromptRow)
    ) or 0
    response_count = await session.scalar(
        select(func.count())
        .select_from(ResponseRow)
        .where(ResponseRow.dataset_id == dataset.id)
    ) or 0
    citation_count = await session.scalar(
        select(func.count()).select_from(CitationRow)
    ) or 0
    rejected_count = await session.scalar(
        select(func.count())
        .select_from(RejectedRecord)
        .where(RejectedRecord.dataset_id == dataset.id)
    ) or 0

    dataset.prompt_count = int(prompt_count)
    dataset.response_count = int(response_count)
    dataset.citation_count = int(citation_count)
    dataset.rejected_count = int(rejected_count)


def _check_expected_counts(report: ImportReport) -> None:
    """Warn if imported totals differ from run.json expected counts."""
    exp = report.expected_counts
    if not exp:
        return
    actual = {
        "prompts": report.prompts_new + report.prompts_reused,
        "responses": report.responses_new + report.responses_reused,
        "citations": report.citations_new + report.citations_reused,
    }
    for key, expected_val in exp.items():
        actual_val = actual.get(key, 0)
        if expected_val != actual_val:
            report.count_warnings.append(
                f"{key}: expected={expected_val}, actual={actual_val} "
                f"(new={report.__dict__.get(key + '_new', 0)}, "
                f"reused={report.__dict__.get(key + '_reused', 0)})"
            )


async def _reject(
    session: AsyncSession,
    dataset_id: str,
    source_file: str,
    line_no: int | None,
    record_id: str | None,
    raw_payload: str,
    error: str,
    rejection_type: str = "validation",
) -> None:
    session.add(
        RejectedRecord(
            dataset_id=dataset_id,
            source_file=source_file,
            line_no=line_no,
            record_id=record_id,
            raw_payload=raw_payload[:10000],  # cap to prevent huge rows
            error=error[:2000],
            rejection_type=rejection_type,
        )
    )
