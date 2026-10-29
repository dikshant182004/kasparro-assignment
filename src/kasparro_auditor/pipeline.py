"""End-to-end pipeline runner (library entry point).

Usage:
    from kasparro_auditor.pipeline import run_full_pipeline
    await run_full_pipeline(source_dir="./data_pack", execution_mode="replay")
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.config import ExecutionMode, get_settings
from kasparro_auditor.db.base import async_session_factory
from kasparro_auditor.db.models import AnalysisRun, Dataset
from kasparro_auditor.llm.factory import build_adapter
from kasparro_auditor.reports.diagnostics import build_diagnostics_and_remedies
from kasparro_auditor.reports.state import create_draft, submit_version
from kasparro_auditor.scoring.classify import run_analysis
from kasparro_auditor.scoring.score import compute_visibility_score


async def run_full_pipeline(
    source_dir: str | Path | None = None,
    execution_mode: ExecutionMode | None = None,
    auto_submit: bool = False,
    out_path: str | Path | None = None,
) -> dict:
    """Run import → analysis → diagnostics → remedies → optional submit.

    Returns a dict summary suitable for the analysis output JSON.
    """
    settings = get_settings()
    mode = execution_mode or settings.execution_mode

    # Ensure tables exist before importing (script entry point convenience).
    from kasparro_auditor.db.base import Base, engine as _engine

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_factory() as session:
        # 1. Ingest
        from kasparro_auditor.ingest.loader import import_pack

        report = await import_pack(session, source_dir)

        # 2. Analysis
        adapter = build_adapter(mode)
        run = AnalysisRun(
            dataset_id=report.dataset_id,
            execution_mode=mode,
            model_name=getattr(adapter, "model_name", None),
            status="running",
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)

        await run_analysis(session, run, report.dataset_id, adapter)

        # 3. Diagnostics + remedies + score
        diagnostics, remedies, score = await build_diagnostics_and_remedies(
            session, run.id, use_llm=(mode == ExecutionMode.LIVE)
        )

        # 4. Optionally submit a draft
        version_id = None
        version_status = None
        if auto_submit:
            from kasparro_auditor.api.deps import Actor
            from kasparro_auditor.config import ActorRole

            version = await create_draft(
                session,
                analysis_run_id=run.id,
                actor_id="pipeline-automation",
                findings={
                    "summary": f"Auto-generated report from {mode.value} run",
                    "score": score.model_dump(),
                },
                scores=score.model_dump(),
                diagnostics=diagnostics,
                remedies=remedies,
            )
            await session.commit()
            version_id = version.id
            version_status = version.status.value

        summary = {
            "dataset_id": report.dataset_id,
            "analysis_run_id": run.id,
            "execution_mode": mode.value,
            "model_name": run.model_name,
            "counts": {
                "prompts": report.prompts_new,
                "responses_new": report.responses_new,
                "responses_reused": report.responses_reused,
                "citations": report.citations_new,
                "rejected": len(report.rejected),
                "successful": run.successful,
                "failed": run.failed,
                "skipped": run.skipped,
            },
            "score": score.model_dump(),
            "diagnostics": diagnostics,
            "remedies": remedies,
            "version_id": version_id,
            "version_status": version_status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

    if out_path:
        Path(out_path).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return summary
