"""Capture live LLM responses into the replay cache.

Run this once with a real GROQ_API_KEY to populate
data_pack/replay/{response_id}.json. After capture, the pipeline can
run in replay mode with no API key.

Usage:
    GROQ_API_KEY=gsk_... python -m scripts.replay_capture
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from kasparro_auditor.config import ExecutionMode, get_settings
from kasparro_auditor.db.base import async_session_factory, engine
from kasparro_auditor.db.base import Base
from kasparro_auditor.db.models import PromptRow, ResponseRow
from kasparro_auditor.ingest.loader import import_pack
from kasparro_auditor.llm.groq_adapter import GroqAdapter
from kasparro_auditor.llm.adapter import LLMContext, LLMError, classify_response
from kasparro_auditor.scoring.competitors import COMPETITOR_ALIASES


async def main() -> int:
    settings = get_settings()
    if not settings.groq_api_key:
        print("ERROR: GROQ_API_KEY not set. Set it in .env or env var.")
        return 1

    # Init DB + import pack
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session_factory() as session:
        report = await import_pack(session)
        print(f"Imported {report.responses_new} responses")

        rows = (
            await session.execute(
                select(ResponseRow, PromptRow)
                .join(PromptRow, ResponseRow.prompt_id == PromptRow.prompt_id)
                .where(ResponseRow.dataset_id == report.dataset_id)
            )
        ).all()

        adapter = GroqAdapter(settings)
        print(f"Using model: {adapter.model_name}")
        print(f"Cache dir:   {settings.replay_cache_dir}")

        sem = asyncio.Semaphore(settings.max_concurrent_llm_calls)

        async def _capture_one(response: ResponseRow, prompt: PromptRow) -> tuple[str, str | None]:
            ctx = LLMContext(
                response_id=response.response_id,
                prompt_id=response.prompt_id,
                engine=response.engine.value,
                prompt_text=prompt.text,
                answer_text=response.answer_text,
                competitor_hints=list(COMPETITOR_ALIASES.keys()),
            )
            async with sem:
                try:
                    result = await classify_response(adapter, ctx)
                    return ctx.response_id, None
                except LLMError as e:
                    return ctx.response_id, str(e)

        tasks = [_capture_one(r, p) for r, p in rows]
        results = await asyncio.gather(*tasks)
        ok = sum(1 for _, err in results if err is None)
        failed = [(rid, err) for rid, err in results if err is not None]

        print(f"\nCapture complete: {ok}/{len(results)} succeeded")
        if failed:
            print(f"Failures ({len(failed)}):")
            for rid, err in failed[:10]:
                print(f"  - {rid}: {err[:100]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
