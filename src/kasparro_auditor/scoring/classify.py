"""Classification orchestration: one response → one ResponseResult.

This module wires the LLM adapter into the pipeline. Per response:
  1. Compute deterministic citation facts (no LLM).
  2. Build the LLMContext.
  3. Call the adapter (with timeout + retry handled inside the adapter).
  4. On success: persist the result with full traceability metadata.
  5. On failure: persist a `failed=True` row with the reason — NEVER
     manufacture a confident zero-score observation.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.config import get_settings
from kasparro_auditor.db.models import (
    AnalysisRun,
    Engine,
    ExecutionMode,
    PromptRow,
    ResponseResult,
    ResponseRow,
)
from kasparro_auditor.llm import (
    LLMAdapter,
    LLMContext,
    LLMError,
    build_adapter,
    classify_response,
    prompt_hash,
)
from kasparro_auditor.scoring.competitors import COMPETITOR_HINTS
from kasparro_auditor.scoring.counts import compute_citation_facts


async def classify_one(
    session: AsyncSession,
    run: AnalysisRun,
    response: ResponseRow,
    prompt: PromptRow,
    adapter: LLMAdapter,
) -> ResponseResult:
    """Classify one response and persist the result row."""
    # 1. Deterministic citation facts (always succeed — no LLM involved)
    facts = await compute_citation_facts(session, response.response_id)

    # 2. Build context
    ctx = LLMContext(
        response_id=response.response_id,
        prompt_id=response.prompt_id,
        engine=response.engine.value,
        prompt_text=prompt.text,
        answer_text=response.answer_text,
        competitor_hints=COMPETITOR_HINTS,
    )

    started = time.perf_counter()
    # 3. Call the adapter
    try:
        result = await classify_response(adapter, ctx)
        latency_ms = int((time.perf_counter() - started) * 1000)

        # 4a. Persist success
        rr = ResponseResult(
            run_id=run.id,
            response_id=response.response_id,
            prompt_id=response.prompt_id,
            engine=response.engine,
            mention=result.mention,
            recommendation=result.recommendation,
            negative_characterisation=result.negative_characterisation,
            recommendation_position=result.recommendation_position,
            competitors_named=result.competitors_named,
            competitors_observed_count=len(result.competitors_named),
            evidence_excerpt=result.evidence_excerpt,
            char_offset_start=result.char_offset_start,
            char_offset_end=result.char_offset_end,
            citation_count_for_response=facts.citation_count_for_response,
            unique_citation_count=facts.unique_citation_count,
            has_kasparro_citation=facts.has_kasparro_citation,
            kasparro_domain_in_citations=facts.has_kasparro_citation,
            execution_mode=ExecutionMode(adapter.mode),
            model_name=adapter.model_name,
            latency_ms=latency_ms,
            raw_llm_output=result.model_dump(),
            prompt_hash=prompt_hash(),
            failed=False,
            skipped=False,
        )
    except LLMError as err:
        # 4b. Persist failure — never manufacture a result
        rr = ResponseResult(
            run_id=run.id,
            response_id=response.response_id,
            prompt_id=response.prompt_id,
            engine=response.engine,
            citation_count_for_response=facts.citation_count_for_response,
            unique_citation_count=facts.unique_citation_count,
            has_kasparro_citation=facts.has_kasparro_citation,
            kasparro_domain_in_citations=facts.has_kasparro_citation,
            execution_mode=ExecutionMode(adapter.mode),
            model_name=adapter.model_name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            prompt_hash=prompt_hash(),
            failed=True,
            failure_reason=str(err)[:1000],
            skipped=False,
        )
    session.add(rr)
    return rr


async def run_analysis(
    session: AsyncSession,
    run: AnalysisRun,
    dataset_id: str,
    adapter: LLMAdapter | None = None,
) -> AnalysisRun:
    """Classify every response in the dataset, concurrently with a bound."""
    settings = get_settings()
    adapter = adapter or build_adapter(ExecutionMode(run.execution_mode))

    # Pull all responses + their prompts in one query.
    stmt = (
        select(ResponseRow, PromptRow)
        .join(PromptRow, ResponseRow.prompt_id == PromptRow.prompt_id)
        .where(ResponseRow.dataset_id == dataset_id)
        .order_by(ResponseRow.prompt_id, ResponseRow.engine)
    )
    rows = (await session.execute(stmt)).all()
    run.total = len(rows)

    sem = asyncio.Semaphore(settings.max_concurrent_llm_calls)

    async def _bounded(rr_tuple) -> ResponseResult:
        async with sem:
            return await classify_one(session, run, rr_tuple[0], rr_tuple[1], adapter)

    # Run in chunks to keep the session transaction manageable.
    chunk_size = 10
    successful = failed = skipped = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        results: Iterable[ResponseResult] = await asyncio.gather(
            *(_bounded(c) for c in chunk), return_exceptions=False
        )
        for r in results:
            if r.failed:
                failed += 1
            elif r.skipped:
                skipped += 1
            else:
                successful += 1
        await session.commit()  # flush chunk

    run.successful = successful
    run.failed = failed
    run.skipped = skipped
    run.status = "completed"
    run.finished_at = run.finished_at or _now()
    run.model_name = adapter.model_name
    run.prompt_hash = prompt_hash()
    await session.commit()
    return run


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
