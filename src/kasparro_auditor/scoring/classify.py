"""Classification orchestration: Facts (deterministic) + Judgment (LLM).

Per-record isolation: every exception is caught and recorded. One bad
record NEVER aborts the run. Every one of the 150 responses ends as
success, failed, or skipped.

Skip-LLM-when-absent optimization: if Facts.mention is False, we don't
call the LLM at all. 78 of 150 responses don't name Kasparro. This
cuts calls from 150 → 72 and removes the fake "evidence of absence"
excerpt problem.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.config import ExecutionMode, get_settings
from kasparro_auditor.db.models import (
    AnalysisRun,
    Engine,
    ErrorCode,
    ExecutionMode as DBExecutionMode,
    PromptRow,
    ResponseResult,
    ResponseRow,
)
from kasparro_auditor.llm import (
    EvidenceUnsupported,
    LLMAdapter,
    LLMContext,
    LLMError,
    LLMProviderError,
    LLMTimeout,
    MalformedLLMOutput,
    ReplaySkip,
    build_adapter,
    prompt_hash,
    verify_evidence,
)
from kasparro_auditor.scoring.facts import (
    Facts,
    compute_mention,
    compute_competitors,
    strip_trailing_source_block,
)
from kasparro_auditor.scoring.prompt_categories import (
    infer_prompt_category,
    prompt_names_brand,
)
from kasparro_auditor.scoring.counts import CitationFacts


async def classify_one(
    session: AsyncSession,
    run: AnalysisRun,
    response: ResponseRow,
    prompt: PromptRow,
    adapter: LLMAdapter,
    precomputed_citations: CitationFacts | None = None,
) -> ResponseResult:
    """Classify one response and persist the result row.

    Every exception is caught. One bad record never aborts the run.
    """
    # 1. Compute deterministic Facts (no LLM)
    mention, mention_spans = compute_mention(response.answer_text)
    competitors = compute_competitors(response.answer_text)
    p_names_brand = prompt_names_brand(prompt.text)
    p_category = infer_prompt_category(prompt.text)

    # Use pre-computed citation facts if provided (avoids 150 ORM queries)
    if precomputed_citations is not None:
        facts_citations = precomputed_citations
    else:
        from kasparro_auditor.scoring.counts import compute_citation_facts
        facts_citations = await compute_citation_facts(session, response.response_id)

    # 2. Strip trailing [n] url source block before sending to LLM (saves tokens)
    trimmed_answer = strip_trailing_source_block(response.answer_text)

    # 3. Skip LLM if mention is False (78/150 responses)
    #    Record as a successful zero-judgment row, NOT as a failure.
    if not mention:
        rr = ResponseResult(
            run_id=run.id,
            response_id=response.response_id,
            prompt_id=response.prompt_id,
            engine=response.engine,
            mention=False,
            mention_disagreement=False,
            mention_spans=[{"start": s, "end": e} for s, e in mention_spans],
            prompt_names_brand=p_names_brand,
            prompt_category=p_category,
            competitors_known=competitors,
            recommendation=False,
            negative_characterisation=False,
            recommendation_position=None,
            competitors_extra=[],
            evidence_mention="",
            evidence_recommendation="",
            evidence_negative="",
            citation_count_for_response=facts_citations.citation_count_for_response,
            unique_citation_count=facts_citations.unique_citation_count,
            has_kasparro_citation=facts_citations.has_kasparro_citation,
            failed=False,
            skipped=False,
            execution_mode=DBExecutionMode(adapter.mode),
            model_name=getattr(adapter, "model_name", None),
            attempts=0,
            latency_ms=0,
            prompt_hash=prompt_hash(),
            scoring_version="v2",
        )
        session.add(rr)
        return rr

    # 4. Build context with Facts pre-computed
    ctx = LLMContext(
        response_id=response.response_id,
        prompt_id=response.prompt_id,
        engine=response.engine.value,
        prompt_text=prompt.text,
        answer_text=trimmed_answer,
        mention=mention,
        prompt_names_brand=p_names_brand,
        competitors_known=competitors,
    )

    started = time.perf_counter()
    judgment = None
    actual_mode = adapter.mode
    actual_model = getattr(adapter, "model_name", None)
    attempts = 0
    error_code: ErrorCode | None = None
    failure_reason: str | None = None
    skipped = False
    skip_reason: str | None = None

    # 5. Call the LLM with per-record catch-all isolation
    try:
        judgment = await adapter.classify(ctx)
        # Per-call model name (not the adapter default)
        actual_model = getattr(judgment, "_model_used", actual_model)
        actual_mode = getattr(judgment, "_actual_mode", actual_mode)
        attempts = getattr(judgment, "_attempts", 1)

        # Verify evidence locally — never trust the LLM to vouch for itself
        verify_evidence(judgment, response.answer_text, response_id=response.response_id)

    except ReplaySkip as err:
        # No cached output — skip, NOT fail
        skipped = True
        skip_reason = err.reason
        error_code = ErrorCode.NO_RECORDED_OUTPUT
        actual_mode = "replay"
    except LLMTimeout as err:
        failed = True
        failure_reason = str(err)
        error_code = err.error_code
    except MalformedLLMOutput as err:
        failed = True
        failure_reason = str(err)
        error_code = err.error_code
    except EvidenceUnsupported as err:
        failed = True
        failure_reason = str(err)
        error_code = err.error_code
    except LLMProviderError as err:
        failed = True
        failure_reason = str(err)
        error_code = err.error_code
    except LLMError as err:
        failed = True
        failure_reason = str(err)
        error_code = err.error_code
    except Exception as err:
        # Per-record isolation: catch EVERYTHING. One bad row never aborts the run.
        failed = True
        failure_reason = f"Unexpected error: {type(err).__name__}: {err}"
        error_code = ErrorCode.UNEXPECTED
    else:
        failed = False

    latency_ms = int((time.perf_counter() - started) * 1000)

    # 6. Persist the result row
    if judgment is not None and not failed and not skipped:
        # Success path — combine Facts + Judgment
        rr = ResponseResult(
            run_id=run.id,
            response_id=response.response_id,
            prompt_id=response.prompt_id,
            engine=response.engine,
            mention=mention,  # from Facts (regex), never overridden by LLM
            mention_disagreement=False,  # LLM doesn't return mention anymore
            mention_spans=[{"start": s, "end": e} for s, e in mention_spans],
            prompt_names_brand=p_names_brand,
            prompt_category=p_category,
            competitors_known=competitors,
            recommendation=judgment.recommendation,
            negative_characterisation=judgment.negative_characterisation,
            recommendation_position=judgment.recommendation_position,
            competitors_extra=judgment.competitors_extra,
            evidence_mention=judgment.evidence_mention,
            evidence_recommendation=judgment.evidence_recommendation,
            evidence_negative=judgment.evidence_negative,
            citation_count_for_response=facts_citations.citation_count_for_response,
            unique_citation_count=facts_citations.unique_citation_count,
            has_kasparro_citation=facts_citations.has_kasparro_citation,
            failed=False,
            skipped=False,
            execution_mode=DBExecutionMode(actual_mode),
            model_name=actual_model,
            attempts=attempts,
            latency_ms=latency_ms,
            prompt_hash=prompt_hash(),
            scoring_version="v2",
        )
    elif skipped:
        rr = ResponseResult(
            run_id=run.id,
            response_id=response.response_id,
            prompt_id=response.prompt_id,
            engine=response.engine,
            mention=mention,
            mention_disagreement=False,
            mention_spans=[{"start": s, "end": e} for s, e in mention_spans],
            prompt_names_brand=p_names_brand,
            prompt_category=p_category,
            competitors_known=competitors,
            citation_count_for_response=facts_citations.citation_count_for_response,
            unique_citation_count=facts_citations.unique_citation_count,
            has_kasparro_citation=facts_citations.has_kasparro_citation,
            failed=False,
            skipped=True,
            skip_reason=skip_reason,
            error_code=error_code,
            execution_mode=DBExecutionMode(actual_mode),
            model_name=actual_model,
            latency_ms=latency_ms,
            prompt_hash=prompt_hash(),
            scoring_version="v2",
        )
    else:
        # Failed path — never manufacture a confident zero-score
        rr = ResponseResult(
            run_id=run.id,
            response_id=response.response_id,
            prompt_id=response.prompt_id,
            engine=response.engine,
            mention=mention,
            mention_disagreement=False,
            mention_spans=[{"start": s, "end": e} for s, e in mention_spans],
            prompt_names_brand=p_names_brand,
            prompt_category=p_category,
            competitors_known=competitors,
            citation_count_for_response=facts_citations.citation_count_for_response,
            unique_citation_count=facts_citations.unique_citation_count,
            has_kasparro_citation=facts_citations.has_kasparro_citation,
            failed=True,
            skipped=False,
            failure_reason=failure_reason,
            error_code=error_code,
            execution_mode=DBExecutionMode(actual_mode),
            model_name=actual_model,
            attempts=attempts,
            latency_ms=latency_ms,
            prompt_hash=prompt_hash(),
            scoring_version="v2",
        )

    session.add(rr)
    return rr


async def run_analysis(
    session: AsyncSession,
    run: AnalysisRun,
    dataset_id: str,
    adapter: LLMAdapter | None = None,
) -> AnalysisRun:
    """Classify every response in the dataset, concurrently with a bound.

    Pre-computes ALL citation facts in ONE query (not 150) and passes
    them into classify_one. This removes DB access from the concurrent
    tasks (fixes the shared-AsyncSession warning).
    """
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

    # Pre-compute ALL citation facts in ONE query, group by response_id.
    # This replaces 150 ORM queries and removes DB access from concurrent tasks.
    from kasparro_auditor.db.models import CitationRow
    from kasparro_auditor.ingest.normalize import is_kasparro_domain

    response_ids = [r[0].response_id for r in rows]
    citation_rows = (
        await session.scalars(
            select(CitationRow).where(CitationRow.response_id.in_(response_ids))
        )
    ).all() if response_ids else []

    citations_by_response: dict[str, CitationFacts] = {}
    # Track unique URLs per response in a side dict (CitationFacts is slots=True)
    unique_urls_by_response: dict[str, set[str]] = {}
    for c in citation_rows:
        rid = c.response_id
        if rid not in citations_by_response:
            citations_by_response[rid] = CitationFacts(
                citation_count_for_response=0,
                unique_citation_count=0,
                has_kasparro_citation=False,
            )
            unique_urls_by_response[rid] = set()
        facts = citations_by_response[rid]
        facts.citation_count_for_response += 1
        unique_urls_by_response[rid].add(c.normalized_url)
        if is_kasparro_domain(c.domain):
            facts.has_kasparro_citation = True
    for rid, urls in unique_urls_by_response.items():
        citations_by_response[rid].unique_citation_count = len(urls)

    sem = asyncio.Semaphore(settings.max_concurrent_llm_calls)

    async def _bounded(rr_tuple) -> ResponseResult:
        async with sem:
            response, prompt = rr_tuple
            precomputed = citations_by_response.get(response.response_id)
            # Per-record isolation: catch ALL exceptions so the run never aborts
            try:
                return await classify_one(
                    session, run, response, prompt, adapter, precomputed
                )
            except Exception as err:
                # Last-resort catch — should be rare since classify_one already catches
                from kasparro_auditor.db.models import ErrorCode as EC
                rr = ResponseResult(
                    run_id=run.id,
                    response_id=response.response_id,
                    prompt_id=response.prompt_id,
                    engine=response.engine,
                    failed=True,
                    failure_reason=f"Orchestration-level error: {type(err).__name__}: {err}",
                    error_code=EC.UNEXPECTED,
                    execution_mode=DBExecutionMode(adapter.mode),
                    model_name=getattr(adapter, "model_name", None),
                    prompt_hash=prompt_hash(),
                    scoring_version="v2",
                )
                session.add(rr)
                return rr

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
    run.model_name = getattr(adapter, "model_name", None)
    run.prompt_hash = prompt_hash()
    await session.commit()
    return run


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
