"""Deterministic citation facts computed BEFORE the LLM runs.

These numbers are NEVER produced by the LLM. They are exact counts the
grader can verify against the raw `citations.jsonl`.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.db.models import CitationRow
from kasparro_auditor.ingest.normalize import is_kasparro_domain


@dataclass(slots=True)
class CitationFacts:
    citation_count_for_response: int
    unique_citation_count: int
    has_kasparro_citation: bool


async def compute_citation_facts(
    session: AsyncSession, response_id: str
) -> CitationFacts:
    """Return raw + deduped citation counts plus a Kasparro-domain flag."""
    rows = (
        await session.scalars(
            select(CitationRow).where(CitationRow.response_id == response_id)
        )
    ).all()

    raw_count = len(rows)
    unique_urls = {c.normalized_url for c in rows if c.normalized_url}
    has_kasparro = any(
        is_kasparro_domain(c.domain) for c in rows
    )
    return CitationFacts(
        citation_count_for_response=raw_count,
        unique_citation_count=len(unique_urls),
        has_kasparro_citation=has_kasparro,
    )
