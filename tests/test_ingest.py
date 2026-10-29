"""Ingest + URL normalization + loader tests.

Uses INLINE fixtures (tests/fixtures.py) — never depends on the real
data_pack, so tests never silently skip in CI.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from kasparro_auditor.ingest.normalize import (
    domain_of,
    is_kasparro_domain,
    normalize_url,
)
from tests import fixtures as F


# ---------------------------------------------------------------------------
# normalize_url tests
# ---------------------------------------------------------------------------


def test_normalize_url_strips_fragment():
    assert normalize_url("https://example.com/path#frag") == "https://example.com/path"


def test_normalize_url_strips_utm():
    assert normalize_url(
        "https://example.com/path?utm_source=reddit&utm_medium=web&id=42"
    ) == "https://example.com/path?id=42"


def test_normalize_url_preserves_ref_param():
    """ref is a real param on GitHub — must NOT be stripped."""
    url1 = "https://github.com/repo?ref=main"
    url2 = "https://github.com/repo?ref=dev"
    assert normalize_url(url1) != normalize_url(url2)


def test_normalize_url_preserves_percent_encoding():
    """%26 must stay %26, not be decoded to &."""
    url = "https://example.com/search?q=a%26b"
    norm = normalize_url(url)
    assert "%26" in norm
    assert "&b" not in norm.split("?")[1] or "%26b" in norm


def test_normalize_url_strips_www():
    assert normalize_url("https://www.example.com/p") == "https://example.com/p"
    assert normalize_url("https://example.com/p") == "https://example.com/p"


def test_normalize_url_root_path_consistency():
    """https://x.com/ and https://x.com must merge."""
    assert normalize_url("https://example.com/") == normalize_url("https://example.com")


def test_normalize_url_strips_default_port():
    assert normalize_url("https://example.com:443/p") == "https://example.com/p"
    assert normalize_url("http://example.com:80/p") == "http://example.com/p"


def test_normalize_url_strips_trailing_slash():
    assert normalize_url("https://example.com/p/") == "https://example.com/p"


def test_normalize_url_empty():
    assert normalize_url("") == ""
    assert normalize_url("   ") == ""


# ---------------------------------------------------------------------------
# domain_of tests
# ---------------------------------------------------------------------------


def test_domain_of_strips_www():
    assert domain_of("https://www.example.com/p") == "example.com"
    assert domain_of("https://example.com/p") == "example.com"


def test_domain_of_strips_port():
    assert domain_of("https://example.com:8080/p") == "example.com"


def test_domain_of_strips_userinfo():
    """Security fix: user:pw@ must not leak into the domain."""
    assert domain_of("https://user:pw@evil.com/x") == "evil.com"


def test_domain_of_invalid():
    assert domain_of("not a url") is None
    assert domain_of("") is None


def test_is_kasparro_domain():
    assert is_kasparro_domain("kasparro.com")
    assert is_kasparro_domain("www.kasparro.com")
    assert is_kasparro_domain("api.kasparro.com")
    assert not is_kasparro_domain("notkasparro.com")
    assert not is_kasparro_domain(None)


# ---------------------------------------------------------------------------
# Loader tests — all use inline fixtures, never the real pack
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_valid_pack(session):
    """A clean import loads the expected counts."""
    from kasparro_auditor.ingest.loader import import_pack

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSES_JSONL,
            citations=F.CITATIONS_JSONL,
        )
        report = await import_pack(session, source_dir=tmp)

    assert report.prompts_new == 2
    assert report.responses_new == 3
    assert report.citations_new == 4
    assert report.responses_reused == 0
    assert report.rejected == []
    assert report.conflicts == []


@pytest.mark.asyncio
async def test_import_is_idempotent_same_content(session):
    """Re-importing the exact same pack → all reused, 0 new."""
    from kasparro_auditor.ingest.loader import import_pack

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSES_JSONL,
            citations=F.CITATIONS_JSONL,
        )
        first = await import_pack(session, source_dir=tmp)
        second = await import_pack(session, source_dir=tmp)

    assert first.dataset_id != second.dataset_id
    assert second.prompts_new == 0
    assert second.responses_new == 0
    assert second.citations_new == 0
    assert second.prompts_reused == 2
    assert second.responses_reused == 3
    assert second.citations_reused == 4
    assert second.conflicts == []


@pytest.mark.asyncio
async def test_import_detects_content_conflict(session):
    """Same response_id + different content → conflict, NOT overwrite."""
    from kasparro_auditor.ingest.loader import import_pack
    from kasparro_auditor.db.models import ResponseRow
    from sqlalchemy import select

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSES_JSONL,
            citations=F.CITATIONS_JSONL,
        )
        await import_pack(session, source_dir=tmp)

        # Re-import with a DIFFERENT answer_text for r001
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSE_DIFFERENT_CONTENT,
            citations=[],
        )
        second = await import_pack(session, source_dir=tmp)

    assert second.responses_new == 0
    assert second.responses_reused == 0  # NOT reused — content differs
    assert len(second.conflicts) == 1
    assert second.conflicts[0]["id"] == "r001"

    # The original row must NOT be overwritten
    existing = await session.scalar(
        select(ResponseRow).where(ResponseRow.response_id == "r001")
    )
    assert "Kasparro is mentioned as an AI visibility tool" in existing.answer_text
    assert "COMPLETELY DIFFERENT TEXT" not in existing.answer_text


@pytest.mark.asyncio
async def test_import_malformed_json_line_does_not_abort(session):
    """One bad JSON line → rejected, rest of the file still imports."""
    from kasparro_auditor.ingest.loader import import_pack

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL + [F.MALFORMED_JSON_LINE],  # last line is broken
            responses=F.RESPONSES_JSONL,
            citations=F.CITATIONS_JSONL,
        )
        report = await import_pack(session, source_dir=tmp)

    # The 2 valid prompts imported; the malformed line rejected
    assert report.prompts_new == 2
    assert len(report.rejected) == 1
    assert "line" in report.rejected[0]


@pytest.mark.asyncio
async def test_import_rejects_whitespace_only_text(session):
    """Whitespace-only prompt text → validation error, rejected."""
    from kasparro_auditor.ingest.loader import import_pack

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL + F.BAD_PROMPT_WHITESPACE,
            responses=F.RESPONSES_JSONL,
            citations=F.CITATIONS_JSONL,
        )
        report = await import_pack(session, source_dir=tmp)

    assert report.prompts_new == 2  # only the 2 valid ones
    assert len(report.rejected) == 1


@pytest.mark.asyncio
async def test_import_rejects_missing_required_field(session):
    """Response missing archive_stored_at → rejected."""
    from kasparro_auditor.ingest.loader import import_pack

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSES_JSONL + F.BAD_RESPONSE_MISSING_FIELD,
            citations=F.CITATIONS_JSONL,
        )
        report = await import_pack(session, source_dir=tmp)

    assert report.responses_new == 3  # only the 3 valid ones
    assert len(report.rejected) == 1


@pytest.mark.asyncio
async def test_import_rejects_uppercase_engine(session):
    """Engine 'ChatGPT' (uppercase) → rejected; contract is exact-match."""
    from kasparro_auditor.ingest.loader import import_pack

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSES_JSONL + F.BAD_RESPONSE_UPPERCASE_ENGINE,
            citations=F.CITATIONS_JSONL,
        )
        report = await import_pack(session, source_dir=tmp)

    assert report.responses_new == 3
    assert len(report.rejected) == 1


@pytest.mark.asyncio
async def test_import_rejects_orphan_citation(session):
    """Citation referencing non-existent response_id → rejected."""
    from kasparro_auditor.ingest.loader import import_pack

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSES_JSONL,
            citations=F.CITATIONS_JSONL + F.ORPHAN_CITATION,
        )
        report = await import_pack(session, source_dir=tmp)

    assert report.citations_new == 4  # only the 4 valid ones
    assert len(report.rejected) == 1
    assert "unknown response_id" in report.rejected[0]["error"]


@pytest.mark.asyncio
async def test_import_rejects_engine_mismatch(session):
    """Citation engine ≠ response engine → rejected."""
    from kasparro_auditor.ingest.loader import import_pack

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSES_JSONL,
            citations=F.CITATIONS_JSONL + F.ENGINE_MISMATCH_CITATION,
        )
        report = await import_pack(session, source_dir=tmp)

    assert report.citations_new == 4
    assert len(report.rejected) == 1
    assert "engine" in report.rejected[0]["error"].lower()


@pytest.mark.asyncio
async def test_import_rejects_prompt_mismatch(session):
    """Citation prompt_id ≠ response prompt_id → rejected."""
    from kasparro_auditor.ingest.loader import import_pack

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSES_JSONL,
            citations=F.CITATIONS_JSONL + F.PROMPT_MISMATCH_CITATION,
        )
        report = await import_pack(session, source_dir=tmp)

    assert report.citations_new == 4
    assert len(report.rejected) == 1
    assert "prompt_id" in report.rejected[0]["error"].lower()


@pytest.mark.asyncio
async def test_import_dataset_counts_reflect_actual_linked_rows(session):
    """Dataset counts reflect true linked totals, not deltas.

    This tests the fix for the bug where re-import showed count=0.
    """
    from kasparro_auditor.ingest.loader import import_pack
    from kasparro_auditor.db.models import Dataset

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSES_JSONL,
            citations=F.CITATIONS_JSONL,
        )
        report = await import_pack(session, source_dir=tmp)
        ds = await session.get(Dataset, report.dataset_id)

    # Counts reflect the true totals, not "new in this run"
    assert ds.prompt_count == 2
    assert ds.response_count == 3
    assert ds.citation_count == 4
    assert ds.rejected_count == 0


@pytest.mark.asyncio
async def test_import_preserves_raw_json(session):
    """Raw JSON is stored in raw_json column (brief: 'persist raw inputs')."""
    from kasparro_auditor.ingest.loader import import_pack
    from kasparro_auditor.db.models import PromptRow
    from sqlalchemy import select

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSES_JSONL,
            citations=F.CITATIONS_JSONL,
        )
        await import_pack(session, source_dir=tmp)

    row = await session.scalar(
        select(PromptRow).where(PromptRow.prompt_id == "p001")
    )
    assert row.raw_json is not None
    raw = json.loads(row.raw_json)
    assert raw["prompt_id"] == "p001"
    assert raw["text"] == "does kasparro work?"


@pytest.mark.asyncio
async def test_import_stores_content_hash(session):
    """Content hash is computed and stored for conflict detection."""
    from kasparro_auditor.ingest.loader import import_pack
    from kasparro_auditor.db.models import ResponseRow
    from sqlalchemy import select

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSES_JSONL,
            citations=F.CITATIONS_JSONL,
        )
        await import_pack(session, source_dir=tmp)

    row = await session.scalar(
        select(ResponseRow).where(ResponseRow.response_id == "r001")
    )
    assert row.content_hash is not None
    assert len(row.content_hash) == 16  # truncated SHA-256


@pytest.mark.asyncio
async def test_import_normalizes_domain_at_ingest(session):
    """Domain is normalized (www. stripped) at ingest time."""
    from kasparro_auditor.ingest.loader import import_pack
    from kasparro_auditor.db.models import CitationRow
    from sqlalchemy import select

    with tempfile.TemporaryDirectory() as tmp:
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSES_JSONL,
            citations=F.CITATIONS_JSONL,
        )
        await import_pack(session, source_dir=tmp)

    # c004 has url=https://www.athenahq.ai/
    row = await session.scalar(
        select(CitationRow).where(CitationRow.citation_id == "c004")
    )
    assert row.domain == "athenahq.ai"  # www. stripped
    assert row.normalized_url == "https://athenahq.ai/"  # root path keeps trailing slash


@pytest.mark.asyncio
async def test_import_count_warnings_on_mismatch(session):
    """Warns when imported totals differ from run.json expected counts."""
    from kasparro_auditor.ingest.loader import import_pack

    with tempfile.TemporaryDirectory() as tmp:
        # run.json says 100 prompts but we only have 2
        F.write_pack(
            Path(tmp),
            prompts=F.PROMPTS_JSONL,
            responses=F.RESPONSES_JSONL,
            citations=F.CITATIONS_JSONL,
            run_json={
                "run_id": "test-run-001",
                "original_run_counts": {"prompts": 100, "responses": 3, "citations": 4},
            },
        )
        report = await import_pack(session, source_dir=tmp)

    assert len(report.count_warnings) == 1
    assert "prompts" in report.count_warnings[0]
    assert "expected=100" in report.count_warnings[0]
