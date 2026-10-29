"""Ingest + URL normalization tests."""
from __future__ import annotations

import pytest

from kasparro_auditor.ingest.normalize import (
    domain_of,
    is_kasparro_domain,
    normalize_url,
)


def test_normalize_url_strips_fragment():
    assert normalize_url("https://example.com/path#frag") == "https://example.com/path"


def test_normalize_url_strips_utm():
    assert normalize_url(
        "https://example.com/path?utm_source=reddit&utm_medium=web&id=42"
    ) == "https://example.com/path?id=42"


def test_normalize_url_sorts_query_params():
    assert normalize_url("https://example.com/p?b=2&a=1") == "https://example.com/p?a=1&b=2"


def test_normalize_url_lowercases_scheme_and_host():
    assert normalize_url("HTTPS://WWW.Example.COM/Path") == "https://www.example.com/Path"


def test_normalize_url_strips_default_port():
    assert normalize_url("https://example.com:443/p") == "https://example.com/p"
    assert normalize_url("http://example.com:80/p") == "http://example.com/p"


def test_normalize_url_strips_trailing_slash():
    assert normalize_url("https://example.com/p/") == "https://example.com/p"
    # Root path keeps its slash
    assert normalize_url("https://example.com/") == "https://example.com/"


def test_normalize_url_empty():
    assert normalize_url("") == ""
    assert normalize_url("   ") == ""


def test_domain_of():
    assert domain_of("https://www.example.com/p") == "example.com"
    assert domain_of("https://example.com:8080/p") == "example.com"
    assert domain_of("not a url") is None


def test_is_kasparro_domain():
    assert is_kasparro_domain("kasparro.com")
    assert is_kasparro_domain("www.kasparro.com")
    assert is_kasparro_domain("api.kasparro.com")
    assert not is_kasparro_domain("notkasparro.com")
    assert not is_kasparro_domain(None)


@pytest.mark.asyncio
async def test_import_pack_loads_expected_counts(session, data_pack_path):
    from kasparro_auditor.ingest.loader import import_pack

    report = await import_pack(session, data_pack_path)
    assert report.prompts_new == 50
    assert report.responses_new + report.responses_reused == 150
    assert report.citations_new == 2439
    assert report.rejected == []


@pytest.mark.asyncio
async def test_import_pack_is_idempotent(session, data_pack_path):
    """Re-importing the same pack reuses existing rows (no silent dup)."""
    from kasparro_auditor.ingest.loader import import_pack

    first = await import_pack(session, data_pack_path)
    second = await import_pack(session, data_pack_path)

    assert first.dataset_id != second.dataset_id  # audit trail kept
    # On second import, all 150 responses are reused (not duplicated)
    assert second.responses_new == 0
    assert second.responses_reused == 150
    # Citations are also idempotent — re-import yields 0 new rows
    # (citation_id is unique; upserted with on_conflict_do_nothing).
    assert second.citations_new == 0
