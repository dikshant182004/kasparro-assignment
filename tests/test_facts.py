"""Tests for the deterministic Facts layer + prompt categorization."""
from __future__ import annotations

import pytest

from kasparro_auditor.scoring.facts import (
    compute_mention,
    compute_competitors,
    strip_trailing_source_block,
)
from kasparro_auditor.scoring.prompt_categories import (
    infer_prompt_category,
    prompt_names_brand,
)


# ---------------------------------------------------------------------------
# compute_mention (word-boundary regex)
# ---------------------------------------------------------------------------


def test_compute_mention_finds_brand():
    mention, spans = compute_mention("Kasparro is a great tool.")
    assert mention is True
    assert len(spans) == 1
    assert spans[0] == (0, 8)


def test_compute_mention_case_insensitive():
    mention, _ = compute_mention("kasparro is mentioned here.")
    assert mention is True


def test_compute_mention_multiple_occurrences():
    mention, spans = compute_mention("Kasparro here. Kasparro there.")
    assert mention is True
    assert len(spans) == 2


def test_compute_mention_word_boundary_no_false_positives():
    """'Kasparros' (with trailing 's') should NOT match 'kasparro'."""
    mention, spans = compute_mention("Kasparros is a different company.")
    assert mention is False
    assert spans == []


def test_compute_mention_absent():
    mention, spans = compute_mention("No brand named here.")
    assert mention is False
    assert spans == []


# ---------------------------------------------------------------------------
# compute_competitors (alias matching)
# ---------------------------------------------------------------------------


def test_compute_competitors_finds_profound():
    found = compute_competitors("Profound is a popular alternative.")
    assert "profound" in found


def test_compute_competitors_finds_multiple():
    found = compute_competitors("Profound and Otterly AI are both options.")
    assert "profound" in found
    assert "otterly ai" in found


def test_compute_competitors_canonicalises_alias():
    """'athena hq' should canonicalize to 'athenahq'."""
    found = compute_competitors("Athena HQ is one option.")
    assert "athenahq" in found


def test_compute_competitors_dedupes():
    """Multiple aliases of the same competitor should produce one entry."""
    found = compute_competitors("Otterly AI is great. Otterly also has features.")
    assert found.count("otterly ai") == 1


def test_compute_competitors_empty_when_none_found():
    found = compute_competitors("No competitor names here.")
    assert found == []


# ---------------------------------------------------------------------------
# prompt_names_brand
# ---------------------------------------------------------------------------


def test_prompt_names_brand_true():
    assert prompt_names_brand("does kasparro actually work?") is True


def test_prompt_names_brand_false_for_generic():
    assert prompt_names_brand("what tools measure AI visibility?") is False


def test_prompt_names_brand_word_boundary():
    """'kasparros' should NOT match 'kasparro'."""
    assert prompt_names_brand("kasparros are chess pieces.") is False


# ---------------------------------------------------------------------------
# infer_prompt_category
# ---------------------------------------------------------------------------


def test_category_pricing():
    assert infer_prompt_category("how much does kasparro cost?") == "pricing"


def test_category_legitimacy():
    assert infer_prompt_category("is kasparro legit or a scam?") == "legitimacy"


def test_category_comparison():
    assert infer_prompt_category("is kasparro better than profound?") == "comparison"


def test_category_onboarding():
    assert infer_prompt_category("how long does setup take?") == "onboarding"


def test_category_how_to():
    assert infer_prompt_category("how do brands track AI recommendations?") == "how_to"


def test_category_tracking():
    assert infer_prompt_category("what tools measure AI visibility?") == "tracking"


def test_category_features():
    assert infer_prompt_category("what features should I look for?") == "features"


def test_category_general_fallback():
    assert infer_prompt_category("some random text with no keywords") == "general"


# ---------------------------------------------------------------------------
# strip_trailing_source_block
# ---------------------------------------------------------------------------


def test_strip_trailing_source_block_basic():
    text = "Kasparro is great.\n\n[1] https://example.com/foo\n[2] https://example.com/bar"
    result = strip_trailing_source_block(text)
    assert "[1]" not in result
    assert "Kasparro is great." in result


def test_strip_trailing_source_block_citations_header():
    text = "Kasparro is great.\n\nCitations:\n[1] https://example.com/foo"
    result = strip_trailing_source_block(text)
    assert "Citations:" not in result
    assert "Kasparro is great." in result


def test_strip_trailing_source_block_no_change_when_absent():
    text = "Kasparro is great. No source block here."
    result = strip_trailing_source_block(text)
    assert result == text


def test_strip_trailing_source_block_preserves_prose():
    """The prose must remain intact so evidence excerpts stay verbatim."""
    text = (
        "Kasparro is a great AI visibility tool. "
        "It tracks mentions across engines.\n\n"
        "[1] https://example.com/foo"
    )
    result = strip_trailing_source_block(text)
    assert "Kasparro is a great AI visibility tool." in result
    assert "It tracks mentions across engines." in result
