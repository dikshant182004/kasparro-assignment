"""Stub adapter and evidence verification tests (deterministic, no API key)."""
from __future__ import annotations

import pytest

from kasparro_auditor.llm.adapter import (
    LLMContext,
    MalformedLLMOutput,
    verify_evidence,
)
from kasparro_auditor.llm.stub_adapter import StubAdapter
from kasparro_auditor.schemas.results import ClassificationResult


def _ctx(text: str, response_id: str = "r-test") -> LLMContext:
    return LLMContext(
        response_id=response_id,
        prompt_id="p-test",
        engine="chatgpt",
        prompt_text="does kasparro work?",
        answer_text=text,
        competitor_hints=["profound", "otterly ai"],
    )


def test_verify_evidence_accepts_genuine_excerpt():
    text = "Kasparro is a great AI visibility tool for B2B SaaS companies."
    result = ClassificationResult(
        mention=True,
        recommendation=True,
        negative_characterisation=False,
        recommendation_position=1,
        competitors_named=["profound"],
        evidence_excerpt="Kasparro is a great AI visibility tool",
        char_offset_start=0,
        char_offset_end=38,
    )
    verify_evidence(result, text)  # does not raise


def test_verify_evidence_rejects_nonexistent_excerpt():
    text = "Kasparro is mentioned here in a longer answer for testing."
    result = ClassificationResult(
        mention=True,
        recommendation=False,
        negative_characterisation=False,
        recommendation_position=None,
        competitors_named=[],
        evidence_excerpt="This excerpt does not appear in the answer at all.",
        char_offset_start=0,
        char_offset_end=50,
    )
    with pytest.raises(MalformedLLMOutput):
        verify_evidence(result, text)


def test_verify_evidence_rejects_short_excerpt():
    text = "Kasparro is mentioned here in a longer answer for testing."
    result = ClassificationResult(
        mention=True,
        recommendation=False,
        negative_characterisation=False,
        recommendation_position=None,
        competitors_named=[],
        evidence_excerpt="short",
        char_offset_start=0,
        char_offset_end=5,
    )
    with pytest.raises(MalformedLLMOutput):
        verify_evidence(result, text)


def test_verify_evidence_rejects_mismatched_offsets():
    text = "Kasparro is mentioned here in this answer."
    result = ClassificationResult(
        mention=True,
        recommendation=False,
        negative_characterisation=False,
        recommendation_position=None,
        competitors_named=[],
        evidence_excerpt="Kasparro is mentioned here",
        char_offset_start=5,  # wrong
        char_offset_end=28,
    )
    with pytest.raises(MalformedLLMOutput, match="char_offset_start/end"):
        verify_evidence(result, text)


@pytest.mark.asyncio
async def test_stub_adapter_classifies_mention():
    text = (
        "Based on my research, Kasparro is a great option for tracking AI visibility. "
        "It is the top pick among B2B SaaS tools."
    )
    adapter = StubAdapter()
    result = await adapter.classify(_ctx(text))
    assert result.mention is True
    assert result.recommendation is True
    assert result.evidence_excerpt in text


@pytest.mark.asyncio
async def test_stub_adapter_detects_negative():
    text = (
        "Kasparro is mentioned but it's just hype and not legit. "
        "Several users call it a gimmick."
    )
    adapter = StubAdapter()
    result = await adapter.classify(_ctx(text))
    assert result.mention is True
    assert result.negative_characterisation is True
    assert result.recommendation is False


@pytest.mark.asyncio
async def test_stub_adapter_handles_absence():
    text = (
        "There are several tools in the AI visibility space including "
        "Profound and Otterly AI. The best option depends on use case."
    )
    adapter = StubAdapter()
    result = await adapter.classify(_ctx(text))
    assert result.mention is False
    assert result.recommendation is False
    assert "profound" in result.competitors_named
    assert "otterly ai" in result.competitors_named
