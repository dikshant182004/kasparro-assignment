"""Stub adapter and evidence verification tests (deterministic, no API key)."""
from __future__ import annotations

import pytest

from kasparro_auditor.llm.adapter import (
    EvidenceUnsupported,
    LLMContext,
    MalformedLLMOutput,
    Judgment,
    verify_evidence,
)


def _ctx(text: str, response_id: str = "r-test", mention: bool = True) -> LLMContext:
    return LLMContext(
        response_id=response_id,
        prompt_id="p-test",
        engine="chatgpt",
        prompt_text="does kasparro work?",
        answer_text=text,
        mention=mention,
        prompt_names_brand=True,
        competitors_known=["profound"],
    )


def _judgment(
    recommendation: bool = False,
    negative: bool = False,
    position: int | None = None,
    competitors_extra: list[str] | None = None,
    evidence_mention: str = "",
    evidence_recommendation: str = "",
    evidence_negative: str = "",
) -> Judgment:
    return Judgment(
        recommendation=recommendation,
        negative_characterisation=negative,
        recommendation_position=position,
        competitors_extra=competitors_extra or [],
        evidence_mention=evidence_mention,
        evidence_recommendation=evidence_recommendation,
        evidence_negative=evidence_negative,
    )


def test_verify_evidence_accepts_genuine_excerpt():
    text = "Kasparro is a great AI visibility tool for B2B SaaS companies."
    j = _judgment(
        recommendation=True,
        evidence_mention="Kasparro is a great AI visibility tool",
        evidence_recommendation="Kasparro is a great AI visibility tool",
    )
    verify_evidence(j, text)  # does not raise


def test_verify_evidence_rejects_nonexistent_excerpt():
    text = "Kasparro is mentioned here in a longer answer for testing."
    j = _judgment(
        evidence_mention="This excerpt does not appear in the answer at all.",
    )
    with pytest.raises(EvidenceUnsupported):
        verify_evidence(j, text)


def test_verify_evidence_rejects_long_excerpt():
    text = "Kasparro is mentioned here."
    j = _judgment(
        evidence_mention="x" * 400,  # > 300 chars
    )
    with pytest.raises(EvidenceUnsupported, match="300"):
        verify_evidence(j, text)


def test_verify_evidence_allows_empty_when_label_false():
    text = "Some answer text without Kasparro."
    j = _judgment()  # all labels False, all evidence empty
    verify_evidence(j, text)  # does not raise


def test_judgment_cross_field_validator_evidence_required_when_recommendation():
    """Judgment should reject recommendation=True without evidence_recommendation."""
    with pytest.raises(ValueError, match="evidence_recommendation is required"):
        Judgment(
            recommendation=True,
            negative_characterisation=False,
            evidence_mention="Kasparro is mentioned.",
            evidence_recommendation="",  # missing
        )


def test_judgment_cross_field_validator_evidence_required_when_negative():
    """Judgment should reject negative=True without evidence_negative."""
    with pytest.raises(ValueError, match="evidence_negative is required"):
        Judgment(
            recommendation=False,
            negative_characterisation=True,
            evidence_mention="Kasparro is mentioned.",
            evidence_negative="",  # missing
        )


@pytest.mark.asyncio
async def test_stub_adapter_classifies_mention_and_recommendation():
    text = (
        "Kasparro is a great option for tracking AI visibility. "
        "It is the top pick among B2B SaaS tools."
    )
    adapter = _make_stub()
    result = await adapter.classify(_ctx(text))
    assert result.recommendation is True
    assert result.evidence_mention in text
    assert result.evidence_recommendation in text


@pytest.mark.asyncio
async def test_stub_adapter_detects_negative():
    text = (
        "Kasparro is mentioned but it's just hype and not legit. "
        "Several users call it a gimmick."
    )
    adapter = _make_stub()
    result = await adapter.classify(_ctx(text))
    assert result.negative_characterisation is True
    assert result.recommendation is False
    assert result.evidence_negative in text


def _make_stub():
    """Build a StubAdapter without triggering circular imports."""
    from kasparro_auditor.llm.stub_adapter import StubAdapter
    return StubAdapter()
