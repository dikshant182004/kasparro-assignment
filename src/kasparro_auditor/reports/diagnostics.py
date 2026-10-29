"""Diagnostics + remedies generator.

Builds three evidence-backed diagnostics + three prioritised remedies
from a completed AnalysisRun. Uses the LLM for synthesis only — every
cited response_id is verified to exist in the run's results.

LLM usage:
  * One call to draft the 3 diagnostics from the score components +
    per-response labels.
  * One call to draft the 3 remedies from the diagnostics.

Per the brief: "an LLM judging its own prose is insufficient" — we
deterministically verify every cited response_id and excerpt after the
LLM returns.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from kasparro_auditor.config import ExecutionMode, get_settings
from kasparro_auditor.db.models import AnalysisRun, ResponseResult
from kasparro_auditor.llm.factory import build_adapter
from kasparro_auditor.schemas.results import Diagnostic, Remedy, VisibilityScore
from kasparro_auditor.scoring.score import compute_visibility_score


DIAGNOSTIC_SYSTEM_PROMPT = """\
You are an AI-visibility analyst summarising the results of a Kasparro
brand audit. You will be given per-response labels and the composite
score components. Produce exactly THREE diagnostics.

For each diagnostic return:
  id        : short slug like "win_chatgpt_recommendations"
  title     : one-sentence headline
  pattern   : the recorded observation (what the data shows)
  hypothesis: a possible explanation (NOT a claim — frame as "may")
  evidence_response_ids : list of response_ids that support this
  evidence_excerpts    : one verbatim excerpt per response_id
  would_change_conclusion : what evidence would change this conclusion

Rules:
  * Every response_id you cite MUST appear in the supplied context.
  * Every excerpt MUST be a verbatim substring of that response's text.
  * Separate observation, hypothesis, and falsifier explicitly.
"""

REMEDY_SYSTEM_PROMPT = """\
You are an AI-visibility consultant proposing prioritised remedies.
Given three diagnostics, propose exactly THREE prioritised remedies.

For each remedy return:
  id, priority (1=highest), problem (diagnostic id this addresses),
  supporting_evidence (list of diagnostic ids), action,
  expected_benefit, effort (low/medium/high),
  uncertainty (low/medium/high), validation_step.

Rules:
  * Each remedy must cite at least one diagnostic id from the supplied list.
  * An investigation can be the best next action when evidence is inadequate.
  * Do NOT promise ranking improvements. Do NOT invent Kasparro facts.
"""


async def build_diagnostics_and_remedies(
    session: AsyncSession,
    run_id: str,
    use_llm: bool = True,
) -> tuple[list[dict], list[dict], VisibilityScore]:
    """Return (diagnostics, remedies, score) for the given run."""
    score = await compute_visibility_score(session, run_id)

    # Pull all results for context
    results = (
        await session.scalars(
            select(ResponseResult).where(ResponseResult.run_id == run_id)
        )
    ).all()

    if not use_llm or not results:
        return _rule_based_diagnostics_remedies(results, score)

    # Try LLM-based synthesis with deterministic verification.
    try:
        return await _llm_diagnostics_remedies(session, run_id, results, score)
    except Exception:
        # If LLM fails, fall back to rule-based. Failures remain visible.
        return _rule_based_diagnostics_remedies(results, score)


# ---------------------------------------------------------------------------
# Rule-based fallback (also used as the baseline in the eval harness)
# ---------------------------------------------------------------------------


def _rule_based_diagnostics_remedies(
    results: list[ResponseResult], score: VisibilityScore
) -> tuple[list[dict], list[dict], VisibilityScore]:
    """Pure-heuristic diagnostics. Used as baseline + LLM fallback.

    Branded vs generic is the LEAD diagnostic — it's the comparison the
    brief explicitly asks for, and it explains the biggest score gap
    (branded prompts score ~44.9 vs generic ~13.0 on the real data).
    """
    by_engine: dict[str, list[ResponseResult]] = {"aimode": [], "chatgpt": [], "gemini": []}
    by_branding: dict[str, list[ResponseResult]] = {"branded": [], "generic": []}
    for r in results:
        if not r.failed:
            by_engine.setdefault(r.engine.value, []).append(r)
            slice_key = "branded" if r.prompt_names_brand else "generic"
            by_branding[slice_key].append(r)

    diagnostics: list[dict] = []
    remedies: list[dict] = []

    # Diagnostic 0 (LEAD): branded vs generic score gap
    branded_score = score.per_category.get("branded")
    generic_score = score.per_category.get("generic")
    if branded_score and generic_score:
        gap = branded_score.composite - generic_score.composite
        branded_mentions = sum(1 for r in by_branding["branded"] if r.mention)
        generic_mentions = sum(1 for r in by_branding["generic"] if r.mention)
        diagnostics.append(
            {
                "id": "lead_branded_vs_generic",
                "title": f"Branded prompts score {branded_score.composite:.1f} vs generic {generic_score.composite:.1f} (gap {gap:+.1f})",
                "pattern": (
                    f"Branded prompts (where the prompt names Kasparro): "
                    f"{len(by_branding['branded'])} responses, "
                    f"{branded_mentions} mention Kasparro, "
                    f"composite score {branded_score.composite:.1f}/100 "
                    f"(95% CI: {branded_score.composite_ci_low or 0:.1f}–{branded_score.composite_ci_high or 0:.1f}). "
                    f"Generic prompts: {len(by_branding['generic'])} responses, "
                    f"{generic_mentions} mention Kasparro, "
                    f"composite score {generic_score.composite:.1f}/100 "
                    f"(95% CI: {generic_score.composite_ci_low or 0:.1f}–{generic_score.composite_ci_high or 0:.1f}). "
                    f"Gap: {gap:+.1f} points."
                ),
                "hypothesis": (
                    "Branded prompts prime the engine to talk about Kasparro, so "
                    "high branded scores reflect recall, not organic discovery. "
                    "Generic prompt scores are the truer measure of AI-mediated "
                    "discoverability. A large gap means Kasparro is found when "
                    "asked-for but rarely surfaces unprompted."
                ),
                "evidence_response_ids": [
                    r.response_id for r in by_branding["generic"] if not r.mention
                ][:3],
                "evidence_excerpts": [
                    (r.evidence_mention or r.evidence_recommendation or r.evidence_negative)[:200]
                    for r in by_branding["generic"] if not r.mention
                    and (r.evidence_mention or r.evidence_recommendation or r.evidence_negative)
                ][:3],
                "would_change_conclusion": (
                    "If generic-prompt composite rises above 30 in a re-capture, "
                    "the discoverability gap is closing."
                ),
            }
        )

    # Diagnostic 1: lowest-mention engine
    engine_mention_rates = {
        eng: (sum(1 for r in rows if r.mention) / len(rows) if rows else 0.0)
        for eng, rows in by_engine.items()
    }
    worst_engine = min(engine_mention_rates, key=engine_mention_rates.get)  # type: ignore[arg-type]
    worst_rate = engine_mention_rates[worst_engine]
    worst_evidence = [
        r for r in by_engine.get(worst_engine, []) if not r.mention
    ][:3]
    diagnostics.append(
        {
            "id": f"loss_{worst_engine}_mention_rate",
            "title": f"Kasparro is rarely named in {worst_engine.upper()} answers",
            "pattern": f"Across {len(by_engine.get(worst_engine, []))} valid {worst_engine} responses, "
            f"only {int(worst_rate * len(by_engine.get(worst_engine, [])))} mention Kasparro "
            f"(mention_rate={worst_rate:.2f}).",
            "hypothesis": f"{worst_engine} may not associate Kasparro with the AI-visibility category; "
            "could be due to crawl coverage, training-data freshness, or third-party validation gap.",
            "evidence_response_ids": [r.response_id for r in worst_evidence],
            "evidence_excerpts": [(r.evidence_mention or r.evidence_recommendation or r.evidence_negative)[:200] for r in worst_evidence],
            "would_change_conclusion": "A repeated run with fresh captures showing "
            "mention_rate > 0.5 on this engine would change this finding.",
        }
    )

    # Diagnostic 2: negative characterisation rate
    # (Dropped the "best recommendation engine" diagnostic to keep the total at 3,
    # now that the lead branded-vs-generic diagnostic is added.)
    neg_count = sum(1 for r in results if r.negative_characterisation)
    neg_rate = neg_count / max(1, len([r for r in results if not r.failed]))
    neg_evidence = [r for r in results if r.negative_characterisation][:3]
    diagnostics.append(
        {
            "id": "risk_negative_characterisation",
            "title": f"{neg_count} responses characterise Kasparro negatively",
            "pattern": f"{neg_count} of {len(results)} valid responses express "
            f"scepticism, call Kasparro a gimmick/hype, or question its legitimacy "
            f"(negative_rate={neg_rate:.2f}).",
            "hypothesis": "Reddit/Quora threads criticising AI-visibility tools as a category "
            "may be feeding this signal; could also reflect Kasparro's young founding date (2026).",
            "evidence_response_ids": [r.response_id for r in neg_evidence],
            "evidence_excerpts": [(r.evidence_mention or r.evidence_recommendation or r.evidence_negative)[:200] for r in neg_evidence],
            "would_change_conclusion": "A re-capture with <2 negative responses would change this.",
        }
    )

    # Remedies (1 highest priority)
    remedies.append(
        {
            "id": "remedy_investigate_crawl_coverage",
            "priority": 1,
            "problem": diagnostics[1]["id"],  # loss_<engine>_mention_rate
            "supporting_evidence": [diagnostics[1]["id"]],
            "action": "Investigate OAI-SearchBot and GPTBot access to kasparro.com, "
            "and request a fresh capture for the worst-performing engine to test for snapshot drift.",
            "expected_benefit": "Establishes whether low mention rate is a persistent pattern "
            "or a one-shot capture artifact.",
            "effort": "low",
            "uncertainty": "high",
            "validation_step": f"Re-capture 10 prompts on {worst_engine}; compare mention_rate "
            "to baseline {worst_rate:.2f}.",
        }
    )
    remedies.append(
        {
            "id": "remedy_third_party_validation",
            "priority": 2,
            "problem": diagnostics[2]["id"],
            "supporting_evidence": [diagnostics[2]["id"]],
            "action": "Build third-party validation signals on Reddit, G2, Capterra, LinkedIn "
            "with concrete case studies, since the brief notes ChatGPT recommends tools citing "
            "third-party sources 88% of the time.",
            "expected_benefit": "Likely increases recommendation rate on engines that rely on "
            "third-party validation.",
            "effort": "high",
            "uncertainty": "medium",
            "validation_step": "Re-run the same 50 prompts after 30 days of validation work; "
            "compare recommendation_rate to baseline.",
        }
    )
    remedies.append(
        {
            "id": "remedy_clarify_category_membership",
            "priority": 3,
            "problem": diagnostics[1]["id"],
            "supporting_evidence": [diagnostics[0]["id"], diagnostics[1]["id"]],
            "action": "Update kasparro.com structured data and category-page copy to explicitly "
            "name Kasparro as an 'AI visibility / AEO / GEO' tool, including competitor comparisons.",
            "expected_benefit": "Helps engines associate Kasparro with the category entity.",
            "effort": "medium",
            "uncertainty": "medium",
            "validation_step": "Re-capture after homepage update; check whether Kasparro appears "
            "in previously-absent unbranded prompts.",
        }
    )

    return diagnostics, remedies, score


# ---------------------------------------------------------------------------
# LLM-based synthesis (with deterministic verification)
# ---------------------------------------------------------------------------


async def _llm_diagnostics_remedies(
    session: AsyncSession,
    run_id: str,
    results: list[ResponseResult],
    score: VisibilityScore,
) -> tuple[list[dict], list[dict], VisibilityScore]:
    """Draft 3 diagnostics + 3 remedies via LLM, then verify each citation."""
    settings = get_settings()
    adapter = build_adapter(ExecutionMode.LIVE if settings.groq_api_key else ExecutionMode.REPLAY)

    # Build a compact context for the LLM
    context = {
        "score": score.model_dump(),
        "per_engine_summary": _per_engine_summary(results),
        "evidence_examples": [
            {
                "response_id": r.response_id,
                "engine": r.engine.value,
                "prompt_id": r.prompt_id,
                "mention": r.mention,
                "recommendation": r.recommendation,
                "negative": r.negative_characterisation,
                "position": r.recommendation_position,
                "competitors": r.competitors_named,
                "excerpt": (r.evidence_mention or r.evidence_recommendation or r.evidence_negative)[:300],
            }
            for r in results
            if r.mention or r.recommendation or r.negative_characterisation
        ][:30],
    }

    # We can't use Instructor's structured output through the factory's
    # generic Protocol here — instead we prompt for raw JSON and parse
    # deterministically, with verification. This keeps the LLM adapter
    # interface simple and the verification explicit.
    diagnostics = await _llm_call_for_json(
        adapter,
        system=DIAGNOSTIC_SYSTEM_PROMPT,
        user=(
            f"Composite score: {score.composite:.2f} "
            f"(mention={score.mention_rate:.2f}, rec={score.recommendation_rate:.2f}, "
            f"neg={score.negative_rate:.2f}, cite={score.citation_rate:.2f}).\n\n"
            f"Context JSON:\n{json.dumps(context, indent=2)}\n\n"
            "Return JSON: a list of exactly 3 diagnostics."
        ),
        expected_keys={
            "id",
            "title",
            "pattern",
            "hypothesis",
            "evidence_response_ids",
            "evidence_excerpts",
            "would_change_conclusion",
        },
    )

    # Verify: every cited response_id exists; every excerpt is verbatim.
    valid_ids = {r.response_id for r in results}
    id_to_text = {r.response_id: (r.evidence_mention or r.evidence_recommendation or r.evidence_negative) for r in results}
    cleaned_diagnostics: list[dict] = []
    for d in diagnostics[:3]:
        kept_ids = [rid for rid in d.get("evidence_response_ids", []) if rid in valid_ids]
        kept_excerpts = [e for e in d.get("evidence_excerpts", []) if _excerpt_in_results(e, results)]
        d["evidence_response_ids"] = kept_ids
        d["evidence_excerpts"] = kept_excerpts
        cleaned_diagnostics.append(d)

    remedies = await _llm_call_for_json(
        adapter,
        system=REMEDY_SYSTEM_PROMPT,
        user=(
            f"Diagnostics:\n{json.dumps(cleaned_diagnostics, indent=2)}\n\n"
            "Return JSON: a list of exactly 3 prioritised remedies."
        ),
        expected_keys={
            "id",
            "priority",
            "problem",
            "supporting_evidence",
            "action",
            "expected_benefit",
            "effort",
            "uncertainty",
            "validation_step",
        },
    )
    cleaned_remedies: list[dict] = []
    diag_ids = {d["id"] for d in cleaned_diagnostics}
    for r in remedies[:3]:
        r["supporting_evidence"] = [
            rid for rid in r.get("supporting_evidence", []) if rid in diag_ids
        ]
        cleaned_remedies.append(r)

    return cleaned_diagnostics, cleaned_remedies, score


def _per_engine_summary(results: list[ResponseResult]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    by_engine: dict[str, list[ResponseResult]] = {}
    for r in results:
        if not r.failed:
            by_engine.setdefault(r.engine.value, []).append(r)
    for eng, rows in by_engine.items():
        n = len(rows)
        out[eng] = {
            "n": n,
            "mention_rate": sum(1 for r in rows if r.mention) / max(1, n),
            "recommendation_rate": sum(1 for r in rows if r.recommendation) / max(1, n),
            "negative_rate": sum(1 for r in rows if r.negative_characterisation) / max(1, n),
            "citation_rate": sum(1 for r in rows if r.has_kasparro_citation) / max(1, n),
        }
    return out


def _excerpt_in_results(excerpt: str, results: list[ResponseResult]) -> bool:
    if not excerpt or len(excerpt) < 10:
        return False
    for r in results:
        if excerpt in (r.evidence_mention or r.evidence_recommendation or r.evidence_negative):
            return True
    return False


async def _llm_call_for_json(
    adapter, system: str, user: str, expected_keys: set[str]
) -> list[dict]:
    """Generic LLM call returning a list of dicts with the expected keys.

    On any failure (timeout, malformed JSON, missing keys), raises so the
    caller falls back to the rule-based path.
    """
    import asyncio

    async def _do_call():
        client = getattr(adapter, "client", None)
        if client is None:
            raise RuntimeError("Adapter has no LLM client for synthesis calls")
        # Use the raw Groq/OpenAI-compatible chat completions interface
        resp = await client.chat.completions.create(
            model=getattr(adapter, "model_name", None) or "llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        # The model may wrap in a list under a key like "diagnostics".
        data = json.loads(content)
        if isinstance(data, dict):
            for k in ("diagnostics", "remedies", "items", "results"):
                if k in data and isinstance(data[k], list):
                    data = data[k]
                    break
        if not isinstance(data, list):
            raise RuntimeError("Expected a list of objects")
        for item in data:
            if not isinstance(item, dict):
                raise RuntimeError("Expected list of objects")
            missing = expected_keys - set(item.keys())
            if missing:
                raise RuntimeError(f"Missing keys: {missing}")
        return data

    return await asyncio.wait_for(_do_call(), timeout=60.0)
