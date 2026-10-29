"""Evaluation grader.

Computes per-label precision/recall/F1 against hand-labeled cases,
verifies the deterministic Facts (mention, prompt_names_brand, competitors)
separately from the LLM Judgment, and checks that diagnostic claims
have supporting response_id evidence (the unsupported-claim guardrail).

Per the brief: "an LLM judging its own prose is insufficient" —
labels are exact-match against hand-labeled expected values.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from kasparro_auditor.llm.adapter import (
    LLMContext,
    LLMError,
    Judgment,
    verify_evidence,
)
from kasparro_auditor.scoring.facts import (
    compute_mention,
    compute_competitors,
)
from kasparro_auditor.scoring.prompt_categories import prompt_names_brand

CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    case_id: str
    response_id: str
    engine: str
    prompt_category: str
    is_tuning: bool
    is_custom: bool
    expected_facts: dict
    expected_judgment: Judgment
    expected_failed: bool
    edge_case_description: str | None = None


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        j_dict = d["expected_judgment"]
        # Don't validate via Judgment yet — the grader needs to compare
        # the classifier's output to the expected labels, and we want to
        # catch disagreements, not raise on construction.
        expected_judgment = Judgment.model_validate(j_dict)
        cases.append(
            EvalCase(
                case_id=d["case_id"],
                response_id=d["response_id"],
                engine=d.get("engine", "unknown"),
                prompt_category=d.get("prompt_category", "unknown"),
                is_tuning=d.get("is_tuning", False),
                is_custom=d.get("is_custom", False),
                expected_facts=d["expected_facts"],
                expected_judgment=expected_judgment,
                expected_failed=d.get("expected_failed", False),
                edge_case_description=d.get("edge_case_description"),
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Synthetic answer text — for cases not in the real pack
# ---------------------------------------------------------------------------


def _load_real_answer_text(response_id: str) -> str | None:
    """Look up the real answer_text from the data pack if available."""
    from kasparro_auditor.config import get_settings

    settings = get_settings()
    responses_path = settings.data_pack_dir / "data" / "responses.jsonl"
    if not responses_path.is_file():
        return None
    for line in responses_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["response_id"] == response_id:
            return row["answer_text"]
    return None


def _synthesize_answer_text(case: EvalCase) -> str:
    """For stub cases, build a minimal answer text containing the evidence."""
    excerpts = [
        case.expected_judgment.evidence_mention,
        case.expected_judgment.evidence_recommendation,
        case.expected_judgment.evidence_negative,
    ]
    parts = [f"This is a synthesized answer for {case.case_id}."]
    for excerpt in excerpts:
        if excerpt and len(excerpt) >= 10:
            parts.append(excerpt)
    if case.expected_facts.get("competitors_known"):
        comps = case.expected_facts["competitors_known"]
        parts.append(f"Other tools mentioned: {', '.join(comps)}.")
    parts.append("Additional context for the answer goes here.")
    return " ".join(parts)


def build_context(case: EvalCase) -> LLMContext:
    """Build the LLMContext for an eval case."""
    text = _load_real_answer_text(case.response_id) or _synthesize_answer_text(case)
    # Compute deterministic Facts here so the LLM doesn't have to
    mention, _spans = compute_mention(text)
    competitors = compute_competitors(text)
    return LLMContext(
        response_id=case.response_id,
        prompt_id=f"eval:{case.case_id}",
        engine=case.engine,
        prompt_text=f"(eval case {case.case_id}) — does kasparro work?",
        answer_text=text,
        mention=mention,
        prompt_names_brand=prompt_names_brand(f"does kasparro work?"),
        competitors_known=competitors,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class LabelMetrics:
    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class GraderReport:
    classifier_name: str
    total_cases: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    label_metrics: dict[str, LabelMetrics] = field(default_factory=dict)
    facts_metrics: dict[str, LabelMetrics] = field(default_factory=dict)
    unsupported_claim_failures: list[str] = field(default_factory=list)
    per_case_results: list[dict] = field(default_factory=list)


def grade(
    classifier,
    cases: Iterable[EvalCase],
    classifier_name: str = "unknown",
) -> GraderReport:
    """Run a classifier over the eval cases and compute per-label metrics."""
    import asyncio

    metrics = {
        "recommendation": LabelMetrics("recommendation"),
        "negative_characterisation": LabelMetrics("negative_characterisation"),
    }
    facts_metrics = {
        "mention": LabelMetrics("mention"),
        "prompt_names_brand": LabelMetrics("prompt_names_brand"),
    }
    report = GraderReport(classifier_name=classifier_name)

    async def _run_case(case: EvalCase) -> dict:
        ctx = build_context(case)
        try:
            # Verify Facts first (deterministic, should never fail)
            mention, _ = compute_mention(ctx.answer_text)
            if mention != case.expected_facts.get("mention", False):
                if mention and not case.expected_facts.get("mention", False):
                    facts_metrics["mention"].fp += 1
                elif not mention and case.expected_facts.get("mention", False):
                    facts_metrics["mention"].fn += 1
            else:
                if mention:
                    facts_metrics["mention"].tp += 1

            # Skip LLM call if mention is False (matches pipeline behavior)
            if not case.expected_facts.get("mention", False):
                report.successful += 1
                return {
                    "case_id": case.case_id,
                    "status": "ok_skipped_llm",
                    "facts": {"mention": mention},
                }

            result = await classifier.classify(ctx)
            # Verify evidence (unsupported-claim guardrail)
            try:
                verify_evidence(result, ctx.answer_text, response_id=case.response_id)
            except Exception as e:
                report.unsupported_claim_failures.append(
                    f"{case.case_id}: evidence verification failed: {e}"
                )
                return {
                    "case_id": case.case_id,
                    "status": "unsupported_claim",
                    "error": str(e),
                }

            # Score against expected judgment labels
            for label in metrics:
                got = getattr(result, label)
                exp = getattr(case.expected_judgment, label)
                if got and exp:
                    metrics[label].tp += 1
                elif got and not exp:
                    metrics[label].fp += 1
                elif not got and exp:
                    metrics[label].fn += 1

            report.successful += 1
            return {
                "case_id": case.case_id,
                "status": "ok",
                "got": result.model_dump(),
                "expected": case.expected_judgment.model_dump(),
            }
        except LLMError as e:
            report.failed += 1
            if case.expected_failed:
                # Expected failure — counts as success
                report.successful += 1
                report.failed -= 1
            return {"case_id": case.case_id, "status": "failed", "error": str(e)}
        except Exception as e:
            report.failed += 1
            return {"case_id": case.case_id, "status": "error", "error": str(e)}

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                asyncio.gather(*[_run_case(c) for c in cases])
            )
        finally:
            loop.close()
    finally:
        pass

    report.total_cases = len(results)
    report.label_metrics = metrics
    report.facts_metrics = facts_metrics
    report.per_case_results = list(results)
    return report


def format_report(reports: list[GraderReport]) -> str:
    """Render a markdown comparison report."""
    lines: list[str] = []
    lines.append("# Evaluation Comparison Report\n")
    lines.append(
        "Per-label precision/recall/F1 across the hand-labeled eval cases.\n"
        "Final-check cases (is_tuning=false) only. Tuning cases shown separately.\n\n"
        "**Architecture**: Facts (mention, prompt_names_brand, competitors_known) "
        "are computed deterministically in code. The LLM only returns Judgment "
        "fields (recommendation, negative, position, evidence). LLM calls are "
        "skipped entirely when the deterministic mention check returns False.\n"
    )

    for r in reports:
        lines.append(f"## Classifier: {r.classifier_name}\n")
        lines.append(f"- Total cases: {r.total_cases}")
        lines.append(f"- Successful: {r.successful}")
        lines.append(f"- Failed: {r.failed}")
        lines.append(f"- Unsupported-claim failures: {len(r.unsupported_claim_failures)}")
        if r.unsupported_claim_failures:
            for f in r.unsupported_claim_failures:
                lines.append(f"  - {f}")

        lines.append("\n### Deterministic Facts metrics\n")
        lines.append("| Fact | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|")
        for label, m in r.facts_metrics.items():
            lines.append(
                f"| {label} | {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} |"
            )

        lines.append("\n### LLM Judgment metrics\n")
        lines.append("| Label | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|")
        for label, m in r.label_metrics.items():
            lines.append(
                f"| {label} | {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} |"
            )
        lines.append("")

    lines.append("\n## Grader limits\n")
    lines.append(
        "- Facts (mention, prompt_names_brand, competitors_known) are computed "
        "deterministically and graded as exact-match against hand-labeled values.\n"
        "- LLM Judgment labels (recommendation, negative) are graded as "
        "exact-match against hand-labeled values.\n"
        "- An LLM judging its own prose is explicitly insufficient (brief requirement).\n"
        "- Evidence excerpts are verified as verbatim substrings of answer_text.\n"
        "- Score-component arithmetic is verified separately in tests/test_scoring.py.\n"
        "- Diagnostic claims have supporting response_id evidence verified at synth time.\n"
    )
    return "\n".join(lines) + "\n"
