"""Evaluation runner — one command to rerun the full evaluation.

Usage:
    python -m eval.run_eval --out eval/results/comparison_report.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from eval.baseline_rules import RuleBaseline
from eval.grader import format_report, grade, load_cases
from kasparro_auditor.llm.factory import build_adapter
from kasparro_auditor.config import ExecutionMode, get_settings


async def _build_classifier():
    settings = get_settings()
    mode = settings.execution_mode
    return build_adapter(mode), mode.value


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation harness")
    parser.add_argument(
        "--out",
        default="eval/results/comparison_report.md",
        help="Output path for the comparison report",
    )
    parser.add_argument(
        "--json-out",
        default="eval/results/comparison_report.json",
        help="Output path for the machine-readable JSON report",
    )
    args = parser.parse_args()

    cases = load_cases()
    final_cases = [c for c in cases if not c.is_tuning]
    tuning_cases = [c for c in cases if c.is_tuning]

    print(f"Loaded {len(cases)} cases ({len(final_cases)} final, {len(tuning_cases)} tuning)")

    # Build the candidate classifier
    adapter, mode_value = asyncio.run(_build_classifier())

    candidate = type("CandidateAdapter", (), {
        "name": f"candidate-{mode_value}",
        "model_name": getattr(adapter, "model_name", mode_value),
        "classify": staticmethod(adapter.classify),
    })()

    reports = [
        grade(candidate, final_cases, classifier_name=f"candidate ({mode_value})"),
        grade(RuleBaseline(), final_cases, classifier_name="rule-baseline"),
    ]

    # Also grade on tuning cases (informational, not final-check)
    print("\nTuning-case results (informational only):")
    for r in [
        grade(candidate, tuning_cases, classifier_name=f"candidate-tuning ({mode_value})"),
        grade(RuleBaseline(), tuning_cases, classifier_name="rule-baseline-tuning"),
    ]:
        print(f"  {r.classifier_name}: successful={r.successful}/{r.total_cases}")

    md = format_report(reports)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")

    # JSON output for downstream tooling
    json_payload = {
        "candidate_mode": mode_value,
        "reports": [
            {
                "classifier_name": r.classifier_name,
                "total_cases": r.total_cases,
                "successful": r.successful,
                "failed": r.failed,
                "unsupported_claim_failures": r.unsupported_claim_failures,
                "label_metrics": {
                    label: {
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                        "tp": m.tp,
                        "fp": m.fp,
                        "fn": m.fn,
                    }
                    for label, m in r.label_metrics.items()
                },
            }
            for r in reports
        ],
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print(f"\nComparison report written to: {args.out}")
    print(f"JSON report written to: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
