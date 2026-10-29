"""Evaluation runner — one command to rerun the full evaluation.

Usage:
    python -m eval.run_eval --out eval/results/comparison_report.md

Outputs:
    eval/results/comparison_report.md   — human-readable
    eval/results/comparison_report.json — machine-readable
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from eval.baseline_rules import RuleBaseline
from eval.grader import (
    check_diagnostics_evidence,
    format_report,
    grade,
    load_cases,
)
from kasparro_auditor.config import ExecutionMode, get_settings
from kasparro_auditor.llm.factory import build_adapter


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
    real_final = [c for c in final_cases if not c.synthetic]
    synthetic_final = [c for c in final_cases if c.synthetic]

    print(f"Loaded {len(cases)} cases:")
    print(f"  - {len(real_final)} real final-check cases")
    print(f"  - {len(synthetic_final)} synthetic final-check cases")
    print(f"  - {len(tuning_cases)} tuning cases")
    print(
        f"  - Engine spread (final, real): "
        f"{sorted({c.engine for c in real_final})}"
    )
    print(
        f"  - Category spread (final, real): "
        f"{sorted({c.prompt_category for c in real_final})}"
    )

    adapter, mode_value = asyncio.run(_build_classifier())

    # Candidate classifier
    candidate = type("CandidateAdapter", (), {
        "name": f"candidate-{mode_value}",
        "mode": mode_value,
        "model_name": getattr(adapter, "model_name", mode_value),
        "classify": staticmethod(adapter.classify),
    })()

    reports = [
        grade(candidate, final_cases, classifier_name=f"candidate ({mode_value})", mode=mode_value),
        grade(RuleBaseline(), final_cases, classifier_name="rule-baseline-v2", mode="stub"),
    ]

    # Tuning cases (informational only)
    print("\nTuning-case results (informational only):")
    for r in [
        grade(candidate, tuning_cases, classifier_name=f"candidate-tuning ({mode_value})", mode=mode_value),
        grade(RuleBaseline(), tuning_cases, classifier_name="rule-baseline-tuning", mode="stub"),
    ]:
        print(f"  {r.classifier_name}: successful={r.successful}/{r.total_cases}")

    # Diagnostics checker — plant a wrong-evidence case and verify it's caught
    print("\n=== Diagnostics evidence checker ===")
    fake_diag = [{
        "id": "planted_wrong_evidence",
        "title": "Fake diagnostic with bad evidence",
        "evidence_response_ids": ["nonexistent-response-id-12345"],
        "evidence_excerpts": ["This excerpt doesn't exist in any answer."],
    }]
    problems = check_diagnostics_evidence(
        fake_diag,
        valid_response_ids={"real-id-1", "real-id-2"},
        answer_text_lookup={"real-id-1": "Kasparro is mentioned.", "real-id-2": "Other text."},
    )
    if problems:
        print(f"  ✅ Planted wrong-evidence case correctly caught ({len(problems)} problems):")
        for p in problems:
            print(f"     - {p}")
    else:
        print("  ❌ FAILED: planted wrong-evidence case was NOT caught by the checker.")

    md = format_report(reports)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")

    json_payload = {
        "mode": mode_value,
        "reports": [
            {
                "classifier_name": r.classifier_name,
                "mode": r.mode,
                "total_cases": r.total_cases,
                "successful": r.successful,
                "failed": r.failed,
                "skipped": r.skipped,
                "unsupported_claim_failures": r.unsupported_claim_failures,
                "label_metrics": {
                    label: {
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                        "tp": m.tp, "fp": m.fp, "fn": m.fn,
                    }
                    for label, m in r.label_metrics.items()
                },
                "facts_metrics": {
                    label: {
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                        "tp": m.tp, "fp": m.fp, "fn": m.fn,
                    }
                    for label, m in r.facts_metrics.items()
                },
                "competitor_metrics": {
                    "precision": r.competitor_metrics.precision,
                    "recall": r.competitor_metrics.recall,
                    "f1": r.competitor_metrics.f1,
                    "intersection": r.competitor_metrics.intersection,
                    "predicted_only": r.competitor_metrics.predicted_only,
                    "expected_only": r.competitor_metrics.expected_only,
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
