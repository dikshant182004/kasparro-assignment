"""End-to-end pipeline runner. Defaults to replay mode (no API key needed).

Usage:
    python -m scripts.run_analysis                       # replay mode
    EXECUTION_MODE=live python -m scripts.run_analysis   # live Groq calls
    python -m scripts.run_analysis --out outputs/run.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from kasparro_auditor.config import ExecutionMode, get_settings
from kasparro_auditor.pipeline import run_full_pipeline


async def main(
    source_dir: str | None,
    execution_mode: ExecutionMode | None,
    out_path: str,
    auto_submit: bool,
) -> int:
    summary = await run_full_pipeline(
        source_dir=source_dir,
        execution_mode=execution_mode,
        auto_submit=auto_submit,
        out_path=out_path,
    )
    print(f"\n=== Analysis summary ===")
    print(f"Dataset:        {summary['dataset_id']}")
    print(f"Analysis run:   {summary['analysis_run_id']}")
    print(f"Mode:           {summary['execution_mode']}")
    print(f"Model:          {summary['model_name']}")
    counts = summary["counts"]
    print(
        f"Counts:         prompts={counts['prompts']} "
        f"responses(new={counts['responses_new']}, reused={counts['responses_reused']}) "
        f"citations={counts['citations']} rejected={counts['rejected']}"
    )
    print(
        f"Results:        successful={counts['successful']} "
        f"failed={counts['failed']} skipped={counts['skipped']}"
    )
    score = summary["score"]
    print(
        f"Score:          composite={score['composite']:.2f}/100 "
        f"(95% CI: {score['composite_ci_low']:.2f}-{score['composite_ci_high']:.2f})"
    )
    print(f"  mention_rate   = {score['mention_rate']:.3f}")
    print(f"  rec_rate       = {score['recommendation_rate']:.3f}")
    print(f"  neg_rate       = {score['negative_rate']:.3f}")
    print(f"  citation_rate  = {score['citation_rate']:.3f}")
    print(f"\nDiagnostics ({len(summary['diagnostics'])}):")
    for d in summary["diagnostics"]:
        print(f"  - [{d['id']}] {d['title']}")
    print(f"\nRemedies ({len(summary['remedies'])}):")
    for r in summary["remedies"]:
        print(f"  #{r['priority']} [{r['id']}] {r['action'][:80]}")
    print(f"\nFull output written to: {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Kasparro analysis pipeline")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Path to the assignment pack. Defaults to DATA_PACK_DIR.",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "replay", "stub"],
        default=None,
        help="Override EXECUTION_MODE for this run.",
    )
    parser.add_argument(
        "--out",
        default="outputs/analysis_output.json",
        help="Output path for the analysis summary JSON.",
    )
    parser.add_argument(
        "--auto-submit",
        action="store_true",
        help="Create and submit a draft report automatically after analysis.",
    )
    args = parser.parse_args()
    mode = ExecutionMode(args.mode) if args.mode else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    raise SystemExit(
        asyncio.run(main(args.source_dir, mode, args.out, args.auto_submit))
    )
