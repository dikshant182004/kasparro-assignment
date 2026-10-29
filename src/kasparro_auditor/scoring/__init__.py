"""Scoring package: classification orchestration, counts, visibility score."""
from kasparro_auditor.scoring.classify import classify_one, run_analysis
from kasparro_auditor.scoring.counts import compute_citation_facts
from kasparro_auditor.scoring.score import compute_visibility_score, SCORE_WEIGHTS
from kasparro_auditor.scoring.competitors import COMPETITOR_HINTS

__all__ = [
    "classify_one",
    "run_analysis",
    "compute_citation_facts",
    "compute_visibility_score",
    "SCORE_WEIGHTS",
    "COMPETITOR_HINTS",
]
