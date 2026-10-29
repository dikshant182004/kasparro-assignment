"""Scoring package: Facts + Judgment + composite score + counts."""
from kasparro_auditor.scoring.classify import classify_one, run_analysis
from kasparro_auditor.scoring.counts import compute_citation_facts
from kasparro_auditor.scoring.score import (
    SCORE_WEIGHTS_V2,
    SCORING_VERSION,
    CATEGORY_VERSION,
    compute_visibility_score,
)
from kasparro_auditor.scoring.competitors import COMPETITOR_ALIASES
from kasparro_auditor.scoring.facts import (
    Facts,
    compute_mention,
    compute_competitors,
    strip_trailing_source_block,
)
from kasparro_auditor.scoring.prompt_categories import (
    infer_prompt_category,
    prompt_names_brand,
)

__all__ = [
    "classify_one",
    "run_analysis",
    "compute_citation_facts",
    "compute_visibility_score",
    "SCORE_WEIGHTS_V2",
    "SCORING_VERSION",
    "CATEGORY_VERSION",
    "COMPETITOR_ALIASES",
    "Facts",
    "compute_mention",
    "compute_competitors",
    "strip_trailing_source_block",
    "infer_prompt_category",
    "prompt_names_brand",
]
