"""Deterministic Facts computed in code, never sent for LLM judgment.

Per the reviewer's P0 fix: mention, prompt_names_brand, competitors_known,
and citation counts are all computed deterministically. The LLM only
returns Judgment fields (recommendation, negative, position, evidence).

The LLM's mention cannot override the regex. If the LLM disagrees with
the regex mention, the regex wins and a disagreement flag is set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from kasparro_auditor.scoring.competitors import COMPETITOR_ALIASES

# Word-boundary regex for "kasparro" — case-insensitive
BRAND_PATTERN = re.compile(r"\bkasparro\b", re.IGNORECASE)


@dataclass(slots=True)
class Facts:
    """Deterministic facts about one response. Computed in code."""

    mention: bool = False
    mention_spans: list[tuple[int, int]] = field(default_factory=list)
    prompt_names_brand: bool = False
    prompt_category: str = "general"
    competitors_known: list[str] = field(default_factory=list)
    citation_count_for_response: int = 0
    unique_citation_count: int = 0
    has_kasparro_citation: bool = False


def compute_mention(answer_text: str) -> tuple[bool, list[tuple[int, int]]]:
    """Return (mention, spans) where spans is a list of (start, end) char offsets.

    Uses word-boundary regex so "Kasparro" is detected but not "Kasparros".
    """
    spans: list[tuple[int, int]] = []
    for m in BRAND_PATTERN.finditer(answer_text):
        spans.append((m.start(), m.end()))
    return (len(spans) > 0, spans)


def compute_competitors(answer_text: str) -> list[str]:
    """Return canonical competitor names mentioned verbatim in the text.

    Uses COMPETITOR_ALIASES (single source of truth) so we don't have two
    disagreeing lists. Each canonical name is returned at most once.
    """
    lower = answer_text.lower()
    found: list[str] = []
    seen: set[str] = set()
    for canonical, aliases in COMPETITOR_ALIASES.items():
        for alias in aliases:
            if alias in lower and canonical not in seen:
                found.append(canonical)
                seen.add(canonical)
                break
    return found


def strip_trailing_source_block(answer_text: str) -> str:
    """Strip the trailing '[n] url' source block before sending to the LLM.

    Answers can be up to ~10k chars. The trailing citation list (e.g.
    '[1] https://example.com/...\\n[2] https://example.com/...') is not
    part of the answer's stance and just wastes tokens. We strip it but
    keep the prose intact (so evidence excerpts stay verbatim).
    """
    # Look for a line of the form '[1] https://...' or 'Citations:' marker
    lines = answer_text.split("\n")
    cutoff = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match '[1] https://...' or '[1] http://...'
        if re.match(r"^\[\d+\]\s*https?://", stripped):
            cutoff = i
            break
        # Match 'Citations:' header
        if stripped.lower() in ("citations:", "sources:", "references:"):
            cutoff = i
            break
    return "\n".join(lines[:cutoff]).rstrip()
