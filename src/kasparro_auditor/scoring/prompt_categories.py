"""Infer prompt categories from the question text.

All 50 prompt_ids share the slug `kasparro:ai_shortlisting:` so the pack
gives zero category information. The brief explicitly says "Infer useful
prompt categories from the actual questions."

Categories inferred (regex + keyword matching):
  - pricing         : "how much", "cost", "overpriced", "worth the money", "retainer"
  - legitimacy      : "legit", "scam", "hype", "gimmick", "safe", "trustworthy"
  - comparison      : "vs", "better than", "compared to", "alternatives"
  - onboarding      : "setup", "onboarding", "cancel", "sign up", "self serve"
  - how_to          : "how do", "how can", "how hard", "how long", "what works"
  - tracking        : "track", "measure", "audit", "monitor", "visibility"
  - features        : "features", "what does", "what kind", "what should"
  - general         : default fallback

Also computes `prompt_names_brand` — whether the prompt itself names Kasparro.
"""
from __future__ import annotations

import re

# Word-boundary regex for "kasparro" so we don't match inside other words
BRAND_PATTERN = re.compile(r"\bkasparro\b", re.IGNORECASE)


def prompt_names_brand(prompt_text: str) -> bool:
    """True if the prompt itself mentions Kasparro (vs generic 'AI visibility tools')."""
    return bool(BRAND_PATTERN.search(prompt_text))


# Category rules in priority order — first match wins.
CATEGORY_RULES: list[tuple[str, list[re.Pattern[str]]]] = [
    ("pricing", [
        re.compile(r"\b(?:how\s+much|cost|overpriced|worth\s+the\s+money|retainer|expensive|priced)\b", re.I),
    ]),
    ("legitimacy", [
        re.compile(r"\b(?:legit|legitimate|scam|hype|gimmick|safe|trustworthy|actually\s+work)\b", re.I),
    ]),
    ("comparison", [
        re.compile(r"\b(?:vs\.?|versus|better\s+than|compared\s+to|alternatives?\s+to|other\s+than)\b", re.I),
    ]),
    ("onboarding", [
        re.compile(r"\b(?:setup|onboarding|onboard|cancel|sign\s+up|self\s+serve|turnkey)\b", re.I),
    ]),
    ("how_to", [
        re.compile(r"\b(?:how\s+do|how\s+can|how\s+hard|how\s+long|how\s+much\s+time|what\s+works)\b", re.I),
    ]),
    ("tracking", [
        re.compile(r"\b(?:track|measure|audit|monitor|visibility|appearance|detect|check)\b", re.I),
    ]),
    ("features", [
        re.compile(r"\b(?:features?|what\s+does|what\s+kind|what\s+should|what\s+usually)\b", re.I),
    ]),
]


def infer_prompt_category(prompt_text: str) -> str:
    """Infer a single category from the prompt text. Falls back to 'general'."""
    for category, patterns in CATEGORY_RULES:
        for p in patterns:
            if p.search(prompt_text):
                return category
    return "general"
