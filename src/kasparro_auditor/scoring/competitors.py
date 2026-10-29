"""Single source of truth for competitor names + aliases.

Previously we had two lists (KNOWN_COMPETITORS in stub_adapter and
COMPETITOR_HINTS in scoring/competitors) that disagreed — 8 hint names
appeared 0 times in the real pack. Now we have ONE list of canonical
names with their alias patterns. The Facts computer uses this; the LLM
is NOT given hints (that biases it toward echoing them).
"""

# Canonical competitor name → list of lowercase substrings that indicate it.
# Canonical names are what we store in the DB. Aliases are matched verbatim
# against the lowercased answer_text.
COMPETITOR_ALIASES: dict[str, list[str]] = {
    "profound": ["profound"],
    "otterly ai": ["otterly ai", "otterly"],
    "peec ai": ["peec ai", "peec"],
    "athenahq": ["athenahq", "athena hq", "athenahq.ai"],
    "dageno ai": ["dageno ai", "dageno"],
    "authoritas": ["authoritas"],
    "aeo checker": ["aeo checker"],
    "shareofmodel": ["shareofmodel"],
    "waikay": ["waikay"],
    "rankscale": ["rankscale"],
    "amionai": ["amionai", "amion.ai"],
    "xfunnel": ["xfunnel"],
    "quno": ["quno"],
    "modelmonitor": ["modelmonitor"],
    "airank": ["airank", "ai rank"],
}

# Sorted by length descending so longer aliases match first when canonical
# names overlap (e.g. "otterly ai" before "otterly").
_ALL_ALIASES: list[tuple[str, str]] = sorted(
    [(alias, canon) for canon, aliases in COMPETITOR_ALIASES.items() for alias in aliases],
    key=lambda x: -len(x[0]),
)
