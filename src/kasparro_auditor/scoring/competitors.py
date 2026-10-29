"""Known competitor hints fed to the LLM adapter.

These are the AI-visibility tools that surface across the 50 prompts
in the pack. The LLM may add others it observes in the text — this is
a hint, not a closed list (per brief: "labels may overlap where
justified").
"""
COMPETITOR_HINTS: list[str] = [
    "profound",
    "otterly ai",
    "peec ai",
    "athenahq",
    "dageno ai",
    "authoritas",
    "aeo checker",
    "shareofmodel",
    "waikay",
    "rankscale",
    "amionai",
    "xfunnel",
    "quno",
    "modelmonitor",
    "airank",
]
