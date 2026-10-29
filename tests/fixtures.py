"""Inline JSONL fixtures for ingest tests — no dependency on the real data pack.

Each fixture is a small JSONL string covering a specific scenario.
Tests use these by writing them to a temp dir and calling import_pack().
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# A run.json that matches the fixtures below
RUN_JSON = {
    "run_id": "test-run-001",
    "brand": "Kasparro",
    "original_run_counts": {"prompts": 2, "responses": 3, "citations": 4},
}

# 2 valid prompts
PROMPTS_JSONL = [
    {"prompt_id": "p001", "run_id": "test-run-001", "text": "does kasparro work?"},
    {"prompt_id": "p002", "run_id": "test-run-001", "text": "is kasparro better than profound?"},
]

# 3 valid responses (p001 × 3 engines)
RESPONSES_JSONL = [
    {
        "response_id": "r001",
        "run_id": "test-run-001",
        "prompt_id": "p001",
        "engine": "aimode",
        "engine_name": "Google AI Mode",
        "model_version": None,
        "answer_text": "Kasparro is mentioned as an AI visibility tool.",
        "answer_markdown": None,
        "capture_status": "captured",
        "provider_timestamp": "2026-07-31T04:04:44+00:00",
        "archive_stored_at": "2026-07-31T04:06:40+00:00",
        "settings": {"country": "India"},
    },
    {
        "response_id": "r002",
        "run_id": "test-run-001",
        "prompt_id": "p001",
        "engine": "chatgpt",
        "answer_text": "Kasparro is a great option for tracking AI visibility.",
        "capture_status": "captured",
        "archive_stored_at": "2026-07-31T04:06:40+00:00",
    },
    {
        "response_id": "r003",
        "run_id": "test-run-001",
        "prompt_id": "p002",
        "engine": "gemini",
        "answer_text": "Kasparro is not widely recognized. Profound is better known.",
        "capture_status": "captured",
        "archive_stored_at": "2026-07-31T04:06:40+00:00",
    },
]

# 4 valid citations
CITATIONS_JSONL = [
    {
        "citation_id": "c001",
        "response_id": "r001",
        "prompt_id": "p001",
        "engine": "aimode",
        "order": 1,
        "citation_type": "sources_panel",
        "url": "https://www.linkedin.com/posts/apoorvshrm_example",
    },
    {
        "citation_id": "c002",
        "response_id": "r001",
        "prompt_id": "p001",
        "engine": "aimode",
        "order": 2,
        "citation_type": "sources_panel",
        "url": "https://kasparro.com/?utm_source=reddit",
    },
    {
        "citation_id": "c003",
        "response_id": "r002",
        "prompt_id": "p001",
        "engine": "chatgpt",
        "order": 1,
        "citation_type": "inline_hyperlink",
        "url": "https://example.com/page#section",
    },
    {
        "citation_id": "c004",
        "response_id": "r003",
        "prompt_id": "p002",
        "engine": "gemini",
        "order": 1,
        "citation_type": "sources_panel",
        "url": "https://www.athenahq.ai/",
    },
]

# A prompt with whitespace-only text (should fail validation)
BAD_PROMPT_WHITESPACE = [
    {"prompt_id": "p-bad", "run_id": "test-run-001", "text": "   "},
]

# A response with missing archive_stored_at (should fail validation)
BAD_RESPONSE_MISSING_FIELD = [
    {
        "response_id": "r-bad",
        "run_id": "test-run-001",
        "prompt_id": "p001",
        "engine": "aimode",
        "answer_text": "Some text.",
        "capture_status": "captured",
        # archive_stored_at is MISSING — required per contract
    },
]

# A response with uppercase engine "ChatGPT" (should fail — exact match only)
BAD_RESPONSE_UPPERCASE_ENGINE = [
    {
        "response_id": "r-bad2",
        "run_id": "test-run-001",
        "prompt_id": "p001",
        "engine": "ChatGPT",  # should be lowercase "chatgpt"
        "answer_text": "Some text.",
        "capture_status": "captured",
        "archive_stored_at": "2026-07-31T04:06:40+00:00",
    },
]

# A citation referencing a non-existent response_id (orphan)
ORPHAN_CITATION = [
    {
        "citation_id": "c-orphan",
        "response_id": "r-does-not-exist",
        "prompt_id": "p001",
        "engine": "aimode",
        "order": 1,
        "citation_type": "sources_panel",
        "url": "https://example.com/orphan",
    },
]

# A citation whose engine doesn't match the response's engine
ENGINE_MISMATCH_CITATION = [
    {
        "citation_id": "c-mismatch",
        "response_id": "r001",  # r001 has engine="aimode"
        "prompt_id": "p001",
        "engine": "chatgpt",  # WRONG — should be "aimode"
        "order": 1,
        "citation_type": "sources_panel",
        "url": "https://example.com/mismatch",
    },
]

# A citation whose prompt_id doesn't match the response's prompt_id
PROMPT_MISMATCH_CITATION = [
    {
        "citation_id": "c-prompt-mismatch",
        "response_id": "r001",  # r001 has prompt_id="p001"
        "prompt_id": "p002",  # WRONG — should be "p001"
        "engine": "aimode",
        "order": 1,
        "citation_type": "sources_panel",
        "url": "https://example.com/prompt-mismatch",
    },
]

# A malformed JSON line (not valid JSON at all)
MALFORMED_JSON_LINE = '{"prompt_id": "p-broken", "run_id": "test-run", "text": "missing closing brace"'

# A response with a different answer_text than the original (for conflict testing)
RESPONSE_DIFFERENT_CONTENT = [
    {
        "response_id": "r001",  # same ID as the original
        "run_id": "test-run-001",
        "prompt_id": "p001",
        "engine": "aimode",
        "answer_text": "COMPLETELY DIFFERENT TEXT — this is a conflict.",
        "capture_status": "captured",
        "archive_stored_at": "2026-07-31T04:06:40+00:00",
    },
]


def write_pack(
    base_dir: Path,
    prompts: list[dict] | None = None,
    responses: list[dict] | None = None,
    citations: list[dict] | None = None,
    run_json: dict | None = None,
) -> Path:
    """Write a fake data pack to base_dir/data/ and return the base_dir.

    Any of prompts/responses/citations can be a raw string (for malformed
    JSON tests) or a list of dicts (which gets JSONL-serialized).
    """
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # run.json
    (data_dir / "run.json").write_text(
        json.dumps(run_json or RUN_JSON), encoding="utf-8"
    )

    # prompts.jsonl
    if prompts is not None:
        _write_jsonl_or_raw(data_dir / "prompts.jsonl", prompts)

    # responses.jsonl
    if responses is not None:
        _write_jsonl_or_raw(data_dir / "responses.jsonl", responses)

    # citations.jsonl
    if citations is not None:
        _write_jsonl_or_raw(data_dir / "citations.jsonl", citations)

    return base_dir


def _write_jsonl_or_raw(path: Path, data: list[dict[str, Any]] | list[str]) -> None:
    """Write a list of dicts as JSONL, or a list of strings as raw lines."""
    lines = []
    for item in data:
        if isinstance(item, str):
            lines.append(item)  # raw line, might be malformed
        else:
            lines.append(json.dumps(item, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
