"""Pydantic schemas matching `schemas/input_contract.json` from the pack.

Strict validation: required fields have no defaults, engine is exact-match
(no case-folding), int fields use strict=True, and whitespace-only
answer_text is rejected. Extra fields are allowed (so we never drop
metadata the pack may add later) AND persisted via a raw_json column
in the ORM layer.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EngineLit = Literal["aimode", "chatgpt", "gemini"]
CitationTypeLit = Literal["inline_hyperlink", "sources_panel"]


class PromptRecord(BaseModel):
    """One row from data/prompts.jsonl."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=False)

    prompt_id: str
    run_id: str
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _reject_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be whitespace-only")
        return v


class ResponseRecord(BaseModel):
    """One row from data/responses.jsonl.

    `answer_text` and `answer_markdown` are two views of one response —
    store both, classify on `answer_text` (it includes the appended
    citations section per READ_ME.txt).
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=False)

    response_id: str
    run_id: str
    prompt_id: str
    engine: EngineLit          # exact-match, no case-folding
    engine_name: str | None = None
    model_version: str | None = None  # null = unknown, NEVER invent
    answer_text: str = Field(min_length=1)
    answer_markdown: str | None = None
    capture_status: Literal["captured"]   # required, no default
    provider_timestamp: str | None = None
    archive_stored_at: str                # required per input_contract.json
    settings: dict[str, Any] = Field(default_factory=dict)
    capture_route: str | None = None
    provider_api_version: str | None = None
    provider_task_id: str | None = None

    @field_validator("answer_text")
    @classmethod
    def _reject_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("answer_text must not be whitespace-only")
        return v


class CitationRecord(BaseModel):
    """One row from data/citations.jsonl.

    A URL can appear inline and in the sources panel of the same answer,
    or recur across answers. The dedup rule is implemented in
    `kasparro_auditor.ingest.normalize` and disclosed in the README.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=False)

    citation_id: str
    response_id: str
    prompt_id: str
    engine: EngineLit          # exact-match, no case-folding
    order: int = Field(ge=1, strict=True)
    citation_type: CitationTypeLit
    url: str = Field(min_length=1, strict=True)
    domain: str | None = None
    title: str | None = None
    source_name: str | None = None
    publication_date: str | None = None  # null = unknown, NEVER invent
