"""Pydantic schemas matching `schemas/input_contract.json` from the pack.

These are the *only* schemas used to validate raw JSONL rows. Unknown
fields are kept (`model_config = ConfigDict(extra="allow")`) so we never
silently drop metadata the pack may add later.
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
    engine: EngineLit
    engine_name: str | None = None
    model_version: str | None = None  # null = unknown, NEVER invent
    answer_text: str = Field(min_length=1)
    answer_markdown: str | None = None
    capture_status: Literal["captured"] = "captured"
    provider_timestamp: str | None = None
    archive_stored_at: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    capture_route: str | None = None
    provider_api_version: str | None = None
    provider_task_id: str | None = None

    @field_validator("engine", mode="before")
    @classmethod
    def _normalize_engine(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.lower()
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
    engine: EngineLit
    order: int = Field(ge=1)
    citation_type: CitationTypeLit
    url: str = Field(min_length=1)
    domain: str | None = None
    title: str | None = None
    source_name: str | None = None
    publication_date: str | None = None  # null = unknown, NEVER invent

    @field_validator("engine", mode="before")
    @classmethod
    def _normalize_engine(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.lower()
        return v
