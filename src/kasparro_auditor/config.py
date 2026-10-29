"""Application configuration loaded from environment.

Two local identities (analyst + reviewer) are derived from environment
variables. A caller-supplied role or actor name is NOT authorization —
identity always comes from a server-side token check.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExecutionMode(str, Enum):
    """How the LLM adapter resolves a classification request."""

    LIVE = "live"      # real Groq call
    REPLAY = "replay"  # cached JSON for known response_id
    STUB = "stub"      # deterministic placeholder (tests)


class ActorRole(str, Enum):
    ANALYST = "analyst"
    REVIEWER = "reviewer"


class Settings(BaseSettings):
    """Strongly-typed, validated configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Runtime mode -------------------------------------------------------
    execution_mode: ExecutionMode = ExecutionMode.LIVE

    # ---- Server ------------------------------------------------------------
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"

    # ---- Database ----------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./kasparro.db"

    # ---- LLM (Groq) --------------------------------------------------------
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"
    groq_timeout_seconds: float = 30.0
    groq_max_retries: int = 2
    groq_max_output_tokens: int = 1024
    groq_temperature: float = 0.0

    # ---- Auth (server-side tokens; never trust request body) ---------------
    analyst_token: str = Field(default="analyst-local-token-change-me")
    reviewer_token: str = Field(default="reviewer-local-token-change-me")
    analyst_actor_id: str = "analyst-001"
    reviewer_actor_id: str = "reviewer-001"

    # ---- Pipeline guards ---------------------------------------------------
    max_concurrent_llm_calls: int = 5
    analysis_timeout_seconds: int = 900

    # ---- Paths -------------------------------------------------------------
    data_pack_dir: Path = Path("./data_pack")
    replay_cache_dir: Path = Path("./data_pack/replay")

    @field_validator("execution_mode", mode="before")
    @classmethod
    def _normalize_mode(cls, v: str | ExecutionMode) -> ExecutionMode:
        if isinstance(v, ExecutionMode):
            return v
        return ExecutionMode(str(v).lower())

    # ---- Convenience -------------------------------------------------------
    def actor_for_token(self, token: str) -> tuple[str, ActorRole] | None:
        """Resolve a bearer token to (actor_id, role) or None if unknown."""
        if token == self.analyst_token:
            return (self.analyst_actor_id, ActorRole.ANALYST)
        if token == self.reviewer_token:
            return (self.reviewer_actor_id, ActorRole.REVIEWER)
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Tests override via dependency injection."""
    return Settings()
