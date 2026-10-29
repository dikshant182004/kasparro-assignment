"""Application configuration loaded from environment.

Two local identities (analyst + reviewer) are derived from environment
variables. A caller-supplied role or actor name is NOT authorization —
identity always comes from a server-side token check.

Security:
  - Default tokens are refused at startup (Settings.validate_startup).
  - Token comparison uses hmac.compare_digest (constant-time).
"""
from __future__ import annotations

import hmac
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Default tokens that must be changed before startup.
_DEFAULT_ANALYST_TOKEN = "analyst-local-token-change-me"
_DEFAULT_REVIEWER_TOKEN = "reviewer-local-token-change-me"


class ExecutionMode(str, Enum):
    """How the LLM adapter resolves a classification request."""

    LIVE = "live"      # Groq call (requires hiring-team approval for hosted access)
    REPLAY = "replay"  # cached JSON for known response_id
    STUB = "stub"      # deterministic placeholder (tests)
    LOCAL = "local"    # Ollama or other OpenAI-compatible local endpoint


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
    # NOTE: llama-3.3-70b-versatile and llama-3.1-8b-instant were retired on
    # 16 August 2026 (free + developer tiers). Use openai/gpt-oss-120b as
    # primary (production GA, $0.15/M in, $0.60/M out) with openai/gpt-oss-20b
    # as fallback. Free-tier rate limit: 30 RPM / 1,000 RPD / 8K TPM.
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "openai/gpt-oss-20b"
    groq_timeout_seconds: float = 30.0
    groq_max_retries: int = 2
    groq_max_output_tokens: int = 1024
    groq_temperature: float = 0.0

    # ---- LLM (Ollama, approved local model) --------------------------------
    # Used when EXECUTION_MODE=local. The brief requires "a local model or
    # hosted access explicitly provided or approved by the hiring team".
    # A personal Groq key is NOT that by default. Ollama is OpenAI-compatible.
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1:8b"
    ollama_api_key: str = "ollama"  # placeholder required by SDK

    # ---- Auth (server-side tokens; never trust request body) ---------------
    analyst_token: str = Field(default="analyst-local-token-change-me")
    reviewer_token: str = Field(default="reviewer-local-token-change-me")
    analyst_actor_id: str = "analyst-001"
    reviewer_actor_id: str = "reviewer-001"

    # ---- Pipeline guards ---------------------------------------------------
    # Free-tier Groq limit is 30 RPM. Use concurrency=2 to leave margin.
    max_concurrent_llm_calls: int = 2
    analysis_timeout_seconds: int = 900
    # Hard cap on total LLM calls per run (prevents runaway cost).
    max_llm_calls_per_run: int = 200

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
        """Resolve a bearer token to (actor_id, role) or None if unknown.

        Uses hmac.compare_digest for constant-time comparison to prevent
        timing-side-channel token enumeration.
        """
        if hmac.compare_digest(token, self.analyst_token):
            return (self.analyst_actor_id, ActorRole.ANALYST)
        if hmac.compare_digest(token, self.reviewer_token):
            return (self.reviewer_actor_id, ActorRole.REVIEWER)
        return None

    def validate_startup(self) -> None:
        """Refuse to start if any token is still the default.

        Call this at app startup. Raises ValueError if tokens haven't
        been changed from their .env.example defaults.
        """
        problems: list[str] = []
        if hmac.compare_digest(self.analyst_token, _DEFAULT_ANALYST_TOKEN):
            problems.append(
                "ANALYST_TOKEN is still the default. Set a real value in .env "
                "before starting the server."
            )
        if hmac.compare_digest(self.reviewer_token, _DEFAULT_REVIEWER_TOKEN):
            problems.append(
                "REVIEWER_TOKEN is still the default. Set a real value in .env "
                "before starting the server."
            )
        if problems:
            raise ValueError("Startup refused: " + " | ".join(problems))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Tests override via dependency injection."""
    return Settings()
