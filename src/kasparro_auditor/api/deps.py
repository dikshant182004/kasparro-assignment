"""Auth dependency: resolve bearer token to (actor_id, role).

Identity comes ONLY from a server-side token check. A caller-supplied
role or actor name is NEVER authorisation. (Brief requirement.)
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header
from typing_extensions import Annotated

from kasparro_auditor.api.errors import api_error
from kasparro_auditor.config import ActorRole, Settings, get_settings


@dataclass(slots=True)
class Actor:
    actor_id: str
    role: ActorRole


def _extract_bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


async def get_actor(
    authorization: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> Actor:
    """Resolve the bearer token to an authenticated Actor."""
    token = _extract_bearer(authorization)
    if token is None:
        raise api_error("missing_token")
    pair = settings.actor_for_token(token)
    if pair is None:
        raise api_error("unknown_token")
    actor_id, role = pair
    return Actor(actor_id=actor_id, role=role)


async def require_analyst(actor: Actor = Depends(get_actor)) -> Actor:
    if actor.role != ActorRole.ANALYST:
        raise api_error("wrong_role", "Analyst role required.")
    return actor


async def require_reviewer(actor: Actor = Depends(get_actor)) -> Actor:
    if actor.role != ActorRole.REVIEWER:
        raise api_error("wrong_role", "Reviewer role required.")
    return actor
