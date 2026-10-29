"""Async SQLAlchemy 2.0 engine, session factory, FastAPI dependency."""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from kasparro_auditor.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_settings = get_settings()
engine = create_async_engine(
    _settings.database_url,
    echo=False,
    future=True,
    # SQLite needs this to enable FK constraints on connection.
    connect_args={"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {},
)

async_session_factory = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields a session and rolls back on exception."""
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
