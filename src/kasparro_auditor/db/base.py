"""Async SQLAlchemy 2.0 engine, session factory, FastAPI dependency.

SQLite foreign keys are ENABLED via a connection event listener
(`PRAGMA foreign_keys=ON`). Without this, SQLite silently accepts
orphan rows — the brief requires referential integrity.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy import event
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
    # SQLite needs this to allow multiple threads sharing the same connection.
    connect_args={"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {},
)

# Enable SQLite foreign key enforcement on every connection.
# SQLite has FKs OFF by default — without this, orphan citations and
# responses with unknown prompt_id would import silently.

if _settings.database_url.startswith("sqlite"):

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fk(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


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
