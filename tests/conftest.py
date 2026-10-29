"""Pytest configuration."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Set test env BEFORE importing the app.
os.environ.setdefault("EXECUTION_MODE", "stub")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("ANALYST_TOKEN", "test-analyst-token")
os.environ.setdefault("REVIEWER_TOKEN", "test-reviewer-token")
os.environ.setdefault("GROQ_API_KEY", "")

from kasparro_auditor.config import get_settings  # noqa: E402
from kasparro_auditor.db import base as db_base  # noqa: E402
from kasparro_auditor.db.base import Base  # noqa: E402
from kasparro_auditor.main import create_app  # noqa: E402


@pytest.fixture(scope="session")
def event_loop():
    """Use a single event loop for the whole session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def engine():
    """Fresh in-memory engine per test for isolation.

    Also patches the module-level engine + session_factory used by the
    rest of the codebase so all DB writes go through this engine.
    """
    eng = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False, future=True)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Patch the module-level references so get_session picks up the test engine
    test_session_factory = async_sessionmaker(bind=eng, expire_on_commit=False)
    original_engine = db_base.engine
    original_factory = db_base.async_session_factory
    db_base.engine = eng
    db_base.async_session_factory = test_session_factory

    # Also patch get_session's closure if it captured the originals
    import kasparro_auditor.db.base as base_mod

    original_get_session = base_mod.get_session

    async def _test_get_session():
        async with test_session_factory() as s:
            try:
                yield s
            except Exception:
                await s.rollback()
                raise

    base_mod.get_session = _test_get_session

    yield eng

    db_base.engine = original_engine
    db_base.async_session_factory = original_factory
    base_mod.get_session = original_get_session
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return db_base.async_session_factory


@pytest_asyncio.fixture
async def session(session_factory):
    async with session_factory() as s:
        yield s


@pytest_asyncio.fixture
async def client(engine):
    """HTTP client wired to the FastAPI app."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def analyst_headers():
    return {"Authorization": "Bearer test-analyst-token"}


@pytest.fixture
def reviewer_headers():
    return {"Authorization": "Bearer test-reviewer-token"}


@pytest.fixture
def data_pack_path() -> Path:
    """Absolute path to the supplied assignment pack."""
    here = Path(__file__).resolve().parent.parent
    p = here / "data_pack"
    if not p.is_dir():
        pytest.skip(f"data_pack not available at {p}")
    return p
