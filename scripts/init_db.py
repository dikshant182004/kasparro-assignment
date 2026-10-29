"""Initialize the SQLite database (creates all tables)."""
from __future__ import annotations

import asyncio

from kasparro_auditor.db.base import Base, engine


async def main() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print(f"Database initialised at: {engine.url}")


if __name__ == "__main__":
    asyncio.run(main())
