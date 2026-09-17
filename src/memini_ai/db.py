"""asyncpg pool, migration runner, and thin query helpers. All SQL lives here or in store.py."""

from __future__ import annotations

import re
from importlib import resources
from typing import Any

import asyncpg
import structlog
from pgvector.asyncpg import register_vector

log = structlog.get_logger(__name__)

_MIGRATION_RE = re.compile(r"^(\d{4})_.+\.sql$")


class DatabaseError(Exception):
    """Raised when the database cannot be reached or migrated."""


async def _init_conn(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


class Database:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise DatabaseError("database not connected")
        return self._pool

    async def connect(self) -> None:
        """Apply migrations with a plain connection, then open the pool with the vector codec."""
        try:
            await self.migrate()
            self._pool = await asyncpg.create_pool(
                self._dsn, min_size=1, max_size=4, init=_init_conn, command_timeout=60
            )
        except (OSError, asyncpg.PostgresError) as e:
            raise DatabaseError(f"cannot connect to {self._redacted()}: {e}") from e

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _redacted(self) -> str:
        return re.sub(r"://([^:]+):[^@]*@", r"://\1:***@", self._dsn)

    async def migrate(self) -> list[int]:
        """Apply pending numbered migrations in order, each in its own transaction."""
        try:
            conn = await asyncpg.connect(self._dsn)
        except (OSError, asyncpg.PostgresError) as e:
            raise DatabaseError(f"cannot connect to {self._redacted()}: {e}") from e
        try:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version int PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            applied = {r["version"] for r in await conn.fetch("SELECT version FROM schema_migrations")}
            done: list[int] = []
            for version, name, sql in _load_migrations():
                if version in applied:
                    continue
                async with conn.transaction():
                    await conn.execute(sql)
                    await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
                log.info("migration_applied", version=version, name=name)
                done.append(version)
            return done
        except asyncpg.PostgresError as e:
            raise DatabaseError(f"migration failed: {e}") from e
        finally:
            await conn.close()

    async def fetch(self, sql: str, *args: Any) -> list[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            return list(await conn.fetch(sql, *args))

    async def fetchrow(self, sql: str, *args: Any) -> asyncpg.Record | None:
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(sql, *args)

    async def execute(self, sql: str, *args: Any) -> str:
        async with self.pool.acquire() as conn:
            return str(await conn.execute(sql, *args))


def _load_migrations() -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for entry in resources.files("memini_ai.migrations").iterdir():
        m = _MIGRATION_RE.match(entry.name)
        if m:
            out.append((int(m.group(1)), entry.name, entry.read_text(encoding="utf-8")))
    return sorted(out)
