"""asyncpg pool, migration runner, and thin query helpers. All SQL lives here or in store.py."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from typing import Any

import asyncpg
import structlog
from pgvector.asyncpg import register_vector

log = structlog.get_logger(__name__)

_MIGRATION_RE = re.compile(r"^(\d{4})_.+\.sql$")

# A dropped server, a closed pool or a connection released mid-query: the caller can retry.
_LOST_CONNECTION = (asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError)

# Any advisory lock id will do as long as every memini-ai process agrees on it.
_MIGRATION_LOCK = 7264726


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

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        """Hold one pooled connection with an open transaction.

        Pass the yielded connection to fetch/fetchrow/execute as `conn=` so those statements
        join the transaction instead of acquiring their own connection. Leaving the block with
        an exception rolls everything back.
        """
        async with self.pool.acquire() as conn, conn.transaction():
            yield conn

    async def fetch(
        self, sql: str, *args: Any, conn: asyncpg.Connection | None = None
    ) -> list[asyncpg.Record]:
        try:
            if conn is not None:
                return list(await conn.fetch(sql, *args))
            async with self.pool.acquire() as c:
                return list(await c.fetch(sql, *args))
        except _LOST_CONNECTION as e:
            raise DatabaseError("connection lost; retry") from e

    async def fetchrow(
        self, sql: str, *args: Any, conn: asyncpg.Connection | None = None
    ) -> asyncpg.Record | None:
        try:
            if conn is not None:
                row: asyncpg.Record | None = await conn.fetchrow(sql, *args)
                return row
            async with self.pool.acquire() as c:
                return await c.fetchrow(sql, *args)
        except _LOST_CONNECTION as e:
            raise DatabaseError("connection lost; retry") from e

    async def execute(self, sql: str, *args: Any, conn: asyncpg.Connection | None = None) -> str:
        try:
            if conn is not None:
                return str(await conn.execute(sql, *args))
            async with self.pool.acquire() as c:
                return str(await c.execute(sql, *args))
        except _LOST_CONNECTION as e:
            raise DatabaseError("connection lost; retry") from e


def _load_migrations() -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for entry in resources.files("memini_ai.migrations").iterdir():
        m = _MIGRATION_RE.match(entry.name)
        if m:
            out.append((int(m.group(1)), entry.name, entry.read_text(encoding="utf-8")))
    return sorted(out)
