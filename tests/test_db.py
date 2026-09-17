import pytest

from memini_ai.db import Database


async def test_migrate_is_idempotent(test_dsn, raw_conn):
    await raw_conn.execute("DROP TABLE IF EXISTS memories, chains, schema_migrations CASCADE")
    db = Database(test_dsn)
    await db.connect()
    try:
        first = await db.migrate()
        second = await db.migrate()
    finally:
        await db.close()
    assert first == [] or first == [1]  # connect() already applied 0001
    assert second == []
    rows = await raw_conn.fetch("SELECT version FROM schema_migrations ORDER BY version")
    assert [r["version"] for r in rows] == [1]
    cols = await raw_conn.fetch(
        "SELECT column_name FROM information_schema.columns WHERE table_name='memories'"
    )
    names = {r["column_name"] for r in cols}
    assert {"text", "kind", "embedding", "tsv", "superseded_by", "chain_id"} <= names


async def test_vector_roundtrip(db: Database):
    vec = [0.1] * 1024
    row = await db.fetchrow(
        "INSERT INTO memories (text, kind, content_hash, embedding) VALUES ($1,$2,$3,$4) RETURNING embedding",
        "hello", "note", "h1", vec,
    )
    assert row is not None
    assert len(row["embedding"].to_list()) == 1024


async def test_connect_failure_raises_database_error():
    from memini_ai.db import DatabaseError

    db = Database("postgresql://memini:wrong@localhost:5555/memini_test")
    try:
        await db.connect()
    except DatabaseError as e:
        assert "connect" in str(e).lower() or "authentication" in str(e).lower()
    else:
        raise AssertionError("expected DatabaseError")


async def test_transaction_rolls_back_on_error(db: Database):
    with pytest.raises(RuntimeError):
        async with db.transaction() as conn:
            await db.execute(
                "INSERT INTO memories (text, kind, content_hash) VALUES ('t','note','tx-1')", conn=conn
            )
            assert await db.fetchrow("SELECT 1 FROM memories WHERE content_hash='tx-1'", conn=conn)
            raise RuntimeError("boom")
    assert await db.fetchrow("SELECT 1 FROM memories WHERE content_hash='tx-1'") is None


async def test_concurrent_migrations_on_a_fresh_db_do_not_race(test_dsn, raw_conn):
    import asyncio

    await raw_conn.execute("DROP TABLE IF EXISTS memories, chains, schema_migrations CASCADE")
    results = await asyncio.gather(*(Database(test_dsn).migrate() for _ in range(3)))
    assert sorted(len(r) for r in results) == [0, 0, 1]  # exactly one runner applied 0001
    rows = await raw_conn.fetch("SELECT version FROM schema_migrations")
    assert [r["version"] for r in rows] == [1]


async def test_close_is_idempotent(test_dsn):
    db = Database(test_dsn)
    await db.connect()
    await db.close()
    await db.close()
