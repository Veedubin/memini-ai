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
