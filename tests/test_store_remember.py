import pytest

from memini_ai.config import Settings
from memini_ai.embed import HashEmbedder
from memini_ai.store import ChainStep, Store, StoreError, content_hash


@pytest.fixture
def store(db):
    return Store(db, HashEmbedder(), Settings(model="hash", project="proj"))


async def test_remember_returns_id_and_persists(store, db):
    r = await store.remember("We chose asyncpg over psycopg", kind="decision", tags=["db"])
    assert set(r) == {"id", "kind", "project", "duplicate"}
    assert r["kind"] == "decision" and r["project"] == "proj" and r["duplicate"] is False
    row = await db.fetchrow("SELECT text, kind, project, tags, embedding_model FROM memories WHERE id=$1", r["id"])
    assert row["text"] == "We chose asyncpg over psycopg"
    assert list(row["tags"]) == ["db"] and row["embedding_model"] == "hash"


async def test_duplicate_returns_existing_id(store):
    a = await store.remember("same text  here")
    b = await store.remember("same   text here")
    assert b["id"] == a["id"] and b["duplicate"] is True


async def test_duplicate_is_per_project(store):
    a = await store.remember("shared text", project="x")
    b = await store.remember("shared text", project="y")
    assert a["id"] != b["id"]


async def test_supersedes_marks_old_row(store, db):
    old = await store.remember("port is 5434", kind="fact")
    new = await store.remember("port is 5555", kind="fact", supersedes=old["id"])
    assert new["superseded_id"] == old["id"]
    row = await db.fetchrow("SELECT superseded_by FROM memories WHERE id=$1", old["id"])
    assert str(row["superseded_by"]) == new["id"]


async def test_supersedes_unknown_id_errors(store):
    with pytest.raises(StoreError, match="supersedes"):
        await store.remember("x", supersedes="00000000-0000-0000-0000-000000000000")


async def test_duplicate_with_supersedes_still_marks_old_row(store, db):
    old = await store.remember("port is 5434", kind="fact")
    new = await store.remember("port is 5555", kind="fact")
    again = await store.remember("port is 5555", kind="fact", supersedes=old["id"])
    assert again["duplicate"] is True and again["id"] == new["id"]
    assert again["superseded_id"] == old["id"]
    row = await db.fetchrow("SELECT superseded_by FROM memories WHERE id=$1", old["id"])
    assert str(row["superseded_by"]) == new["id"]


async def test_supersedes_self_is_error(store):
    a = await store.remember("self ref")
    with pytest.raises(StoreError, match="same memory"):
        await store.remember("self ref", supersedes=a["id"])


async def test_chain_cannot_cross_projects(store):
    t1 = await store.remember(
        "a", kind="thought", project="x", chain=ChainStep(number=1, total=2, next_needed=True)
    )
    with pytest.raises(StoreError, match="belongs to project"):
        await store.remember(
            "b",
            kind="thought",
            project="y",
            chain=ChainStep(chain_id=t1["chain_id"], number=2, total=2, next_needed=False),
        )


async def test_rejects_bad_kind_empty_text_and_oversize(store):
    with pytest.raises(StoreError, match="kind"):
        await store.remember("x", kind="poem")
    with pytest.raises(StoreError, match="empty"):
        await store.remember("   ")
    with pytest.raises(StoreError, match="32768"):
        await store.remember("a" * 40000)
    with pytest.raises(StoreError, match="session"):
        await store.remember("x", kind="session")


def test_content_hash_normalizes_whitespace():
    assert content_hash("a  b\n c") == content_hash("a b c")
