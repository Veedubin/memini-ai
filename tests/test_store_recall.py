from datetime import UTC, datetime, timedelta

import pytest

from memini_ai.config import Settings
from memini_ai.embed import HashEmbedder
from memini_ai.search import build_filters, parse_since, rrf
from memini_ai.store import Store


def test_rrf_prefers_items_in_both_lists():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]])
    assert [x for x, _ in fused][:2] == ["a", "c"]
    assert fused[0][1] > fused[-1][1]


def test_parse_since_relative_and_absolute():
    now = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
    assert parse_since("7d", now) == now - timedelta(days=7)
    assert parse_since("24h", now) == now - timedelta(hours=24)
    assert parse_since("2026-09-01", now) == datetime(2026, 9, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        parse_since("yesterday", now)


def test_build_filters_numbers_params_from_first_param():
    sql, params = build_filters("note", "p", None, False, first_param=3)
    assert sql == " AND kind = $3 AND project = $4 AND superseded_by IS NULL"
    assert params == ["note", "p"]
    sql, params = build_filters(None, None, None, True, first_param=1)
    assert sql == "" and params == []


@pytest.fixture
def store(db):
    return Store(db, HashEmbedder(), Settings(model="hash", project="proj"))


async def seed(store):
    await store.remember("Postgres port for memini is 5555", kind="fact")
    await store.remember("We migrated from Qdrant to pgvector for simplicity", kind="decision")
    await store.remember("Banana bread needs ripe bananas", kind="note", project="kitchen")
    old = await store.remember("Embedding model is MiniLM", kind="fact")
    await store.remember("Embedding model is BGE-M3", kind="fact", supersedes=old["id"])
    return old


async def test_recall_hybrid_returns_relevant_first(store):
    await seed(store)
    r = await store.recall("which port does postgres use")
    assert "degraded" not in r
    assert r["results"][0]["text"].startswith("Postgres port")
    top = r["results"][0]
    assert set(top) >= {"id", "text", "kind", "project", "tags", "created_at", "score"}
    assert top["score"] == 1.0
    assert "embedding" not in top


async def test_recall_filters_by_kind_and_project(store):
    await seed(store)
    r = await store.recall("bananas", project="kitchen")
    assert [x["project"] for x in r["results"]] == ["kitchen"]
    r = await store.recall("pgvector", kind="decision")
    assert all(x["kind"] == "decision" for x in r["results"]) and r["results"]


async def test_recall_hides_superseded_unless_asked(store):
    old = await seed(store)
    r = await store.recall("embedding model")
    ids = [x["id"] for x in r["results"]]
    assert old["id"] not in ids
    r2 = await store.recall("embedding model", include_superseded=True)
    hit = next(x for x in r2["results"] if x["id"] == old["id"])
    assert hit["superseded_by"] is not None


async def test_recall_increments_retrieval_count(store, db):
    await seed(store)
    r = await store.recall("postgres port", limit=1)
    row = await db.fetchrow("SELECT retrieval_count FROM memories WHERE id=$1", r["results"][0]["id"])
    assert row["retrieval_count"] == 1


async def test_recall_since_filter(store, db):
    await seed(store)
    await db.execute("UPDATE memories SET created_at = now() - interval '30 days'")
    assert (await store.recall("postgres", since="7d"))["results"] == []
    assert (await store.recall("postgres", since="60d"))["results"]


async def test_recall_text_only_when_embedder_fails(db):
    class Broken:
        name = "broken"
        dim = 1024

        async def embed(self, texts):
            from memini_ai.embed import EmbedError

            raise EmbedError("no model")

    s = Store(db, Broken(), Settings(model="hash"))
    await s.remember("full text search still finds this sentence")
    r = await s.recall("finds sentence")
    assert r.get("degraded") == "text-only"
    assert r["results"] and "still finds" in r["results"][0]["text"]
