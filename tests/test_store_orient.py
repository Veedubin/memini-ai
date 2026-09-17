import pytest

from memini_ai.config import Settings
from memini_ai.embed import HashEmbedder
from memini_ai.store import ChainStep, Store


@pytest.fixture
def store(db):
    return Store(db, HashEmbedder(), Settings(model="hash", project="proj"))


async def test_orient_empty_store(store):
    r = await store.orient()
    assert r["status"] == {"db": "ok", "model": "hash", "memories": 0}
    assert r["decisions"] == [] and r["handoffs"] == [] and r["open_chains"] == 0
    assert "no memories" in r["text"].lower()


async def test_orient_collects_recent_items_and_respects_budget(store):
    for i in range(7):
        await store.remember(f"decision number {i} about the database layer", kind="decision")
    await store.remember("handoff: next session should write docs", kind="handoff")
    await store.remember("kitchen note", project="kitchen")
    await store.remember("thinking", kind="thought", chain=ChainStep(number=1, total=3, next_needed=True))
    r = await store.orient()
    assert len(r["decisions"]) == 5 and r["decisions"][0]["text"].startswith("decision number 6")
    assert len(r["handoffs"]) == 1
    assert r["open_chains"] == 1
    assert {p["project"] for p in r["projects"]} == {"proj", "kitchen"}
    assert len(r["text"]) // 4 <= 300
    small = await store.orient(budget=60)
    assert len(small["text"]) // 4 <= 60


async def test_orient_project_filter(store):
    await store.remember("d1", kind="decision", project="a")
    await store.remember("d2", kind="decision", project="b")
    r = await store.orient(project="a")
    assert [d["text"] for d in r["decisions"]] == ["d1"]
    assert [p["project"] for p in r["projects"]] == ["a"]
