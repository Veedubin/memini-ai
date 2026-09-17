import pytest

from memini_ai.config import Settings
from memini_ai.embed import HashEmbedder
from memini_ai.store import ChainStep, Store, StoreError


@pytest.fixture
def store(db):
    return Store(db, HashEmbedder(), Settings(model="hash", project="proj"))


async def test_first_thought_creates_chain_and_last_closes_it(store, db):
    t1 = await store.remember("Step one: read the spec", kind="thought",
                              chain=ChainStep(number=1, total=2, next_needed=True))
    cid = t1["chain_id"]
    assert cid
    t2 = await store.remember("Step two: done", kind="thought",
                              chain=ChainStep(chain_id=cid, number=2, total=2, next_needed=False))
    assert t2["chain_id"] == cid
    row = await db.fetchrow("SELECT status FROM chains WHERE id=$1::uuid", cid)
    assert row["status"] == "done"
    r = await store.recall("ignored", chain_id=cid)
    assert [x["chain"]["number"] for x in r["results"]] == [1, 2]
    assert r["chain"]["status"] == "done"
    assert r["results"][1]["chain"]["next_needed"] is False


async def test_revision_and_branch_are_recorded(store):
    t1 = await store.remember("A", kind="thought", chain=ChainStep(number=1, total=3, next_needed=True))
    cid = t1["chain_id"]
    await store.remember("A revised", kind="thought",
                         chain=ChainStep(chain_id=cid, number=2, total=3, next_needed=True, revises=1))
    await store.remember("B alt", kind="thought",
                         chain=ChainStep(chain_id=cid, number=3, total=3, next_needed=True, branch_from=1))
    r = await store.recall("x", chain_id=cid)
    assert r["results"][1]["chain"]["revises"] == 1
    assert r["results"][2]["chain"]["branch_from"] == 1
    assert r["chain"]["status"] == "open"


async def test_repeated_thought_text_in_same_chain_is_allowed(store):
    t1 = await store.remember("same", kind="thought", chain=ChainStep(number=1, total=2, next_needed=True))
    t2 = await store.remember("same", kind="thought",
                              chain=ChainStep(chain_id=t1["chain_id"], number=2, total=2, next_needed=False))
    assert t2["duplicate"] is True  # unique index still collapses identical rows


async def test_chain_errors(store):
    with pytest.raises(StoreError, match="requires chain"):
        await store.remember("x", kind="thought")
    with pytest.raises(StoreError, match="requires kind"):
        await store.remember("x", chain=ChainStep(number=1, total=1, next_needed=False))
    with pytest.raises(StoreError, match="chain_id not found"):
        await store.remember("x", kind="thought",
                             chain=ChainStep(chain_id="00000000-0000-0000-0000-000000000000",
                                             number=1, total=1, next_needed=False))
    with pytest.raises(StoreError, match="not a valid id"):
        await store.recall("x", chain_id="nope")
