import os

import pytest

from memini_ai.config import Settings
from memini_ai.embed import make_embedder
from memini_ai.store import Store

pytestmark = pytest.mark.skipif(
    os.environ.get("MEMINI_E2E") != "1", reason="set MEMINI_E2E=1 to run against BGE-M3"
)


async def test_real_model_recall_quality(db):
    settings = Settings(model="BAAI/bge-m3", device="cpu", project="e2e")
    store = Store(db, make_embedder(settings), settings)
    await store.remember("The memory database listens on port 5555 inside a podman container.", kind="fact")
    await store.remember("We replaced the regex entity extractor because it stored every English word.", kind="decision")
    await store.remember("Sourdough needs a 12 hour bulk ferment at room temperature.", kind="note")
    r = await store.recall("which port is the db on")
    assert "5555" in r["results"][0]["text"]
    r = await store.recall("why did we drop entity extraction")
    assert "regex" in r["results"][0]["text"]
