import math

from memini_ai.config import Settings
from memini_ai.embed import HashEmbedder, make_embedder


async def test_hash_embedder_is_deterministic_and_normalized():
    e = HashEmbedder()
    a, b = await e.embed(["postgres migration failed", "postgres migration failed"])
    assert a == b
    assert len(a) == 1024
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-6)


async def test_hash_embedder_similar_texts_are_closer():
    e = HashEmbedder()
    a, b, c = await e.embed(
        ["postgres migration failed on startup", "the postgres migration failed", "banana bread recipe"]
    )
    dot = lambda x, y: sum(p * q for p, q in zip(x, y, strict=True))  # noqa: E731
    assert dot(a, b) > dot(a, c)


def test_make_embedder_selects_hash():
    e = make_embedder(Settings(model="hash"))
    assert e.name == "hash" and e.dim == 1024


def test_make_embedder_selects_bge_lazy():
    e = make_embedder(Settings(model="BAAI/bge-m3", device="cpu"))
    assert e.name == "BAAI/bge-m3" and e.dim == 1024
