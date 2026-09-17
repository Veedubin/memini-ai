import asyncio
import math
import time

import pytest

from memini_ai.config import Settings
from memini_ai.embed import BgeM3Embedder, EmbedError, HashEmbedder, make_embedder


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


class _FakeModel:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        return [[0.0] * 1024 for _ in texts]


async def test_slow_load_survives_caller_timeout_and_loads_once(monkeypatch):
    """A timed-out first call must not discard the in-flight load or start a second one."""
    e = BgeM3Embedder()
    calls = 0

    def slow_load():
        nonlocal calls
        calls += 1
        time.sleep(0.3)
        return _FakeModel()

    monkeypatch.setattr(e, "_load_sync", slow_load)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(e.embed(["x"]), 0.05)
    out = await e.embed(["x"])
    assert len(out) == 1 and len(out[0]) == 1024
    assert calls == 1
    assert e.last_error is None


async def test_load_failure_sets_last_error_and_is_retried(monkeypatch):
    e = BgeM3Embedder()
    calls = 0

    def boom():
        nonlocal calls
        calls += 1
        raise RuntimeError("no model on disk")

    monkeypatch.setattr(e, "_load_sync", boom)
    with pytest.raises(EmbedError):
        await e.embed(["x"])
    assert e.last_error is not None and "no model on disk" in e.last_error
    with pytest.raises(EmbedError):
        await e.embed(["x"])
    assert calls == 2


def test_hash_embedder_has_no_error_attribute_set():
    assert HashEmbedder().last_error is None
