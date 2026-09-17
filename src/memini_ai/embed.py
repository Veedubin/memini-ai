"""Embedding backends. BGE-M3 for real use, a hash-based fake for tests."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Any, Protocol

import structlog

from memini_ai.config import Settings

log = structlog.get_logger(__name__)

DIM = 1024
_WORD_RE = re.compile(r"[a-z0-9]+")


class EmbedError(Exception):
    """Raised when the model cannot be loaded or run."""


class Embedder(Protocol):
    name: str
    dim: int
    last_error: str | None
    """Last model load/encode failure, or None. Surfaced by Store.status()."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Bag-of-words hashed into a fixed vector. Overlapping words give similar vectors."""

    name = "hash"
    dim = DIM
    last_error: str | None = None  # never fails

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in _WORD_RE.findall(text.lower()):
            h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "big")
            vec[h % DIM] += 1.0 if (h >> 63) == 0 else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class BgeM3Embedder:
    name = "BAAI/bge-m3"
    dim = DIM

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._model: Any = None
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[Any] | None = None
        self.last_error: str | None = None

    async def _load(self) -> Any:
        """Load once, even if callers time out.

        The load runs in a task owned by the instance and every caller awaits it through
        `asyncio.shield`, so a caller that is cancelled (a tool timeout, say) abandons its own
        await while the load keeps going; the next caller joins the same task instead of
        starting a second multi-second model load. The task is only cleared when it *failed*,
        which is what makes the next call retry.
        """
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model
            if self._task is None:
                log.info("model_loading", model=self.name, device=self._device)
                self._task = asyncio.create_task(asyncio.to_thread(self._load_sync))
            task = self._task
        try:
            model = await asyncio.shield(task)
        except Exception as e:  # sentence-transformers raises many types
            # Only the task's own failure lands here; a cancelled *caller* raises
            # CancelledError (a BaseException), which leaves self._task in place.
            self._task = None
            self.last_error = f"cannot load {self.name}: {e}"
            raise EmbedError(self.last_error) from e
        self._model = model
        self.last_error = None
        log.info("model_loaded", model=self.name)
        return model

    def _load_sync(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.name, device=self._device)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = await self._load()
        try:
            arr = await asyncio.to_thread(
                model.encode, texts, normalize_embeddings=True, convert_to_numpy=True
            )
        except Exception as e:
            self.last_error = f"embedding failed: {e}"
            raise EmbedError(self.last_error) from e
        self.last_error = None
        return [[float(x) for x in row] for row in arr]


def make_embedder(settings: Settings) -> Embedder:
    if settings.model == "hash":
        return HashEmbedder()
    if settings.model == "BAAI/bge-m3":
        return BgeM3Embedder(device=settings.device)
    raise EmbedError(f"unsupported MEMINI_MODEL={settings.model!r}; use BAAI/bge-m3 or hash")
