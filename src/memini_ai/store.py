"""The three operations: remember, recall, orient. Thought chains ride on remember/recall."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

import asyncpg
import structlog
from pydantic import BaseModel, Field

from memini_ai.config import Settings
from memini_ai.db import Database
from memini_ai.embed import Embedder, EmbedError

log = structlog.get_logger(__name__)

KINDS = ("note", "decision", "handoff", "fact", "thought", "session")
_WS = re.compile(r"\s+")


class StoreError(Exception):
    """User-facing failure; the message is returned to the agent verbatim."""


class ChainStep(BaseModel):
    chain_id: str | None = Field(default=None, description="Omit on the first thought to create a chain.")
    number: int = Field(ge=1, description="1-based position of this thought.")
    total: int = Field(ge=1, description="Current estimate of total thoughts.")
    next_needed: bool = Field(description="False on the final thought; closes the chain.")
    revises: int | None = Field(default=None, description="Thought number this one revises.")
    branch_from: int | None = Field(default=None, description="Thought number this one branches from.")


def content_hash(text: str) -> str:
    return hashlib.sha256(_WS.sub(" ", text).strip().encode()).hexdigest()


def _uuid(value: str, field: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError as e:
        raise StoreError(f"{field} is not a valid id: {value!r}") from e


class Store:
    def __init__(self, db: Database, embedder: Embedder, settings: Settings) -> None:
        self._db = db
        self._embed = embedder
        self._settings = settings

    async def remember(
        self,
        text: str,
        kind: str = "note",
        project: str | None = None,
        tags: list[str] | None = None,
        supersedes: str | None = None,
        chain: ChainStep | None = None,
        source: dict[str, Any] | None = None,
        allow_session: bool = False,
    ) -> dict[str, Any]:
        if kind not in KINDS:
            raise StoreError(f"kind must be one of {', '.join(KINDS)}")
        if kind == "session" and not allow_session:
            raise StoreError("kind 'session' is reserved for ingested transcripts")
        if not text or not text.strip():
            raise StoreError("text is empty")
        if len(text.encode()) > self._settings.max_text_bytes:
            raise StoreError(f"text exceeds {self._settings.max_text_bytes} bytes")
        if chain is not None and kind != "thought":
            raise StoreError("chain requires kind='thought'")
        if kind == "thought" and chain is None:
            raise StoreError("kind='thought' requires chain")

        project = project or self._settings.project
        h = content_hash(text)
        existing = await self._db.fetchrow(
            "SELECT id FROM memories WHERE coalesce(project,'') = $1 AND content_hash = $2",
            project or "", h,
        )
        if existing is not None and chain is None:
            return {"id": str(existing["id"]), "kind": kind, "project": project, "duplicate": True}

        superseded: uuid.UUID | None = None
        if supersedes is not None:
            superseded = _uuid(supersedes, "supersedes")
            if await self._db.fetchrow("SELECT 1 FROM memories WHERE id=$1", superseded) is None:
                raise StoreError(f"supersedes id not found: {supersedes}")

        chain_id = await self._resolve_chain(chain, project) if chain is not None else None

        try:
            vectors = await self._embed.embed([text])
            vector: list[float] | None = vectors[0]
            model: str | None = self._embed.name
        except EmbedError as e:
            log.warning("embed_failed_storing_without_vector", error=str(e))
            vector, model = None, None

        try:
            row = await self._db.fetchrow(
                """
                INSERT INTO memories (text, kind, project, tags, content_hash, embedding, embedding_model,
                    chain_id, thought_number, thought_total, next_needed, revises, branch_from, source)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14::jsonb)
                RETURNING id
                """,
                text, kind, project, tags or [], h, vector, model,
                chain_id,
                chain.number if chain else None,
                chain.total if chain else None,
                chain.next_needed if chain else None,
                chain.revises if chain else None,
                chain.branch_from if chain else None,
                _json(source or {}),
            )
        except asyncpg.exceptions.UniqueViolationError:
            dup = await self._db.fetchrow(
                "SELECT id FROM memories WHERE coalesce(project,'') = $1 AND content_hash = $2",
                project or "", h,
            )
            assert dup is not None
            return {"id": str(dup["id"]), "kind": kind, "project": project, "duplicate": True}
        assert row is not None
        new_id = str(row["id"])

        if superseded is not None:
            await self._db.execute("UPDATE memories SET superseded_by=$1 WHERE id=$2", row["id"], superseded)

        out: dict[str, Any] = {"id": new_id, "kind": kind, "project": project, "duplicate": False}
        if superseded is not None:
            out["superseded_id"] = str(superseded)
        if chain_id is not None:
            out["chain_id"] = str(chain_id)
            if chain is not None and not chain.next_needed:
                await self._db.execute(
                    "UPDATE chains SET status='done', updated_at=now() WHERE id=$1", chain_id
                )
        return out

    async def _resolve_chain(self, chain: ChainStep, project: str | None) -> uuid.UUID:
        if chain.chain_id is None:
            row = await self._db.fetchrow(
                "INSERT INTO chains (project) VALUES ($1) RETURNING id", project
            )
            assert row is not None
            return uuid.UUID(str(row["id"]))
        cid = _uuid(chain.chain_id, "chain_id")
        if await self._db.fetchrow("SELECT 1 FROM chains WHERE id=$1", cid) is None:
            raise StoreError(f"chain_id not found: {chain.chain_id}")
        await self._db.execute("UPDATE chains SET updated_at=now(), status='open' WHERE id=$1", cid)
        return cid


def _json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, default=str)
