"""The three operations: remember, recall, orient. Thought chains ride on remember/recall."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime
from typing import Any

import asyncpg
import structlog
from pydantic import BaseModel, Field

from memini_ai.config import Settings
from memini_ai.db import Database
from memini_ai.embed import Embedder, EmbedError
from memini_ai.search import build_filters, parse_since, rrf

log = structlog.get_logger(__name__)

KINDS = ("note", "decision", "handoff", "fact", "thought", "session")
_DUP_SQL = (
    "SELECT id, kind, project, superseded_by FROM memories "
    "WHERE coalesce(project,'') = $1 AND content_hash = $2"
)
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
    ARM_LIMIT = 20

    def __init__(self, db: Database, embedder: Embedder, settings: Settings) -> None:
        self._db = db
        self._embed = embedder
        self._settings = settings

    async def recall(
        self,
        query: str,
        limit: int = 8,
        kind: str | None = None,
        project: str | None = None,
        since: str | None = None,
        include_superseded: bool = False,
        chain_id: str | None = None,
    ) -> dict[str, Any]:
        if limit < 1 or limit > 50:
            raise StoreError("limit must be between 1 and 50")
        if kind is not None and kind not in KINDS:
            raise StoreError(f"kind must be one of {', '.join(KINDS)}")
        if chain_id is not None:
            return await self._recall_chain(chain_id)
        if not query or not query.strip():
            raise StoreError("query is empty")
        since_dt: datetime | None = None
        if since is not None:
            try:
                since_dt = parse_since(since)
            except ValueError as e:
                raise StoreError(str(e)) from e

        where, params = build_filters(kind, project, since_dt, include_superseded, first_param=2)
        text_ids = [
            str(r["id"])
            for r in await self._db.fetch(
                "SELECT id FROM memories WHERE tsv @@ websearch_to_tsquery('english', $1)" + where
                + " ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', $1)) DESC LIMIT "
                + str(self.ARM_LIMIT),
                query, *params,
            )
        ]
        degraded: str | None = None
        vector_ids: list[str] = []
        try:
            qvec = (await self._embed.embed([query]))[0]
            vector_ids = [
                str(r["id"])
                for r in await self._db.fetch(
                    "SELECT id FROM memories WHERE embedding IS NOT NULL" + where
                    + " ORDER BY embedding <=> $1 LIMIT " + str(self.ARM_LIMIT),
                    qvec, *params,
                )
            ]
        except EmbedError as e:
            log.warning("recall_text_only", error=str(e))
            degraded = "text-only"

        fused = rrf([vector_ids, text_ids])[:limit]
        if not fused:
            out: dict[str, Any] = {"results": []}
            if degraded:
                out["degraded"] = degraded
            return out
        ids = [uuid.UUID(i) for i, _ in fused]
        top = fused[0][1]
        rows = await self._db.fetch(
            """
            SELECT id, text, kind, project, tags, created_at, superseded_by, chain_id, thought_number
            FROM memories WHERE id = ANY($1::uuid[])
            """,
            ids,
        )
        by_id = {str(r["id"]): r for r in rows}
        results = []
        for mid, score in fused:
            r = by_id[mid]
            results.append(self._row_out(r, round(score / top, 4)))
        await self._db.execute(
            "UPDATE memories SET retrieval_count = retrieval_count + 1 WHERE id = ANY($1::uuid[])", ids
        )
        out = {"results": results}
        if degraded:
            out["degraded"] = degraded
        return out

    async def _recall_chain(self, chain_id: str) -> dict[str, Any]:
        cid = _uuid(chain_id, "chain_id")
        chain = await self._db.fetchrow("SELECT id, status, project FROM chains WHERE id=$1", cid)
        if chain is None:
            raise StoreError(f"chain_id not found: {chain_id}")
        rows = await self._db.fetch(
            """
            SELECT id, text, kind, project, tags, created_at, superseded_by, chain_id,
                   thought_number, thought_total, next_needed, revises, branch_from
            FROM memories WHERE chain_id=$1 ORDER BY thought_number, created_at
            """,
            cid,
        )
        results = [self._row_out(r, 1.0) for r in rows]
        return {"results": results, "chain": {"id": chain_id, "status": chain["status"], "project": chain["project"]}}

    @staticmethod
    def _row_out(r: Any, score: float) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": str(r["id"]),
            "text": r["text"],
            "kind": r["kind"],
            "project": r["project"],
            "tags": list(r["tags"]),
            "created_at": r["created_at"].isoformat(),
            "score": score,
        }
        if r["superseded_by"] is not None:
            out["superseded_by"] = str(r["superseded_by"])
        if r["chain_id"] is not None:
            out["chain"] = {
                "id": str(r["chain_id"]),
                "number": r["thought_number"],
                **({"total": r["thought_total"], "next_needed": r["next_needed"],
                    "revises": r["revises"], "branch_from": r["branch_from"]}
                   if "thought_total" in r.keys() else {}),  # noqa: SIM118 -- asyncpg.Record iterates values, not keys
            }
        return out

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

        superseded: uuid.UUID | None = None
        if supersedes is not None:
            superseded = _uuid(supersedes, "supersedes")
            if await self._db.fetchrow("SELECT 1 FROM memories WHERE id=$1", superseded) is None:
                raise StoreError(f"supersedes id not found: {supersedes}")

        if chain is None:
            existing = await self._db.fetchrow(_DUP_SQL, project or "", h)
            if existing is not None:
                return await self._duplicate_result(existing, superseded)

        # Embed before opening the transaction: it is the slow part and holding a pooled
        # connection across a model call would pin it for seconds.
        try:
            vector: list[float] | None = (await self._embed.embed([text]))[0]
            model: str | None = self._embed.name
        except EmbedError as e:
            log.warning("embed_failed_storing_without_vector", error=str(e))
            vector, model = None, None

        chain_id: uuid.UUID | None = None
        try:
            # One transaction for the chain row, the insert, the supersede and the close, so a
            # duplicate (or any failure) leaves no orphan chain and no half-applied supersede.
            async with self._db.transaction() as conn:
                if chain is not None:
                    chain_id = await self._resolve_chain(chain, project, conn)
                row = await self._db.fetchrow(
                    """
                    INSERT INTO memories (text, kind, project, tags, content_hash, embedding,
                        embedding_model, chain_id, thought_number, thought_total, next_needed,
                        revises, branch_from, source)
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
                    conn=conn,
                )
                assert row is not None
                new_id_uuid: uuid.UUID = row["id"]
                if superseded is not None:
                    await self._apply_supersede(new_id_uuid, superseded, conn)
                if chain_id is not None and chain is not None and not chain.next_needed:
                    await self._db.execute(
                        "UPDATE chains SET status='done', updated_at=now() WHERE id=$1",
                        chain_id, conn=conn,
                    )
        except asyncpg.exceptions.UniqueViolationError:
            # The transaction rolled back, so nothing above landed. Reconcile with the row
            # that already holds this text.
            dup = await self._db.fetchrow(_DUP_SQL, project or "", h)
            assert dup is not None
            return await self._duplicate_result(dup, superseded, chain)

        out: dict[str, Any] = {"id": str(new_id_uuid), "kind": kind, "project": project,
                               "duplicate": False}
        if vector is None:
            out["degraded"] = "text-only"
        if superseded is not None:
            out["superseded_id"] = str(superseded)
        if chain_id is not None:
            out["chain_id"] = str(chain_id)
        return out

    async def status(self) -> dict[str, Any]:
        try:
            row = await self._db.fetchrow("SELECT count(*) AS n FROM memories")
            n = int(row["n"]) if row else 0
            return {"db": "ok", "model": self._embed.name, "memories": n}
        except Exception as e:  # DatabaseError or asyncpg errors
            return {"db": "error", "model": self._embed.name, "memories": 0, "error": str(e)}

    async def orient(self, project: str | None = None, budget: int = 300) -> dict[str, Any]:
        if budget < 20 or budget > 4000:
            raise StoreError("budget must be between 20 and 4000 tokens")
        status = await self.status()
        if status["db"] != "ok":
            return {"status": status, "decisions": [], "handoffs": [], "open_chains": 0,
                    "projects": [], "text": f"memini-ai unavailable: {status.get('error')}"}
        pwhere, pparams = ("", []) if project is None else (" AND project = $1", [project])
        decisions = await self._db.fetch(
            "SELECT id, text, project, created_at FROM memories WHERE kind='decision' AND superseded_by IS NULL"
            + pwhere + " ORDER BY created_at DESC LIMIT 5", *pparams)
        handoffs = await self._db.fetch(
            "SELECT id, text, project, created_at FROM memories WHERE kind='handoff' AND superseded_by IS NULL"
            + pwhere + " ORDER BY created_at DESC LIMIT 3", *pparams)
        chains = await self._db.fetchrow(
            "SELECT count(*) AS n FROM chains WHERE status='open'" + pwhere, *pparams)
        projects = await self._db.fetch(
            "SELECT project, max(created_at) AS last FROM memories WHERE project IS NOT NULL"
            + pwhere + " GROUP BY project ORDER BY last DESC LIMIT 10", *pparams)

        def item(r: Any) -> dict[str, Any]:
            return {"id": str(r["id"]), "text": r["text"], "project": r["project"],
                    "created_at": r["created_at"].isoformat()}

        out: dict[str, Any] = {
            "status": status,
            "decisions": [item(r) for r in decisions],
            "handoffs": [item(r) for r in handoffs],
            "open_chains": int(chains["n"]) if chains else 0,
            "projects": [{"project": r["project"], "last_note_at": r["last"].isoformat()} for r in projects],
        }
        out["text"] = _render_orient(out, budget, scoped=project is not None)
        return out

    async def _resolve_chain(
        self, chain: ChainStep, project: str | None, conn: asyncpg.Connection
    ) -> uuid.UUID:
        if chain.chain_id is None:
            row = await self._db.fetchrow(
                "INSERT INTO chains (project) VALUES ($1) RETURNING id", project, conn=conn
            )
            assert row is not None
            return uuid.UUID(str(row["id"]))
        cid = _uuid(chain.chain_id, "chain_id")
        found = await self._db.fetchrow("SELECT project FROM chains WHERE id=$1", cid, conn=conn)
        if found is None:
            raise StoreError(f"chain_id not found: {chain.chain_id}")
        chain_project = found["project"]
        if chain_project != project:
            raise StoreError(f"chain_id belongs to project {chain_project!r}, not {project!r}")
        await self._db.execute(
            "UPDATE chains SET updated_at=now(), status='open' WHERE id=$1", cid, conn=conn
        )
        return cid

    async def _apply_supersede(
        self, new_id: uuid.UUID, superseded: uuid.UUID, conn: asyncpg.Connection | None = None
    ) -> None:
        await self._db.execute(
            "UPDATE memories SET superseded_by=$1 WHERE id=$2", new_id, superseded, conn=conn
        )

    async def _duplicate_result(
        self,
        existing: asyncpg.Record,
        superseded: uuid.UUID | None,
        chain: ChainStep | None = None,
    ) -> dict[str, Any]:
        """Describe the row that already holds this text: its kind and project, not the request's."""
        existing_id: uuid.UUID = existing["id"]
        result: dict[str, Any] = {
            "id": str(existing_id),
            "kind": existing["kind"],
            "project": existing["project"],
            "duplicate": True,
        }
        if superseded is not None:
            if superseded == existing_id:
                raise StoreError("supersedes cannot point at the same memory")
            if existing["superseded_by"] == superseded:
                raise StoreError("supersedes would create a cycle")
            await self._apply_supersede(existing_id, superseded)
            result["superseded_id"] = str(superseded)
        # A duplicate thought in a chain that already exists still belongs to that chain, and a
        # duplicate final thought still closes it. A duplicate *first* thought has no chain: the
        # transaction that would have created one rolled back.
        if chain is not None and chain.chain_id is not None:
            cid = _uuid(chain.chain_id, "chain_id")
            result["chain_id"] = str(cid)
            if not chain.next_needed:
                await self._db.execute(
                    "UPDATE chains SET status='done', updated_at=now() WHERE id=$1", cid
                )
        return result


def _json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, default=str)


def _bullet(row: dict[str, Any], scoped: bool) -> str:
    if scoped:
        return f"- {row['text']}"
    return f"- [{row['project'] or '-'}] {row['text']}"


def _render_orient(o: dict[str, Any], budget: int, scoped: bool = True) -> str:
    """Render a compact block and trim to roughly `budget` tokens (4 chars per token).

    When `scoped` is False (a cross-project view), each decision/handoff line is
    prefixed with its project so items from different projects aren't ambiguous.
    """
    st = o["status"]
    lines = [f"memini-ai: {st['memories']} memories, model {st['model']}, {o['open_chains']} open chains."]
    if st["memories"] == 0:
        lines.append("No memories yet. Call remember() after decisions and handoffs.")
    if o["projects"]:
        lines.append("Projects: " + ", ".join(p["project"] for p in o["projects"]))
    if o["decisions"]:
        lines.append("Recent decisions:")
        lines += [_bullet(d, scoped) for d in o["decisions"]]
    if o["handoffs"]:
        lines.append("Latest handoffs:")
        lines += [_bullet(h, scoped) for h in o["handoffs"]]

    limit = budget * 4
    text = "\n".join(lines)
    if len(text) <= limit:
        return text

    kept = list(lines)
    while len(kept) > 1:
        candidate = "\n".join([*kept, "..."])
        if len(candidate) <= limit:
            return candidate
        kept.pop()

    # Only the status line is left; keep it, adding "..." on its own line if it fits,
    # otherwise hard-cut the line itself.
    first = kept[0]
    candidate = first + "\n..."
    if len(candidate) <= limit:
        return candidate
    return first[: max(0, limit - 3)].rstrip() + "..."
