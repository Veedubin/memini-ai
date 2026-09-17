"""FastMCP application exposing remember, recall, orient."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Annotated, Any, Literal

import structlog
from fastmcp import FastMCP
from pydantic import Field, ValidationError

from memini_ai.config import Settings, load_settings
from memini_ai.db import Database, DatabaseError
from memini_ai.embed import make_embedder
from memini_ai.store import ChainStep, Store, StoreError

log = structlog.get_logger(__name__)

Kind = Literal["note", "decision", "handoff", "fact", "thought"]
RecallKind = Literal["note", "decision", "handoff", "fact", "thought", "session"]


def configure_logging(level: str) -> None:
    """Everything to stderr. stdout belongs to the MCP transport."""
    logging.basicConfig(stream=sys.stderr, level=level.upper(), format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.KeyValueRenderer(key_order=["event"]),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
    )


class AppState:
    """Lazily connects on first use; a failed attempt is retried on the next call."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._db: Database | None = None
        self._store: Store | None = None
        self._lock = asyncio.Lock()
        self.last_error: str | None = None

    async def get_store(self) -> Store:
        if self._store is not None:
            return self._store
        async with self._lock:
            if self._store is None:
                db = Database(self.settings.db_url)
                try:
                    await db.connect()
                except DatabaseError as e:
                    self.last_error = str(e)
                    log.error("db_connect_failed", error=str(e))
                    raise
                self._db = db
                self._store = Store(db, make_embedder(self.settings), self.settings)
                self.last_error = None
                log.info("store_ready")
        return self._store

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
        self._db, self._store = None, None


def create_app(settings: Settings | None = None) -> FastMCP:
    settings = settings or load_settings()
    state = AppState(settings)
    app = FastMCP("memini-ai")
    app.memini_state = state  # type: ignore[attr-defined]

    async def run(coro: Any) -> dict[str, Any]:
        try:
            result: dict[str, Any] = await asyncio.wait_for(coro, timeout=settings.timeout_s)
            return result
        except StoreError as e:
            return {"error": str(e)}
        except DatabaseError as e:
            return {"error": f"database unavailable: {e}"}
        except TimeoutError:
            return {"error": f"timeout after {settings.timeout_s:g}s"}
        except Exception as e:  # last resort; never let the transport see a traceback
            log.exception("tool_failed")
            return {"error": f"{type(e).__name__}: {e}"}

    @app.tool
    async def remember(
        text: Annotated[str, Field(description="One paragraph: what and why. Include paths and commit ids.")],
        kind: Annotated[Kind, Field(description="note | decision | handoff | fact | thought")] = "note",
        project: Annotated[str | None, Field(description="Defaults to the server's MEMINI_PROJECT.")] = None,
        tags: Annotated[list[str] | None, Field(description="Short lowercase labels.")] = None,
        supersedes: Annotated[str | None, Field(description="Id of the memory this replaces.")] = None,
        chain: Annotated[
            dict[str, Any] | None,
            Field(description="Only with kind='thought'. Keys: number, total, next_needed, optional "
                              "chain_id (omit on the first thought), revises, branch_from."),
        ] = None,
    ) -> dict[str, Any]:
        """Store a memory. Call after a decision, a finished task, a handoff, or a learned fact."""

        async def go() -> dict[str, Any]:
            store = await state.get_store()
            step: ChainStep | None = None
            if chain is not None:
                try:
                    step = ChainStep.model_validate(chain)
                except ValidationError as e:
                    raise StoreError(f"invalid chain: {e}") from e
            return await store.remember(text, kind=kind, project=project, tags=tags,
                                        supersedes=supersedes, chain=step)

        return await run(go())

    @app.tool
    async def recall(
        query: Annotated[str, Field(description="Natural-language question or keywords.")],
        limit: Annotated[int, Field(ge=1, le=50, description="Max results.")] = 8,
        kind: Annotated[RecallKind | None, Field(description="Filter by kind; 'session' searches ingested transcripts.")] = None,
        project: Annotated[str | None, Field(description="Filter by project.")] = None,
        since: Annotated[str | None, Field(description="'7d', '24h', or an ISO date.")] = None,
        include_superseded: Annotated[bool, Field(description="Also return replaced memories.")] = False,
        chain_id: Annotated[str | None, Field(description="Return this thought chain in order; query is ignored.")] = None,
    ) -> dict[str, Any]:
        """Search memories before re-deriving anything. Hybrid semantic + full-text."""

        async def go() -> dict[str, Any]:
            store = await state.get_store()
            return await store.recall(query, limit=limit, kind=kind, project=project, since=since,
                                      include_superseded=include_superseded, chain_id=chain_id)

        return await run(go())

    @app.tool
    async def orient(
        project: Annotated[str | None, Field(description="Omit for all projects; pass a project name to scope.")] = None,
        budget: Annotated[int, Field(ge=20, le=4000, description="Approximate token budget for `text`.")] = 300,
    ) -> dict[str, Any]:
        """Call first in every session: recent decisions, handoffs, open chains, and server health."""

        async def go() -> dict[str, Any]:
            try:
                store = await state.get_store()
            except DatabaseError as e:
                return {"status": {"db": "error", "model": settings.model, "memories": 0, "error": str(e)},
                        "decisions": [], "handoffs": [], "open_chains": 0, "projects": [],
                        "text": f"memini-ai unavailable: {e}"}
            return await store.orient(project=project, budget=budget)

        return await run(go())

    return app
