"""Operator-side ingestion: JSONL import and transcript ingestion. Never imported by server.py."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from memini_ai.store import KINDS, Store, StoreError

log = structlog.get_logger(__name__)


async def import_jsonl(store: Store, path: Path, project: str | None) -> dict[str, int]:
    """Each line: {"text": str, "kind"?: str, "project"?: str, "tags"?: [str]}. Bad lines are skipped."""
    counts = {"imported": 0, "duplicates": 0, "skipped": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            counts["skipped"] += 1
            continue
        text = obj.get("text") if isinstance(obj, dict) else None
        if not isinstance(text, str) or not text.strip():
            counts["skipped"] += 1
            continue
        kind = obj.get("kind", "note")
        if kind not in KINDS or kind in ("thought", "session"):
            counts["skipped"] += 1
            continue
        try:
            r = await store.remember(
                text, kind=kind, project=obj.get("project") or project,
                tags=[str(t) for t in obj.get("tags", [])] or None,
            )
        except StoreError as e:
            log.warning("import_line_rejected", error=str(e))
            counts["skipped"] += 1
            continue
        counts["duplicates" if r["duplicate"] else "imported"] += 1
    return counts


async def ingest_claude_sessions(
    store: Store, root: Path | None, project: str | None, since: str | None
) -> dict[str, int]:
    raise NotImplementedError("implemented in Task 10")
