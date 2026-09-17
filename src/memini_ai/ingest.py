"""Operator-side ingestion: JSONL import and transcript ingestion. Never imported by server.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from memini_ai.search import parse_since
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


@dataclass
class Turn:
    role: str
    text: str
    ts: str | None


DEFAULT_CLAUDE_ROOT = Path("~/.claude/projects")


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(p.strip() for p in parts if p and p.strip())
    return ""


def parse_claude_transcript(path: Path) -> list[Turn]:
    """Keep user prompts and assistant prose; drop thinking, tool calls, tool results, attachments."""
    turns: list[Turn] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("type") not in ("user", "assistant"):
            continue
        text = _text_of((obj.get("message") or {}).get("content"))
        if not text:
            continue
        turns.append(Turn(str(obj["type"]), text, obj.get("timestamp")))
    return turns


def chunk_turns(turns: list[Turn], max_chars: int = 1500) -> list[str]:
    """Greedy packing of 'role: text' messages with one-message overlap between chunks."""
    msgs: list[str] = []
    for t in turns:
        body = f"{t.role}: {t.text}"
        while len(body) > max_chars + len(t.role) + 2:
            msgs.append(body[:max_chars])
            body = f"{t.role}: " + body[max_chars:]
        msgs.append(body)
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for m in msgs:
        if cur and size + len(m) + 1 > max_chars:
            chunks.append("\n".join(cur))
            cur, size = [cur[-1]], len(cur[-1])
            if size + len(m) + 1 > max_chars:
                cur, size = [], 0
        cur.append(m)
        size += len(m) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def project_from_slug(slug: str) -> str:
    return slug.rsplit("-Projects-", 1)[-1] if "-Projects-" in slug else slug.strip("-").rsplit("-", 1)[-1]


async def ingest_claude_sessions(
    store: Store, root: Path | None, project: str | None, since: str | None
) -> dict[str, int]:
    root = (root or DEFAULT_CLAUDE_ROOT).expanduser()
    cutoff = parse_since(since).timestamp() if since else None
    counts = {"sessions": 0, "chunks": 0, "duplicates": 0}
    for path in sorted(root.glob("*/*.jsonl")):
        if cutoff is not None and path.stat().st_mtime < cutoff:
            continue
        turns = parse_claude_transcript(path)
        if not turns:
            continue
        counts["sessions"] += 1
        proj = project or project_from_slug(path.parent.name)
        session_id = path.stem
        for i, chunk in enumerate(chunk_turns(turns)):
            r = await store.remember(
                chunk, kind="session", project=proj, allow_session=True,
                source={"client": "claude-code", "session_id": session_id, "chunk": i,
                        "ts": turns[0].ts, "ingested_at": datetime.now(UTC).isoformat()},
            )
            counts["duplicates" if r["duplicate"] else "chunks"] += 1
        log.info("session_ingested", path=str(path), project=proj)
    return counts
