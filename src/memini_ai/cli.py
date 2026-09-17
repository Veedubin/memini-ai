"""Command-line entry point. `memini-ai serve` is what MCP clients run."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from memini_ai import __version__
from memini_ai.config import Settings, load_settings

if TYPE_CHECKING:  # heavy imports stay out of `memini-ai db up` and `--version`
    from memini_ai.db import Database
    from memini_ai.store import Store

# compose derives the project name from the compose file's directory, which for an installed
# wheel is site-packages/memini_ai: name it explicitly so `db status` sees the same containers
# `db up` created, wherever memini-ai is installed from.
COMPOSE_PROJECT = "memini-ai"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="memini-ai", description="Local-first memory for AI coding agents.")
    p.add_argument("--version", action="version", version=f"memini-ai {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="Run the MCP server over stdio.")
    sub.add_parser("migrate", help="Apply pending database migrations.")
    sub.add_parser("reembed", help="Re-embed rows whose embedding_model differs from MEMINI_MODEL.")
    sub.add_parser("warm", help="Load the embedding model (downloads it on first run) and embed once.")

    db = sub.add_parser("db", help="Manage the pgvector container.")
    db.add_argument("action", choices=["up", "down", "status"])

    imp = sub.add_parser("import", help="Import memories from a JSONL file.")
    imp.add_argument("file", type=Path)
    imp.add_argument("--project", default=None)

    ing = sub.add_parser("ingest-sessions", help="Ingest agent session transcripts as kind=session.")
    ing.add_argument("--client", choices=["claude-code"], required=True)
    ing.add_argument("--project", default=None, help="Project label; defaults to the transcript's repo name.")
    ing.add_argument("--since", default=None, help="Only sessions modified after this ('30d', ISO date).")
    ing.add_argument("--root", type=Path, default=None, help="Override the transcript root directory.")

    init = sub.add_parser("init", help="Configure an agent client to use memini-ai.")
    init.add_argument("--client", choices=["claude-code", "opencode", "kimi-code", "generic"], required=True)
    init.add_argument("--scope", choices=["user", "project"], default="project")
    init.add_argument("--project", default=None, help="Project label written as MEMINI_PROJECT.")
    init.add_argument("--command", default=None,
                      help="Override the server command, e.g. 'uv run --directory /path memini-ai serve'.")
    init.add_argument("--cwd", type=Path, default=Path.cwd())
    return p


def _compose_cmd() -> list[str]:
    for exe, extra in (("podman", ["compose"]), ("docker", ["compose"]), ("podman-compose", [])):
        if shutil.which(exe):
            return [exe, *extra]
    raise SystemExit("neither podman nor docker found on PATH")


def _compose_file() -> Path:
    # resources.as_file() would normally need to keep its context open for the returned path to
    # stay valid (e.g. if the package were a zipped wheel needing extraction to a temp file), but
    # for a normal on-disk install (the only supported install method here) the package directory
    # already exists on disk, so the yielded path is the real file and remains valid after the
    # `with` block exits.
    with resources.as_file(resources.files("memini_ai") / "compose.yaml") as p:
        return p


def cmd_db(action: str) -> int:
    compose = [*_compose_cmd(), "-p", COMPOSE_PROJECT, "-f", str(_compose_file())]
    if action == "up":
        return subprocess.call([*compose, "up", "-d"])
    if action == "down":
        return subprocess.call([*compose, "down"])
    return subprocess.call([*compose, "ps"])


async def _store(settings: Settings) -> tuple[Database, Store]:
    from memini_ai.db import Database
    from memini_ai.embed import make_embedder
    from memini_ai.store import Store

    db = Database(settings.db_url)
    await db.connect()
    return db, Store(db, make_embedder(settings), settings)


async def cmd_warm(settings: Settings) -> int:
    """Pay the model download and load cost once, outside a client's tool timeout."""
    from memini_ai.embed import EmbedError, make_embedder

    embedder = make_embedder(settings)
    try:
        vector = (await embedder.embed(["memini-ai warm-up"]))[0]
    except EmbedError as e:
        print(f"model {embedder.name} failed to load: {e}", file=sys.stderr)
        return 1
    print(f"model {embedder.name} ready, dim {len(vector)}")
    return 0


async def cmd_migrate(settings: Settings) -> int:
    from memini_ai.db import Database

    applied = await Database(settings.db_url).migrate()
    print(f"applied migrations: {applied or 'none'}")
    return 0


async def cmd_import(settings: Settings, file: Path, project: str | None) -> int:
    from memini_ai.ingest import import_jsonl

    db, store = await _store(settings)
    try:
        counts = await import_jsonl(store, file, project)
    finally:
        await db.close()
    print(json.dumps(counts))
    return 0


async def cmd_reembed(settings: Settings) -> int:
    db, store = await _store(settings)
    embedder = store.embedder  # the one the store already built; loading a second is wasteful
    try:
        rows = await db.fetch(
            "SELECT id, text FROM memories WHERE embedding_model IS DISTINCT FROM $1 ORDER BY created_at",
            embedder.name,
        )
        for i in range(0, len(rows), 32):
            batch = rows[i : i + 32]
            vecs = await embedder.embed([r["text"] for r in batch])
            for r, v in zip(batch, vecs, strict=True):
                await db.execute("UPDATE memories SET embedding=$1, embedding_model=$2 WHERE id=$3",
                                 v, embedder.name, r["id"])
            print(f"re-embedded {min(i + 32, len(rows))}/{len(rows)}", file=sys.stderr)
    finally:
        await db.close()
    print(f"re-embedded {len(rows)} rows with {embedder.name}")
    return 0


async def cmd_ingest(settings: Settings, args: argparse.Namespace) -> int:
    from memini_ai.ingest import ingest_claude_sessions

    db, store = await _store(settings)
    try:
        counts = await ingest_claude_sessions(store, root=args.root, project=args.project, since=args.since)
    finally:
        await db.close()
    print(json.dumps(counts))
    return 0


def cmd_serve(settings: Settings) -> int:
    from memini_ai.server import create_app

    app = create_app(settings)
    # The banner triggers a PyPI version check on every launch (network call on stderr, but
    # still unwanted on every process start); show_banner=False skips both.
    app.run(transport="stdio", show_banner=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "db":
        return cmd_db(args.action)  # no settings, no heavy imports: just talk to compose
    from memini_ai.server import configure_logging

    settings = load_settings()
    configure_logging(settings.log_level)  # every subcommand logs to stderr, never stdout
    if args.cmd == "serve":
        return cmd_serve(settings)
    if args.cmd == "migrate":
        return asyncio.run(cmd_migrate(settings))
    if args.cmd == "import":
        return asyncio.run(cmd_import(settings, args.file, args.project))
    if args.cmd == "reembed":
        return asyncio.run(cmd_reembed(settings))
    if args.cmd == "warm":
        return asyncio.run(cmd_warm(settings))
    if args.cmd == "ingest-sessions":
        return asyncio.run(cmd_ingest(settings, args))
    if args.cmd == "init":
        from memini_ai.clients import run_init

        return run_init(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
