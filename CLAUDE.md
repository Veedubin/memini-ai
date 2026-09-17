# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Local-first memory for AI coding agents, shipped as an MCP server (`memini-ai serve`) plus a CLI.
Three tools: `remember`, `recall`, `orient`. Storage is Postgres + pgvector; embeddings are BGE-M3
(CPU by default) with hybrid vector + full-text search. No LLM in the loop, no background task, no
in-process index.

## Commands

Setup (uv-managed venv):

    uv sync --extra dev

Database, for both manual testing and running the test suite:

    memini-ai db up                                          # starts pgvector/pgvector:pg18 on 127.0.0.1:5555
    PGPASSWORD=memini psql -h 127.0.0.1 -p 5555 -U memini -d memini -c "CREATE DATABASE memini_test;"

The test suite hits a real Postgres at `MEMINI_TEST_DB_URL` (default
`postgresql://memini:memini@localhost:5555/memini_test`) — there is no DB mocking. Tests use the
`hash` embedder (`Settings(model="hash", ...)`), a deterministic bag-of-words fake; BGE-M3 is never
downloaded in the normal suite.

Lint, type-check, test (same commands CI runs):

    uv run ruff check src tests
    uv run mypy src
    uv run pytest
    uv run pytest tests/test_store_recall.py -k some_test    # single file / test

Real-model end-to-end test (downloads/loads BGE-M3, skipped by default):

    MEMINI_E2E=1 uv run pytest tests/test_e2e_real_model.py

Docs (built separately from the project env; no torch dependency):

    uvx --from mkdocs-material mkdocs build --strict

Other CLI entry points worth knowing while developing: `memini-ai migrate`, `memini-ai warm` (loads
the embedding model once, outside a tool-call timeout), `memini-ai reembed`, `memini-ai
ingest-sessions --client claude-code`, `memini-ai import <file.jsonl>`, `memini-ai init --client
<name>`.

## Architecture

    server.py  ->  store.py  ->  db.py (asyncpg, migrations, SQL)
                            ->  embed.py (BGE-M3 or hash)
    cli.py     ->  ingest.py, clients.py, and the same store

- **`db.py`** owns the asyncpg pool and all migration mechanics. Migrations are numbered SQL files
  in `migrations/`, applied in order under a Postgres advisory lock (so two servers racing on a
  fresh DB don't collide), and recorded in `schema_migrations`. `connect()` runs migrations first
  with a plain connection, then opens the pool with the pgvector codec registered.
- **`store.py`** is where all three tools' logic and SQL actually live (`server.py` is a thin
  FastMCP wrapper). One table, `memories`, holds every kind (`note`, `decision`, `handoff`, `fact`,
  `thought`, `session`). Thought chains are rows sharing a `chain_id` (tracked in a small `chains`
  table) with a `thought_number`; `superseded_by` is the only other relationship. `tsv` is a
  generated `tsvector` column (GIN index) for full-text; `embedding` is `vector(1024)` (HNSW index)
  for semantic search. `recall` runs both arms (top 20 each), fuses with reciprocal rank fusion
  (`search.py`), and reports `score` as relative to the top hit, not an absolute relevance measure.
- **`embed.py`** defines the `Embedder` protocol: `BgeM3Embedder` (real, lazy-loaded once per
  process behind a shared `asyncio.Task` so a cancelled/timed-out caller doesn't restart the load)
  and `HashEmbedder` (deterministic fake used in tests, selected via `MEMINI_MODEL=hash`). A failed
  model load degrades `remember`/`recall` to full-text only (`degraded: "text-only"` in the
  response) rather than failing the call.
- **`server.py`**'s `AppState` connects to the DB lazily on first tool call, behind a lock; a failed
  connection is *not* cached — it's retried on the next call. Every tool result funnels through a
  `run()` wrapper that converts `StoreError`/`DatabaseError`/timeout into a `{"error": ...}` object
  instead of raising — the MCP transport never sees a traceback, and stdout is reserved for the
  transport (all logging goes to stderr via structlog).
- **`config.py`**: every setting is `MEMINI_*` env var only (`pydantic-settings`, `env_prefix`);
  precedence is explicit overrides > process env > `MEMINI_CONFIG_FILE` (default
  `~/.config/memini-ai/config.env`) > field defaults. `MEMINI_PROJECT` is a default for `remember`
  only — `recall`/`orient` search across all projects unless `project` is passed explicitly.
- **`clients.py`**: `memini-ai init --client <claude-code|opencode|kimi-code|generic>` does a
  merge-only edit of the client's MCP config (with a timestamped backup), installs the
  `init-memini` skill where applicable, and appends a memory-protocol block to the project's
  instructions file wrapped in a pair of HTML-comment marker lines (see `clients.START`/`END`) —
  the block is appended once and never overwritten on re-run, unlike the skill file.
  `ensure_protocol()` detects "already added" with a plain substring search for the start marker,
  so never let that exact marker text appear elsewhere in an instructions file (e.g. in prose
  describing it) — it reads as a false positive and the block silently never gets appended.
- **`ingest.py`**: turns Claude Code session transcripts (`~/.claude/projects/<slug>/*.jsonl`, not
  nested subagent transcripts) into `kind="session"` memories, chunked to ~1500 chars, dropping
  thinking/tool-calls/tool-results and `isMeta` lines. Idempotent on re-run. Also handles curated
  JSONL import.

See `docs/architecture.md`, `docs/tools.md`, `docs/configuration.md`, `docs/clients.md`, and
`docs/session-ingest.md` for the user-facing detail behind each of these.

## Working in this repo

- `remember` returns `duplicate: true` and the *existing* row's `kind`/`project` (not the request's)
  when the same text (whitespace-normalized, hashed) already exists for that project — this is
  intentional idempotency, not a bug to route around.
- Heavy imports (`torch`, `sentence-transformers`, `asyncpg`, `db`, `store`) are deferred inside CLI
  subcommand functions, not imported at module top level, so `memini-ai --version` and `memini-ai db
  up` stay fast and don't require the DB or model dependencies to be importable.
- All SQL lives in `db.py` (pool/migrations) or `store.py` (queries) — nowhere else.

<!-- memini-ai:start -->
## Memory (memini-ai)
- Start every session with `orient`. Read it before planning.
- Before re-deriving anything about this project, `recall` it first.
- After a decision, a finished task, or a handoff, call `remember` with the matching `kind`.
  One paragraph, stating what and why. Include file paths and commit ids when relevant.
- When a fact changes, `remember` the new one with `supersedes=<old id>`.
- For multi-step reasoning you want to survive the session, use `kind="thought"` with `chain`.
- Never store secrets, tokens, or full file contents.
<!-- memini-ai:end -->
