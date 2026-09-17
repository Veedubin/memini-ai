# memini-ai design

Date: 2026-09-17. Status: approved by the owner in conversation; this file is the written record.

memini-ai is a local-first memory server for AI coding agents, exposed over MCP. It is a
fresh application, not a version of `memini-ai-dev`. It replaces a 50-tool, 28k-line
server whose review on 2026-09-17 found that about seven tools carried value and most
subsystems silently persisted nothing. The old project and its database on port 5434
are left untouched.

## Goals

- Three MCP tools whose combined schema is under 700 tokens.
- Retrieval that works: hybrid vector plus full-text search with real filters.
- Configuration that is deterministic: no cwd-relative files, one env prefix, no aliases.
- A setup skill so the agent learns when and how to use memory, not just that tools exist.
- A code base one person can read in an afternoon: target 3000 lines of source.

## Non-goals

- Backward compatibility with memini-ai-dev tools, env vars, or schema.
- Trust scoring, decay, knowledge graph, dialectic, multi-peer, audit tools, kanban,
  project file indexing, tiered LLM summaries, image search, dual-model fusion.
- Any LLM call inside the server. All summarization is done by the calling agent.

## 1. Tool surface

All tools are snake_case. Every parameter is declared with `Annotated[..., Field(description=...)]`
so the emitted JSON schema carries descriptions. Docstrings are short and say when to call.

### remember

```
remember(
  text: str,
  kind: Literal["note","decision","handoff","fact","thought","session"] = "note",
  project: str | None = None,
  tags: list[str] | None = None,
  supersedes: str | None = None,
  chain: ChainStep | None = None,
) -> {id, kind, project, duplicate: bool, superseded_id?, chain_id?}
```

- `kind` is the entire taxonomy. `session` is reserved for ingested transcripts and is
  refused from the tool.
- `project` defaults to `MEMINI_PROJECT` from the server environment when omitted.
- `supersedes` marks the named memory as replaced by this one. Superseded rows are hidden
  from recall unless asked for.
- `chain` is `{chain_id?: str, number: int, total: int, next_needed: bool, revises?: int, branch_from?: int}`
  and is only valid with `kind="thought"`. Omitting `chain_id` on the first thought creates a
  chain and returns its id. This keeps the sequential-thinking contract of the old `add_thought`.
- Text limit 32 KB. Duplicate text within the same project returns the existing id with
  `duplicate: true`, never an error.
- Duplicate detection is `sha256` of whitespace-normalized text.

### recall

```
recall(
  query: str,
  limit: int = 8,
  kind: Kind | None = None,
  project: str | None = None,
  since: str | None = None,          # ISO date or "7d", "24h"
  include_superseded: bool = False,
  chain_id: str | None = None,
) -> {results: [{id, text, kind, project, tags, created_at, score, superseded_by?, chain?}]}
```

- Hybrid: pgvector cosine top-20 and Postgres full-text top-20 (`websearch_to_tsquery`), fused
  with reciprocal rank fusion, k=60, in Python. Both arms apply the same SQL filters.
- `score` is the fused score divided by the maximum in the result set, so the top hit is 1.0.
- `chain_id` set returns that chain's thoughts in order and ignores `query`.
- Returned ids get `retrieval_count = retrieval_count + 1` in one statement. Nothing exposes
  that column; it exists for future ranking and for `orient`.
- Never returns vectors, raw metadata, or HTML.

### orient

```
orient(project: str | None = None, budget: int = 300) -> {
  status: {db: "ok"|"error", model: str, memories: int, error?: str},
  decisions: [...last 5], handoffs: [...last 3], open_chains: int,
  projects: [{project, last_note_at}], text: str   # rendered block under `budget` tokens
}
```

- No LLM. Pure SQL plus string trimming. Intended as the first call of a session.
- Doubles as the health probe. It never writes.

## 2. Data model

Postgres 18 with the pgvector extension, run as the `memini-db` container from
`pgvector/pgvector:pg18` on host port 5555. `compose.yaml` in the repo defines it;
`memini-ai db up|down|status` wraps `podman compose` (falls back to `docker compose`).
Database, user, and default password are all `memini`.

```sql
CREATE TABLE chains (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project text,
  status text NOT NULL DEFAULT 'open' CHECK (status IN ('open','done','abandoned')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE memories (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  text text NOT NULL,
  kind text NOT NULL CHECK (kind IN ('note','decision','handoff','fact','thought','session')),
  project text,
  tags text[] NOT NULL DEFAULT '{}',
  content_hash text NOT NULL,
  embedding vector(1024),
  embedding_model text,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  superseded_by uuid REFERENCES memories(id),
  chain_id uuid REFERENCES chains(id) ON DELETE CASCADE,
  thought_number int, thought_total int, next_needed boolean,
  revises int, branch_from int,
  source jsonb NOT NULL DEFAULT '{}',   -- ingest provenance only: {client, session_id, ts}
  retrieval_count int NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX memories_dedup ON memories (coalesce(project,''), content_hash);
CREATE INDEX memories_embedding ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX memories_tsv ON memories USING gin (tsv);
CREATE INDEX memories_project_created ON memories (project, created_at DESC);
CREATE INDEX memories_chain ON memories (chain_id, thought_number);

CREATE TABLE schema_migrations (version int PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now());
```

Migrations are numbered SQL files packaged with the code. `memini-ai migrate` and server
startup apply pending ones inside a transaction, in order. There is no `CREATE TABLE IF NOT EXISTS`
outside migration 0001.

### Embeddings

One model: `BAAI/bge-m3`, 1024 dimensions, via sentence-transformers. CPU by default;
`MEMINI_DEVICE=cuda` opts into the GPU. Loaded lazily on first use inside `asyncio.to_thread`
behind an `asyncio.Lock`, then kept for the process lifetime. `embedding_model` is recorded per
row. `memini-ai reembed` re-embeds rows whose model differs from the configured one.
`MEMINI_MODEL=hash` selects a deterministic fake embedder for tests.

## 3. Configuration

pydantic-settings with `env_prefix="MEMINI_"` and no field aliases. The settings file is
explicit: `MEMINI_CONFIG_FILE`, default `~/.config/memini-ai/config.env`, loaded only if it
exists. Process environment wins over the file. Nothing reads the current working directory.

| Variable | Default | Meaning |
|---|---|---|
| `MEMINI_DB_URL` | `postgresql://memini:memini@localhost:5555/memini` | asyncpg DSN |
| `MEMINI_MODEL` | `BAAI/bge-m3` | embedding model, or `hash` for tests |
| `MEMINI_DEVICE` | `cpu` | `cpu` or `cuda` |
| `MEMINI_PROJECT` | unset | default project for `remember` only; `recall` and `orient` search all projects unless `project` is given |
| `MEMINI_CONFIG_FILE` | `~/.config/memini-ai/config.env` | settings file |
| `MEMINI_LOG_LEVEL` | `INFO` | stderr log level |
| `MEMINI_TIMEOUT_S` | `30` | per-tool timeout |
| `MEMINI_MAX_TEXT_BYTES` | `32768` | `remember` limit |

Logs go to stderr only. The `serve` path never writes to stdout.

## 4. CLI

`memini-ai` with subcommands:

- `serve` runs the MCP server over stdio. This is what client configs invoke.
- `db up | down | status` manages the container via compose.
- `migrate` applies pending migrations.
- `init --client {claude-code,opencode,kimi-code,generic} [--scope user|project] [--project NAME]`
  writes the MCP entry into the client's config with a merge-only edit and a timestamped backup,
  installs the `init-memini` skill where that client looks for skills (overwriting an existing
  `SKILL.md` with the packaged one), and appends the memory protocol block to the client's
  instructions file if the marker is absent. Paths are relative to the current directory.
- `warm` loads the embedding model and embeds one string, so the download happens once in the
  foreground rather than inside a client's first tool call.
- `import FILE.jsonl [--project]` imports `{text, kind, project?, tags?, created_at?}` rows,
  for curating data out of the old database.
- `ingest-sessions --client claude-code [--project] [--since 30d]` chunks transcripts into
  `kind=session` memories.
- `reembed` as above.

Client adapter table (verified on this machine unless marked):

| Client | Config path | Key shape | Skill dir | Instructions file |
|---|---|---|---|---|
| claude-code | `~/.claude.json` (user) or `./.mcp.json` (project) | `mcpServers.<name>: {type: stdio, command, args, env}` | `~/.claude/skills/<name>/SKILL.md` | `CLAUDE.md` |
| opencode | `~/.config/opencode/opencode.json` or `./.opencode/opencode.json` | `mcp.<name>: {type: local, command: [..], environment, enabled}` | `.opencode/skills/<name>/SKILL.md` (best effort) | `AGENTS.md` |
| kimi-code | `./.kimi-code/mcp.json` (always; `--scope` is ignored) | `mcpServers.<name>: {command, args, env, enabled}` | not installed, not printed | `AGENTS.md` |
| generic | prints a `mcpServers` JSON snippet | | not installed, not printed | `AGENTS.md`, written directly |

Cursor, Codex, Pi, Hermes and Gemini CLI use `generic` until their formats are verified.

## 5. The init-memini skill and the memory protocol

The skill is a `SKILL.md` shipped in the package. It tells the agent to:

1. Run `memini-ai db status`, and `memini-ai db up` if needed.
2. Run `memini-ai init --client <detected client> --project <repo name>`.
3. Call `orient` and confirm `status.db == "ok"`.
4. Confirm the instructions file now contains the protocol block; show the user the diff.

The protocol block appended to `CLAUDE.md` or `AGENTS.md` is about 150 tokens:

```
## Memory (memini-ai)
- Start every session with `orient`. Read it before planning.
- Before re-deriving anything about this project, `recall` it first.
- After a decision, a finished task, or a handoff, call `remember` with the matching `kind`.
  One paragraph, stating what and why. Include file paths and commit ids when relevant.
- When a fact changes, `remember` the new one with `supersedes=<old id>`.
- For multi-step reasoning you want to survive the session, use `kind="thought"` with `chain`.
- Never store secrets, tokens, or full file contents.
```

## 6. Session ingest

Claude Code transcripts live at `~/.claude/projects/<slug>/<session>.jsonl`. Each line is a
JSON object with a `type`; `user` and `assistant` lines carry a `message` with text content
blocks. The ingester keeps only human-readable text from those two types, drops tool results and
attachments, joins consecutive turns, splits into chunks of about 1500 characters at message
boundaries with one message of overlap, and stores each chunk as `kind=session` with
`source = {client: "claude-code", session_id, project_slug, ts}`. Re-running is idempotent
because chunks are content-hashed. OpenCode (`opencode.db` SQLite) and Kimi Code
(`~/.kimi-code/sessions`) ingesters are follow-on work once their formats are read.

## 7. Package layout

```
memini-ai/
  pyproject.toml            # name memini-ai, version 0.1.0, requires-python >=3.12
  compose.yaml
  src/memini_ai/
    __init__.py
    config.py               # Settings
    db.py                   # pool, migration runner, all SQL
    migrations/0001_init.sql
    embed.py                # Embedder protocol, BgeM3Embedder, HashEmbedder
    store.py                # remember/recall/orient/chains over db + embed
    search.py               # RRF and query building
    server.py               # FastMCP app with the three tools
    ingest.py               # transcript chunking, JSONL import
    clients.py              # adapter table, merge-only JSON edits, protocol block
    cli.py                  # argparse entry
    skill/init-memini/SKILL.md
  tests/                    # pytest against the real container, hash embedder
  docs/                     # mkdocs material site
  .github/workflows/ci.yml  # ruff, mypy, pytest with a postgres service
```

Dependencies: `fastmcp`, `asyncpg`, `pgvector`, `pydantic-settings`, `sentence-transformers`,
`torch` from the CPU index by default, `structlog`. Dev: `pytest`, `pytest-asyncio`, `ruff`, `mypy`.

## 8. Error handling

- Every tool returns a dict. Failures return `{error: str}` and nothing else; successes never
  carry an `error` key. Timeouts return `{error: "timeout after 30s"}`.
- Startup failures (DB unreachable, migration failure) are logged and surfaced through
  `orient.status`; the server keeps running so the agent can report the problem.
- Model load failure is surfaced the same way; `recall` falls back to full-text only and says
  so in the response with `degraded: "text-only"`.
- Lazy initialization is guarded by one lock and is retried on the next call if it failed.

## 9. Testing

- `tests/conftest.py` starts from `MEMINI_DB_URL` pointing at a scratch database on the 5555
  container (`memini_test`), applies migrations, truncates between tests, and uses the hash embedder.
- Integration tests cover: remember dedup and supersedes; recall filters, hybrid ranking,
  superseded hiding, chain ordering; orient budget trimming and status; migration idempotency;
  thought chain creation, revise, branch.
- Unit tests cover: chunker, RRF, config precedence, client adapter merges on temp files,
  protocol block idempotency, transcript parsing on a fixture.
- CI runs ruff, mypy strict, and pytest with a `pgvector/pgvector:pg18` service.

## 10. Docs and release

- mkdocs Material site: index, getting started, tools, configuration, clients, session ingest,
  architecture, changelog. `site_url` will be set when the GitHub repo exists.
- The owner uses mkdocs output for the frozenmaplelabs site; the old site under memini-ai-dev
  stays as is until the owner points the site at the new one.
- Release: tag `v0.1.0`, GitHub Actions publishes to PyPI on tag via trusted publishing once the
  owner creates the repo and the PyPI project. Neither is done by the agent without being asked.
