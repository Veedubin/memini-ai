# Session ingest

Answering "why is this like this" often means reading an old session. `memini-ai ingest-sessions`
turns transcripts into searchable memories.

    memini-ai ingest-sessions --client claude-code --since 30d

Claude Code transcripts live under `~/.claude/projects/<slug>/<session>.jsonl`. The ingester keeps
user prompts and assistant prose, drops thinking, tool calls, tool results and attachments, packs
messages into chunks of about 1500 characters, and stores each as `kind=session` with the client,
session id, transcript directory name (`project_slug`), and timestamp in `source`. Lines marked
`isMeta` — the boilerplate Claude Code injects around local commands and hooks — are skipped, and
only `<slug>/*.jsonl` is read: nested transcripts such as `subagents/*.jsonl` are not ingested. The
project label comes from the transcript directory name unless `--project` is given. Re-running is
idempotent. A transcript file that can't be read or parsed is skipped with a warning, and ingestion
continues with the rest.

Search them with `recall(query, kind="session")`. Nothing else returns session chunks by default,
because `kind` filters are exact; a plain `recall` searches every kind.

OpenCode and Kimi Code transcripts are not yet supported.

## Curated import

    memini-ai import notes.jsonl --project my-repo

Each line is `{"text": ..., "kind"?: ..., "project"?: ..., "tags"?: [...]}`. Use this to bring
selected memories over from an older store.
