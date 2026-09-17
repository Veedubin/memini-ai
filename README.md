# memini-ai

Local-first memory for AI coding agents. Three MCP tools: `remember`, `recall`, `orient`.
Postgres + pgvector for storage, BGE-M3 embeddings on CPU, hybrid vector + full-text search.

## Quick start

    uv tool install memini-ai
    memini-ai db up          # starts a pgvector container on localhost:5555
    memini-ai init --client claude-code --project my-repo

Then start a session and call `orient`.

See `docs/` for everything else.
