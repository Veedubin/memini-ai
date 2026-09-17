# memini-ai

Local-first memory for AI coding agents. Three MCP tools: `remember`, `recall`, `orient`.
Postgres + pgvector for storage, BGE-M3 embeddings on CPU, hybrid vector + full-text search.

## Quick start

    uv tool install memini-ai
    memini-ai db up            # pgvector/pgvector:pg18 on 127.0.0.1:5555
    memini-ai migrate          # optional; serve applies migrations on first connection

Then, from the project you want memory for:

    cd ~/Projects/my-repo
    memini-ai init --client claude-code --project my-repo

Restart the client and call `orient`.

## Docs

- [Getting started](docs/getting-started.md)
- [The three tools](docs/tools.md)
- [Configuration](docs/configuration.md)
- [Clients and the init-memini skill](docs/clients.md)
- [Session ingest](docs/session-ingest.md)
- [Architecture](docs/architecture.md)
- [Changelog](docs/changelog.md)
