# memini-ai

Local-first memory for AI coding agents. Three MCP tools: `remember`, `recall`, `orient`.
Postgres + pgvector for storage, BGE-M3 embeddings on CPU, hybrid vector + full-text search.

## Quick start

    uv tool install memini-ai --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    memini-ai db up            # pgvector/pgvector:pg18 on 127.0.0.1:5555
    memini-ai migrate          # optional; serve applies migrations on first connection
    memini-ai warm             # one-time ~2.2 GB BGE-M3 download, then embeds one string

The `--index` flags install CPU torch instead of the default CUDA build. Run `warm` before wiring
up a client: the first `remember` or `recall` would otherwise download BGE-M3 (about 2.2 GB) inside
a tool call and time out until it finishes.

Then, from the project you want memory for:

    cd ~/Projects/my-repo
    memini-ai init --client claude-code --project my-repo

Restart the client and call `orient`.

## Docs

- [Getting started](https://github.com/Veedubin/memini-ai/blob/main/docs/getting-started.md)
- [The three tools](https://github.com/Veedubin/memini-ai/blob/main/docs/tools.md)
- [Configuration](https://github.com/Veedubin/memini-ai/blob/main/docs/configuration.md)
- [Clients and the init-memini skill](https://github.com/Veedubin/memini-ai/blob/main/docs/clients.md)
- [Session ingest](https://github.com/Veedubin/memini-ai/blob/main/docs/session-ingest.md)
- [Architecture](https://github.com/Veedubin/memini-ai/blob/main/docs/architecture.md)
- [Changelog](https://github.com/Veedubin/memini-ai/blob/main/CHANGELOG.md)

## License

MIT. See [LICENSE](https://github.com/Veedubin/memini-ai/blob/main/LICENSE).
