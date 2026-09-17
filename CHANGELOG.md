# Changelog

## 0.1.0

First release. Three MCP tools (`remember`, `recall`, `orient`), Postgres + pgvector storage,
BGE-M3 embeddings, hybrid search, thought chains, Claude Code session ingest, `init` for
Claude Code, OpenCode, and Kimi Code, the `init-memini` skill, and `warm` to download and load
the model outside a tool call. MIT licensed.

`serve` no longer shows FastMCP's startup banner, which was making a PyPI network call on every
launch to check for a newer FastMCP version.
