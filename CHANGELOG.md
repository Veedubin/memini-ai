# Changelog

## 0.1.1

Fix `ensure_protocol()` falsely detecting the memory protocol block as already installed when
its marker text merely appeared in prose (e.g. in docs). Fix `.gitignore` patterns for
`.mcp.json` matching at any depth instead of just the repo root, and add coverage for OpenCode,
Kimi Code, and `.claude/settings.local.json` local state. Automate releases: pushing a version
bump to `main` now builds, cuts the GitHub release, and publishes to PyPI via Trusted Publishing.

## 0.1.0

First release. Three MCP tools (`remember`, `recall`, `orient`), Postgres + pgvector storage,
BGE-M3 embeddings, hybrid search, thought chains, Claude Code session ingest, `init` for
Claude Code, OpenCode, and Kimi Code, the `init-memini` skill, and `warm` to download and load
the model outside a tool call. MIT licensed.

`serve` no longer shows FastMCP's startup banner, which was making a PyPI network call on every
launch to check for a newer FastMCP version.
