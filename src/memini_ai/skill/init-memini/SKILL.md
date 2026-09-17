---
name: init-memini
description: Set up or repair memini-ai memory for this project and this client. Use when memory tools are missing, orient() fails, or the user asks to enable memini.
---

# init-memini

memini-ai gives you three tools: `orient`, `recall`, `remember`. This skill wires them into the
current client and teaches the project's instructions file when to use them.

## Steps

1. Detect the client you are running in (Claude Code, OpenCode, Kimi Code, or other).
2. Run `memini-ai db status`. If the container is not running, run `memini-ai db up` and wait
   until `pg_isready` reports ready (about 5 seconds). Then run `memini-ai warm` once: the first
   run downloads the BGE-M3 model (about 2.2 GB) and would otherwise time out inside a tool call.
3. Run `memini-ai init --client <client> --project <repo directory name>`.
   - Add `--scope user` if the user wants memory in every project rather than this one.
   - Add `--command "uv run --directory <checkout> memini-ai serve"` when running from a source checkout.
   - From a source checkout, invoke init as `uv run --project <checkout> memini-ai init ...`:
     `--project` keeps the current directory, while `--directory` would write the config and
     `CLAUDE.md`/`AGENTS.md` into the checkout instead of this repository. Keep `--directory` in
     the `--command` above; the server does not care about the current directory.
4. The client must reload MCP servers. Tell the user how for their client (Claude Code: `/mcp`
   or restart; OpenCode and Kimi Code: restart the TUI).
5. Call `orient`. Confirm `status.db == "ok"` and report `status.memories`.
6. Show the user the block that was appended to `CLAUDE.md` or `AGENTS.md`. Do not edit it.

## If something fails

- `orient` returns `status.db == "error"`: run `memini-ai db status` and `memini-ai migrate`, then retry.
- The tools are not listed: the client did not reload; ask the user to restart it.
- `memini-ai` is not on PATH: `uv tool install memini-ai --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match`.
- `orient` reports `status.model_error`, or results carry `degraded: "text-only"`: the model did
  not load. Run `memini-ai warm` and read its error.
