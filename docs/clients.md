# Clients and the init-memini skill

`memini-ai init --client <name>` does three things: writes one MCP server entry into the client's
config with a merge-only edit and a timestamped backup, installs the `init-memini` skill where the
client looks for skills, and appends a short memory protocol to the project's instructions file
between `<!-- memini-ai:start -->` and `<!-- memini-ai:end -->` markers.

| Client | Config written | Skill location | Instructions file |
|---|---|---|---|
| `claude-code` | `.mcp.json` (project) or `~/.claude.json` (user) | `~/.claude/skills/init-memini/` | `CLAUDE.md` |
| `opencode` | `.opencode/opencode.json` or `~/.config/opencode/opencode.json` | `.opencode/skills/init-memini/` | `AGENTS.md` |
| `kimi-code` | `.kimi-code/mcp.json` | not installed | `AGENTS.md` |
| `generic` | prints a `mcpServers` snippet | not installed | `AGENTS.md` |

Cursor, Codex, Pi, Hermes, and Gemini CLI: use `generic` and paste the snippet into that client's
MCP config.

## The protocol block

The block is the part that makes memory useful. It tells the agent to call `orient` first, `recall`
before re-deriving, and `remember` after decisions and handoffs, with `supersedes` when a fact
changes. It is about 150 tokens and lives in the instructions file the client already loads.

## The skill

`/init-memini` (or the client's equivalent) walks the agent through setup and repair: start the
database, run `init`, reload, call `orient`, confirm `status.db == "ok"`.
