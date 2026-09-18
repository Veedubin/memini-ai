# Clients and the init-memini skill

`memini-ai init --client <name>` does three things: writes one MCP server entry into the client's
config with a merge-only edit and a timestamped backup, installs the `init-memini` skill where the
client looks for skills, and appends a short memory protocol to the project's instructions file
between `<!-- memini-ai:start -->` and `<!-- memini-ai:end -->` markers. It writes into the current
directory, so run it from the repository you want memory for.

| Client | Config written | Skill location | Instructions file |
|---|---|---|---|
| `claude-code` | `.mcp.json` (project) or `~/.claude.json` (user) | `~/.claude/skills/init-memini/` | `CLAUDE.md` |
| `opencode` | `.opencode/opencode.json` or `~/.config/opencode/opencode.json` | `.opencode/skills/init-memini/` (project) or `~/.config/opencode/skills/init-memini/` (user) | `AGENTS.md` |
| `kimi-code` | `.kimi-code/mcp.json` | not installed | `AGENTS.md` |
| `generic` | prints a `mcpServers` snippet | not installed | `AGENTS.md` |

`kimi-code` and `generic` do not install the skill and do not print it; copy
`init-memini/SKILL.md` by hand if that client supports skills. Both still write the protocol block
to `AGENTS.md` in the current directory — `generic` prints the MCP snippet *and* edits the file.
`kimi-code` ignores `--scope`: its config is always the project's `.kimi-code/mcp.json`.

Where the skill is installed, an existing `init-memini/SKILL.md` is overwritten with the packaged
version, so upgrading memini-ai and re-running `init` refreshes it. The config edit and the
protocol block are never overwritten: the config is merged key by key with a timestamped `.bak-`
copy alongside it, and the protocol block is appended only if the start marker isn't already
present on a line by itself — mentioning the marker in prose elsewhere in the file (like this
paragraph does) doesn't count as already-installed.

Cursor, Codex, Pi, Hermes, and Gemini CLI: use `generic` and paste the snippet into that client's
MCP config.

## The protocol block

The block is the part that makes memory useful. It tells the agent to call `orient` first, `recall`
before re-deriving, and `remember` after decisions and handoffs, with `supersedes` when a fact
changes. It is about 150 tokens and lives in the instructions file the client already loads.

## The skill

`/init-memini` (or the client's equivalent) walks the agent through setup and repair: start the
database, run `init`, reload, call `orient`, confirm `status.db == "ok"`.
