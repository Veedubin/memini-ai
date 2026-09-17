# Configuration

Every setting is an environment variable with the `MEMINI_` prefix. There are no aliases and
nothing is read from the current directory. A config file at `MEMINI_CONFIG_FILE`
(default `~/.config/memini-ai/config.env`, `KEY=VALUE` lines) supplies defaults; process
environment wins.

| Variable | Default | Meaning |
|---|---|---|
| `MEMINI_DB_URL` | `postgresql://memini:memini@localhost:5555/memini` | asyncpg DSN |
| `MEMINI_MODEL` | `BAAI/bge-m3` | embedding model; `hash` is a test fake |
| `MEMINI_DEVICE` | `cpu` | `cpu` or `cuda` |
| `MEMINI_PROJECT` | unset | default project label |
| `MEMINI_CONFIG_FILE` | `~/.config/memini-ai/config.env` | settings file |
| `MEMINI_LOG_LEVEL` | `INFO` | stderr log level |
| `MEMINI_TIMEOUT_S` | `30` | per-tool timeout |
| `MEMINI_MAX_TEXT_BYTES` | `32768` | `remember` size limit |

`memini-ai init` writes `MEMINI_PROJECT` into the client's MCP entry so each project has its own
label. Everything else comes from the config file or your shell.
