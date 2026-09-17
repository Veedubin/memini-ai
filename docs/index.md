# memini-ai

Local-first memory for AI coding agents, exposed over MCP.

Three tools. `orient` tells the agent where things stand. `recall` finds what was decided and why.
`remember` stores a decision, a fact, a handoff, or a thought. Storage is Postgres with pgvector,
embeddings are BGE-M3 on CPU, search is hybrid vector plus full-text.

Nothing here calls an LLM. The agent you already run does the summarizing.

- [Getting started](getting-started.md)
- [The three tools](tools.md)
- [Wire it into Claude Code, OpenCode, Kimi Code](clients.md)
