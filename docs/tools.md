# The three tools

Every tool returns a JSON object. Failures are `{"error": "..."}` and nothing else.

## orient

`orient(project=None, budget=300)`

Call it first in every session. Returns recent decisions, latest handoffs, a count of open thought
chains, active projects, a rendered `text` block trimmed to `budget` tokens, and a `status` object
with `db`, `model`, and `memories`. It never writes.

`project` is not a default; omitting it is a deliberate cross-project view across every project in
the store, not the server's `MEMINI_PROJECT`. Passing `project` scopes decisions, handoffs, the open
chain count, and the projects list to that one project. In the cross-project view, each decision and
handoff line in `text` is prefixed with its project, e.g. `[my-repo] switched to pgvector`.

## recall

`recall(query, limit=8, kind=None, project=None, since=None, include_superseded=False, chain_id=None)`

Hybrid search: pgvector cosine top-20 and Postgres full-text top-20, fused with reciprocal rank
fusion. Filters are SQL predicates. `since` accepts `7d`, `24h`, or an ISO date. Superseded
memories are hidden unless asked for. `chain_id` returns a thought chain in order and ignores `query`.
If the embedding model is unavailable the response carries `degraded: "text-only"`.

Only results that came back from a search increment `retrieval_count`; reading a chain with
`chain_id` returns it in order without touching `retrieval_count`.

## remember

`remember(text, kind="note", project=None, tags=None, supersedes=None, chain=None)`

`kind` is one of `note`, `decision`, `handoff`, `fact`, `thought`. `supersedes` takes the id of the
memory this one replaces. Duplicate text in the same project returns the existing id with
`duplicate: true`; if `supersedes` is also given, the supersede is still applied to the existing row
and the response carries `superseded_id`. Text is limited to 32 KB.

### Thought chains

Set `kind="thought"` and pass `chain`, a plain object with `number`, `total`, `next_needed`, and
optionally `chain_id`, `revises`, `branch_from`:

    {"number": 1, "total": 3, "next_needed": true}

Omit `chain_id` on the first thought; the response returns one. Later thoughts pass it back. A
`chain_id` from a different project is rejected. `revises` and `branch_from` take thought numbers.
`next_needed: false` closes the chain. Read a chain back with `recall(query="", chain_id=...)`.
