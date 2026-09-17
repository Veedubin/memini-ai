# Architecture

    server.py  ->  store.py  ->  db.py (asyncpg, migrations, SQL)
                            ->  embed.py (BGE-M3 or hash)
    cli.py     ->  ingest.py, clients.py, and the same store

One table, `memories`, holds every kind. Thought chains are rows with `chain_id` and
`thought_number`, grouped by a small `chains` table. `superseded_by` is the only relationship.
`tsv` is a generated `tsvector` column with a GIN index; `embedding` is `vector(1024)` with an HNSW
index. Migrations are numbered SQL files applied in order and recorded in `schema_migrations`.

The server connects lazily on the first tool call behind a lock. A failed connection is reported
through `orient.status` and retried on the next call. Model load failure degrades `recall` to
full-text only and says so.

There is no LLM in the loop, no background task, and no in-process index that can go stale.
