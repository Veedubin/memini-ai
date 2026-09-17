CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chains (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project text,
  status text NOT NULL DEFAULT 'open' CHECK (status IN ('open','done','abandoned')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE memories (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  text text NOT NULL,
  kind text NOT NULL CHECK (kind IN ('note','decision','handoff','fact','thought','session')),
  project text,
  tags text[] NOT NULL DEFAULT '{}',
  content_hash text NOT NULL,
  embedding vector(1024),
  embedding_model text,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  superseded_by uuid REFERENCES memories(id),
  chain_id uuid REFERENCES chains(id) ON DELETE CASCADE,
  thought_number int,
  thought_total int,
  next_needed boolean,
  revises int,
  branch_from int,
  source jsonb NOT NULL DEFAULT '{}',
  retrieval_count int NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX memories_dedup ON memories (coalesce(project, ''), content_hash);
CREATE INDEX memories_embedding ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX memories_tsv ON memories USING gin (tsv);
CREATE INDEX memories_project_created ON memories (project, created_at DESC);
CREATE INDEX memories_chain ON memories (chain_id, thought_number);
