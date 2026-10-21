-- ndup schema, architecture v3 §7 plus the sync queue in §5.3.
-- embedding_model.runtime is part of the unique key so ONNX int8 and CUDA fp16
-- register as different model ids and do not share an embedding cache (WP0.4).
-- HNSW indexes are not created here; model registration creates them.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE notion_user (
  id uuid PRIMARY KEY,
  type text,
  name text,
  email text,
  avatar_url text,
  updated_at timestamptz
);

CREATE TABLE root (
  id uuid PRIMARY KEY,
  label text NOT NULL,
  kind text NOT NULL,
  excluded bool NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE data_source (
  id uuid PRIMARY KEY,
  database_id uuid,
  title text,
  row_count int,
  template_title text,
  poll_interval interval DEFAULT '1 hour',
  next_poll_at timestamptz,
  last_polled_at timestamptz,
  last_change_seen timestamptz,
  watermark timestamptz
);

CREATE TABLE document (
  id uuid PRIMARY KEY,
  kind text NOT NULL,
  parent_type text,
  parent_id uuid,
  data_source_id uuid REFERENCES data_source,
  root_id uuid REFERENCES root,
  area text,
  breadcrumb text[],
  title text,
  title_hash bytea,
  url text,
  created_time timestamptz,
  created_by uuid,
  last_edited_time timestamptz,
  last_edited_by uuid,
  last_content_change timestamptz,
  properties jsonb,
  markdown text,
  content_hash bytea,
  token_count int,
  lang text,
  contributors uuid[],
  is_stub bool NOT NULL DEFAULT false,
  status text NOT NULL DEFAULT 'active',
  discovered_via text,
  first_seen_at timestamptz,
  last_synced_at timestamptz
);
CREATE INDEX ON document (last_edited_time);
CREATE INDEX ON document (data_source_id);
CREATE INDEX ON document (parent_id);
CREATE INDEX ON document (root_id);
CREATE INDEX ON document (status) WHERE status <> 'active';

CREATE TABLE document_version (
  doc_id uuid REFERENCES document ON DELETE CASCADE,
  content_hash bytea,
  last_edited_time timestamptz,
  last_edited_by uuid,
  markdown_zstd bytea,
  synced_at timestamptz,
  PRIMARY KEY (doc_id, content_hash)
);

CREATE TABLE block (
  id uuid PRIMARY KEY,
  doc_id uuid REFERENCES document ON DELETE CASCADE,
  type text,
  start_char int,
  end_char int,
  created_time timestamptz,
  created_by uuid,
  last_edited_time timestamptz,
  last_edited_by uuid,
  is_synced bool NOT NULL DEFAULT false
) WITH (autovacuum_vacuum_scale_factor = 0.02, autovacuum_analyze_scale_factor = 0.01);
CREATE INDEX ON block (doc_id);

CREATE TABLE doc_link (
  src uuid,
  dst uuid,
  PRIMARY KEY (src, dst)
);

CREATE TABLE chunk (
  id bigserial PRIMARY KEY,
  doc_id uuid REFERENCES document ON DELETE CASCADE,
  ord int,
  heading_path text[],
  text text,
  embed_text text,
  start_char int,
  end_char int,
  block_ids uuid[],
  span_ids text[],
  span_tokens int[],
  token_count int,
  text_hash bytea NOT NULL,
  embed_hash bytea NOT NULL,
  minhash bytea,
  lang text,
  weight real DEFAULT 1,
  is_boilerplate bool NOT NULL DEFAULT false,
  boilerplate_reason text,
  is_synced bool NOT NULL DEFAULT false,
  earliest_block_time timestamptz,
  UNIQUE (doc_id, ord)
) WITH (autovacuum_vacuum_scale_factor = 0.02);
CREATE INDEX ON chunk (text_hash);
CREATE INDEX ON chunk (embed_hash);
CREATE INDEX ON chunk (doc_id);

CREATE TABLE embedding_model (
  id smallserial PRIMARY KEY,
  model text NOT NULL,
  dims int NOT NULL,
  runtime text NOT NULL DEFAULT 'onnx-int8',
  storage text NOT NULL DEFAULT 'halfvec',
  is_active bool NOT NULL DEFAULT false,
  thresholds jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (model, dims, runtime)
);

CREATE TABLE embedding (
  model_id smallint REFERENCES embedding_model (id),
  embed_hash bytea,
  vec halfvec NOT NULL,
  PRIMARY KEY (model_id, embed_hash)
);

CREATE TABLE doc_embedding (
  model_id smallint REFERENCES embedding_model (id),
  doc_id uuid REFERENCES document ON DELETE CASCADE,
  vec halfvec NOT NULL,
  source_hash bytea,
  PRIMARY KEY (model_id, doc_id)
);

CREATE TABLE analysis_run (
  id serial PRIMARY KEY,
  model_id smallint,
  mode text,
  params jsonb,
  started_at timestamptz,
  finished_at timestamptz,
  stats jsonb
);

CREATE TABLE doc_pair (
  doc_a uuid REFERENCES document ON DELETE CASCADE,
  doc_b uuid REFERENCES document ON DELETE CASCADE,
  doc_sim real,
  cov_ab real,
  cov_ba real,
  lex_jaccard real,
  lex_cont_ab real,
  lex_cont_ba real,
  score real,
  relation text,
  first_run int,
  last_run int,
  PRIMARY KEY (doc_a, doc_b),
  CHECK (doc_a < doc_b)
);
CREATE INDEX ON doc_pair (doc_b);
CREATE INDEX ON doc_pair (relation) WHERE relation IN ('near_copy', 'duplicate', 'subset');

CREATE TABLE passage_match (
  doc_a uuid,
  doc_b uuid,
  chunk_a bigint,
  chunk_b bigint,
  sim real,
  lex real,
  PRIMARY KEY (doc_a, doc_b, chunk_a, chunk_b),
  FOREIGN KEY (doc_a, doc_b) REFERENCES doc_pair ON DELETE CASCADE
);
CREATE INDEX ON passage_match (chunk_a);
CREATE INDEX ON passage_match (chunk_b);

CREATE TABLE cluster (
  stable_key uuid PRIMARY KEY,
  label text,
  size int,
  member_hash bytea,
  oldest_doc uuid,
  newest_doc uuid,
  active_copies int,
  areas text[],
  last_run int,
  updated_at timestamptz
);

CREATE TABLE cluster_member (
  stable_key uuid REFERENCES cluster ON DELETE CASCADE,
  doc_id uuid REFERENCES document ON DELETE CASCADE,
  relation_to_cluster text,
  is_satellite bool NOT NULL DEFAULT false,
  PRIMARY KEY (stable_key, doc_id)
);
CREATE INDEX ON cluster_member (doc_id);

CREATE TABLE cluster_event (
  id bigserial PRIMARY KEY,
  stable_key uuid,
  doc_id uuid,
  event text,
  run_id int,
  at timestamptz
);
CREATE INDEX ON cluster_event (at);
CREATE INDEX ON cluster_event (stable_key);

CREATE TABLE cluster_review (
  stable_key uuid PRIMARY KEY,
  status text,
  reviewed_member_hash bytea,
  note text,
  reviewed_by text,
  reviewed_at timestamptz
);

CREATE TABLE ignore_rule (
  id serial PRIMARY KEY,
  kind text,
  a uuid,
  b uuid,
  created_by text,
  created_at timestamptz,
  UNIQUE (kind, a, b)
);

CREATE TABLE calibration_label (
  doc_a uuid,
  doc_b uuid,
  label text,
  labeled_by text,
  labeled_at timestamptz,
  PRIMARY KEY (doc_a, doc_b)
);

CREATE TABLE sync_state (
  key text PRIMARY KEY,
  value jsonb
);

CREATE TABLE sync_task (
  doc_id uuid PRIMARY KEY,
  kind text NOT NULL,
  priority int NOT NULL,
  remote_last_edited timestamptz,
  discovered_via text,
  state text DEFAULT 'pending',
  attempts int DEFAULT 0,
  next_attempt_at timestamptz,
  last_error text,
  leased_until timestamptz
);
CREATE INDEX ON sync_task (state, priority, next_attempt_at);

CREATE TABLE webhook_event (
  id uuid PRIMARY KEY,
  type text,
  entity_id uuid,
  received_at timestamptz,
  processed_at timestamptz
);

CREATE TABLE audit_log (
  at timestamptz,
  user_email text,
  action text,
  target_kind text,
  target_id uuid
);
CREATE INDEX ON audit_log (at);
