-- AI Insights Copilot — Postgres + pgvector schema
-- Idempotent: safe to run on every app start. Also mounted by docker-compose
-- as an init script. Embedding dimension (1536) matches text-embedding-3-small;
-- changing the embedding model to a different dimension requires re-indexing.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per uploaded file, scoped to a workspace (document-set isolation).
CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    workspace    TEXT        NOT NULL,
    filename     TEXT        NOT NULL,
    sha256       TEXT        NOT NULL,
    file_type    TEXT        NOT NULL,
    num_pages    INTEGER,
    size_bytes   INTEGER     NOT NULL,
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace, sha256)
);

-- Source units before chunking: one PDF page, or one Markdown/JSON section.
-- Kept so a citation can open the full page it came from.
CREATE TABLE IF NOT EXISTS segments (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,
    page         INTEGER,           -- NULL for non-paginated files (never invented)
    section      TEXT,
    content      TEXT    NOT NULL
);

-- Searchable chunks. Every chunk carries its document, page and section.
CREATE TABLE IF NOT EXISTS chunks (
    id               BIGSERIAL PRIMARY KEY,
    document_id      BIGINT  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    segment_id       BIGINT  NOT NULL REFERENCES segments(id)  ON DELETE CASCADE,
    workspace        TEXT    NOT NULL,
    page             INTEGER,
    section          TEXT,
    chunk_index      INTEGER NOT NULL,
    content          TEXT    NOT NULL,
    embedding_model  TEXT    NOT NULL,
    embedding        vector(1536) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_workspace_idx ON chunks (workspace);
CREATE INDEX IF NOT EXISTS chunks_document_idx  ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Human-in-the-loop review trail: append-only and hash-chained.
-- Each row stores the SHA-256 of (previous hash + canonical record), so any
-- edit to history breaks verification. Triggers block UPDATE/DELETE/TRUNCATE.
-- This is tamper-evident, not tamper-proof: a database superuser can still
-- drop the triggers. Production would ship hashes to WORM storage.
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL,
    workspace   TEXT  NOT NULL,
    reviewer    TEXT  NOT NULL,
    action      TEXT  NOT NULL,
    target      TEXT  NOT NULL,
    record      JSONB NOT NULL,
    prev_hash   TEXT  NOT NULL,
    hash        TEXT  NOT NULL UNIQUE
);

CREATE OR REPLACE FUNCTION audit_log_block_changes() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only (% blocked)', TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_log_no_update ON audit_log;
CREATE TRIGGER audit_log_no_update
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_log_block_changes();

DROP TRIGGER IF EXISTS audit_log_no_truncate ON audit_log;
CREATE TRIGGER audit_log_no_truncate
    BEFORE TRUNCATE ON audit_log
    FOR EACH STATEMENT EXECUTE FUNCTION audit_log_block_changes();
