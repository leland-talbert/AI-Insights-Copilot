"""pgvector storage: documents, page/section segments, embedded chunks.

Every read and write is scoped by workspace, so one workspace can never
retrieve another workspace's documents.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from . import config
from .ingest import ParsedDocument

SCHEMA_PATH = config.PROJECT_ROOT / "db" / "schema.sql"


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    segment_id: int
    source: str
    page: int | None
    section: str | None
    chunk_index: int
    text: str
    similarity: float


@contextlib.contextmanager
def connect(url: str | None = None) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(url or config.DATABASE_URL, autocommit=True, row_factory=dict_row)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def ensure_schema(url: str | None = None) -> None:
    with psycopg.connect(url or config.DATABASE_URL, autocommit=True) as conn:
        conn.execute(SCHEMA_PATH.read_text())


def find_document(workspace: str, sha256: str) -> dict | None:
    with connect() as conn:
        return conn.execute(
            "SELECT id, filename FROM documents WHERE workspace = %s AND sha256 = %s",
            (workspace, sha256),
        ).fetchone()


def insert_document(
    workspace: str,
    doc: ParsedDocument,
    embeddings: Sequence[np.ndarray],
    embedding_model: str,
) -> int:
    if len(embeddings) != len(doc.chunks):
        raise ValueError("One embedding is required per chunk.")
    with connect() as conn, conn.transaction():
        doc_id = conn.execute(
            """INSERT INTO documents (workspace, filename, sha256, file_type, num_pages, size_bytes)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (workspace, doc.filename, doc.sha256, doc.file_type, doc.num_pages, doc.size_bytes),
        ).fetchone()["id"]

        segment_ids: dict[int, int] = {}
        for seg in doc.segments:
            segment_ids[seg.seq] = conn.execute(
                """INSERT INTO segments (document_id, seq, page, section, content)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (doc_id, seg.seq, seg.page, seg.section, seg.text),
            ).fetchone()["id"]

        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (document_id, segment_id, workspace, page, section,
                                       chunk_index, content, embedding_model, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (doc_id, segment_ids[c.segment_seq], workspace, c.page, c.section,
                     c.chunk_index, c.text, embedding_model, np.asarray(e, dtype=np.float32))
                    for c, e in zip(doc.chunks, embeddings)
                ],
            )
    return doc_id


def list_documents(workspace: str) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            """SELECT d.id, d.filename, d.file_type, d.num_pages, d.size_bytes, d.uploaded_at,
                      COUNT(c.id) AS chunks
               FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
               WHERE d.workspace = %s
               GROUP BY d.id ORDER BY d.filename""",
            (workspace,),
        ).fetchall()


def delete_document(workspace: str, document_id: int) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM documents WHERE workspace = %s AND id = %s", (workspace, document_id))
        return cur.rowcount > 0


def list_chunks(workspace: str, document_id: int | None = None, limit: int = 500) -> list[dict]:
    """Chunk inventory with metadata and the first values of each stored vector."""
    sql = """SELECT d.filename, c.page, c.section, c.chunk_index, length(c.content) AS chars,
                    c.content, vector_dims(c.embedding) AS dims,
                    (c.embedding::real[])[1:5] AS first_5_dims
             FROM chunks c JOIN documents d ON d.id = c.document_id
             WHERE c.workspace = %s"""
    params: list = [workspace]
    if document_id is not None:
        sql += " AND c.document_id = %s"
        params.append(document_id)
    sql += " ORDER BY d.filename, c.chunk_index LIMIT %s"
    params.append(limit)
    with connect() as conn:
        return conn.execute(sql, params).fetchall()


def search(workspace: str, query_embedding: np.ndarray, k: int) -> list[RetrievedChunk]:
    """Cosine-similarity search restricted to one workspace."""
    vec = np.asarray(query_embedding, dtype=np.float32)
    with connect() as conn:
        rows = conn.execute(
            """SELECT c.id AS chunk_id, c.document_id, c.segment_id, d.filename AS source,
                      c.page, c.section, c.chunk_index, c.content AS text,
                      1 - (c.embedding <=> %s) AS similarity
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.workspace = %s
               ORDER BY c.embedding <=> %s
               LIMIT %s""",
            (vec, workspace, vec, k),
        ).fetchall()
    return [RetrievedChunk(**{**r, "similarity": float(r["similarity"])}) for r in rows]


def get_segment(workspace: str, segment_id: int) -> dict | None:
    """Full page (or section) text behind a citation."""
    with connect() as conn:
        return conn.execute(
            """SELECT s.page, s.section, s.content, d.filename
               FROM segments s JOIN documents d ON d.id = s.document_id
               WHERE s.id = %s AND d.workspace = %s""",
            (segment_id, workspace),
        ).fetchone()


def get_document_text(workspace: str, document_id: int) -> tuple[str, str] | None:
    with connect() as conn:
        rows = conn.execute(
            """SELECT d.filename, s.page, s.section, s.content
               FROM segments s JOIN documents d ON d.id = s.document_id
               WHERE d.workspace = %s AND d.id = %s ORDER BY s.seq""",
            (workspace, document_id),
        ).fetchall()
    if not rows:
        return None
    parts = []
    for r in rows:
        header = f"[Page {r['page']}]" if r["page"] is not None else (f"[Section: {r['section']}]" if r["section"] else "")
        parts.append(f"{header}\n{r['content']}".strip())
    return rows[0]["filename"], "\n\n".join(parts)


def health_check() -> tuple[bool, str]:
    try:
        with connect() as conn:
            version = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        return True, f"pgvector {version['extversion']}" if version else "connected (pgvector missing)"
    except Exception as exc:  # pragma: no cover - surfaced in UI
        return False, str(exc).splitlines()[0]
