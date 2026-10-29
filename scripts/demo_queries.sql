-- "Where does it get stored?" — run with:
--   docker exec -i copilot-db psql -U copilot -d copilot < scripts/demo_queries.sql
-- or paste into any SQL client connected to DATABASE_URL.

-- 1. Every chunk with its document, page, a text preview and the stored vector
SELECT d.filename,
       c.page,
       c.chunk_index,
       left(c.content, 60)              AS preview,
       vector_dims(c.embedding)         AS dims,
       (c.embedding::real[])[1:5]       AS first_5_dims
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.workspace = 'demo'
ORDER BY d.filename, c.chunk_index;

-- 2. Documents and chunk counts per workspace
SELECT d.workspace, d.filename, d.num_pages, COUNT(c.id) AS chunks
FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
GROUP BY d.workspace, d.filename, d.num_pages
ORDER BY d.workspace, d.filename;

-- 3. The audit trail is append-only — this fails by design
-- UPDATE audit_log SET reviewer = 'someone else';
