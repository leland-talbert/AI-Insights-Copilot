"""Parse → chunk → embed → store, shared by the app and the evaluation runner."""

from __future__ import annotations

from dataclasses import dataclass

from . import store
from .cost import Usage
from .embeddings import Embedder
from .ingest import ParsedDocument, UnsupportedFileError, parse_document


@dataclass
class IndexResult:
    filename: str
    status: str  # "indexed" | "duplicate" | "error"
    message: str = ""
    document_id: int | None = None
    pages: int | None = None
    chunks: int = 0
    usage: Usage | None = None
    parsed: ParsedDocument | None = None


def index_file(workspace: str, filename: str, data: bytes, embedder: Embedder) -> IndexResult:
    try:
        parsed = parse_document(data, filename)
    except UnsupportedFileError as exc:
        return IndexResult(filename, "error", str(exc))
    except Exception as exc:  # malformed PDFs etc.
        return IndexResult(filename, "error", f"Could not read file: {exc}")

    existing = store.find_document(workspace, parsed.sha256)
    if existing:
        return IndexResult(parsed.filename, "duplicate", "Already indexed in this workspace.",
                           document_id=existing["id"], pages=parsed.num_pages, parsed=parsed)

    if not parsed.chunks:
        return IndexResult(parsed.filename, "error", "No extractable text (scanned PDFs need OCR first).")

    vectors, usage = embedder.embed([c.text for c in parsed.chunks])
    doc_id = store.insert_document(workspace, parsed, vectors, embedder.model)
    return IndexResult(parsed.filename, "indexed", "", doc_id, parsed.num_pages, len(parsed.chunks), usage, parsed)
