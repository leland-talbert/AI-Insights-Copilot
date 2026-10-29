"""Document parsing and chunking.

PDFs are split page by page so every chunk keeps a real page number.
Markdown and JSON are split by section; their page is left empty rather than invented.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from . import config

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".json"}


@dataclass
class Segment:
    """A source unit before chunking: one PDF page or one text section."""

    seq: int
    text: str
    page: int | None
    section: str | None


@dataclass
class Chunk:
    segment_seq: int
    page: int | None
    section: str | None
    chunk_index: int  # position within the document
    text: str


@dataclass
class ParsedDocument:
    filename: str
    file_type: str
    sha256: str
    size_bytes: int
    num_pages: int | None
    segments: list[Segment] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)


class UnsupportedFileError(ValueError):
    pass


# --- text helpers ----------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _units(text: str, max_chars: int) -> list[str]:
    """Break text into pieces no longer than max_chars: paragraphs, then sentences, then words."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sentence in _SENTENCE_SPLIT.split(para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_chars:
                units.append(sentence)
                continue
            words, current = sentence.split(), ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if len(candidate) > max_chars and current:
                    units.append(current)
                    current = word[:max_chars]
                else:
                    current = candidate[:max_chars]
            if current:
                units.append(current)
    return units


def _tail(text: str, n: int) -> str:
    """Last ~n characters of text, starting on a word boundary."""
    if n <= 0 or not text:
        return ""
    if len(text) <= n:
        return text
    tail = text[-n:]
    space = tail.find(" ")
    return tail[space + 1:] if 0 <= space < len(tail) - 1 else tail


def split_into_chunks(text: str, max_chars: int | None = None, overlap: int | None = None) -> list[str]:
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    overlap = config.CHUNK_OVERLAP_CHARS if overlap is None else overlap
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for unit in _units(text, max_chars):
        candidate = f"{current}\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        tail = _tail(current, overlap)
        current = f"{tail}\n{unit}" if tail and len(tail) + 1 + len(unit) <= max_chars else unit
    if current:
        chunks.append(current)
    return chunks


# --- extraction ------------------------------------------------------------

def strip_repeating_lines(pages: list[str], min_share: float = 0.6) -> list[str]:
    """Remove running headers/footers: short lines that recur on most pages (digits ignored)."""
    if len(pages) < 3:
        return pages
    key = lambda line: re.sub(r"\d+", "#", line.strip().lower())
    counts: dict[str, int] = {}
    for text in pages:
        for k in {key(line) for line in text.split("\n") if 0 < len(line.strip()) <= 120}:
            counts[k] = counts.get(k, 0) + 1
    repeating = {k for k, n in counts.items() if n >= max(2, min_share * len(pages))}
    return [
        normalize_text("\n".join(line for line in text.split("\n") if key(line) not in repeating))
        for text in pages
    ]


def _markdown_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    buf: list[str] = []
    for line in text.split("\n"):
        match = re.match(r"^\s{0,3}#{1,6}\s+(.*)$", line)
        if match:
            if "\n".join(buf).strip():
                sections.append((heading, "\n".join(buf)))
            heading, buf = match.group(1).strip(), [line]
        else:
            buf.append(line)
    if "\n".join(buf).strip():
        sections.append((heading, "\n".join(buf)))
    return sections


def extract_segments(data: bytes, filename: str) -> tuple[list[Segment], int | None]:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(f"Unsupported file type '{ext}'. Use PDF, TXT, Markdown or JSON.")

    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        pages = strip_repeating_lines([normalize_text(p.extract_text() or "") for p in reader.pages])
        segments = []
        for page_number, text in enumerate(pages, start=1):
            if text:
                segments.append(Segment(len(segments), text, page_number, None))
        return segments, len(reader.pages)

    raw = data.decode("utf-8", errors="replace")

    if ext in {".md", ".markdown"}:
        return [
            Segment(i, normalize_text(body), None, heading)
            for i, (heading, body) in enumerate(_markdown_sections(raw))
            if normalize_text(body)
        ], None

    if ext == ".json":
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise UnsupportedFileError(f"Invalid JSON: {exc}") from exc
        if isinstance(obj, dict):
            items = [(str(k), json.dumps(v, indent=2, ensure_ascii=False)) for k, v in obj.items()]
        else:
            items = [(None, json.dumps(obj, indent=2, ensure_ascii=False))]
        return [
            Segment(i, normalize_text(f"{k}: {v}" if k else v), None, k)
            for i, (k, v) in enumerate(items)
            if v.strip()
        ], None

    text = normalize_text(raw)
    return ([Segment(0, text, None, None)] if text else []), None


def parse_document(data: bytes, filename: str) -> ParsedDocument:
    size_limit = int(config.MAX_UPLOAD_MB * 1024 * 1024)
    if len(data) > size_limit:
        raise UnsupportedFileError(f"{filename} is {len(data) / 1e6:.1f} MB; the limit is {config.MAX_UPLOAD_MB:g} MB.")

    segments, num_pages = extract_segments(data, filename)
    chunks: list[Chunk] = []
    for seg in segments:
        for piece in split_into_chunks(seg.text):
            chunks.append(Chunk(seg.seq, seg.page, seg.section, len(chunks), piece))

    return ParsedDocument(
        filename=Path(filename).name,
        file_type=Path(filename).suffix.lower().lstrip("."),
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        num_pages=num_pages,
        segments=segments,
        chunks=chunks,
    )


def location_label(page: int | None, section: str | None) -> str:
    if page is not None:
        return f"p. {page}"
    if section:
        return f"§ {section}"
    return "page n/a"
