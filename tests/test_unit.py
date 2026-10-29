from pathlib import Path

import pytest

from copilot import audit, citations, llm
from copilot.ingest import UnsupportedFileError, parse_document, split_into_chunks

SAMPLES = Path(__file__).resolve().parent.parent / "sample_docs"
RETRIEVED = {
    "S1": "ATL-1 ran at an average utilization of 87% against a 75% operating target.",
    "S2": "Phase 1 would add 24 MW at an estimated capital cost of $310 million.",
}


# --- chunking ----------------------------------------------------------------

def test_pdf_chunks_keep_document_and_page():
    pdf = SAMPLES / "ApexSphere_Q1_Data_Center_Operations_Report.pdf"
    doc = parse_document(pdf.read_bytes(), pdf.name)
    assert doc.num_pages == 4
    assert {c.page for c in doc.chunks} == {1, 2, 3, 4}
    assert all(c.page is not None for c in doc.chunks)
    page3 = " ".join(c.text for c in doc.chunks if c.page == 3)
    assert "February 11" in page3


def test_markdown_sections_do_not_invent_pages():
    md = b"# Overview\nIntro text.\n\n## Risks\nPower risk is high.\n"
    doc = parse_document(md, "notes.md")
    assert [c.section for c in doc.chunks] == ["Overview", "Risks"]
    assert all(c.page is None for c in doc.chunks)


def test_chunks_respect_max_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about capacity and cooling." for i in range(80))
    chunks = split_into_chunks(text, max_chars=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[1].split("\n")[0] in chunks[0]  # overlap carried forward


def test_unsupported_and_oversized_files_rejected(monkeypatch):
    with pytest.raises(UnsupportedFileError):
        parse_document(b"data", "payload.exe")
    monkeypatch.setattr("copilot.config.MAX_UPLOAD_MB", 0.0001)
    with pytest.raises(UnsupportedFileError):
        parse_document(b"x" * 1000, "big.txt")


# --- citation check ----------------------------------------------------------

def test_citation_outside_retrieved_set_is_rejected():
    claim = citations.check_claim({"statement": "Utilization was 87%.", "citations": ["S9"]}, RETRIEVED)
    assert claim.status == citations.REJECTED
    assert "S9" in claim.reason


def test_uncited_claim_is_rejected():
    claim = citations.check_claim({"statement": "Expansion is wise.", "citations": []}, RETRIEVED)
    assert claim.status == citations.REJECTED


def test_valid_citation_verified_and_labels_normalized():
    claim = citations.check_claim({"statement": "Utilization was 87% vs a 75% target.", "citations": ["[s1]"]}, RETRIEVED)
    assert claim.status == citations.VERIFIED
    assert claim.valid_citations == ["S1"]


def test_figure_missing_from_cited_passage_needs_review():
    claim = citations.check_claim({"statement": "Phase 1 costs $410 million.", "citations": ["S2"]}, RETRIEVED)
    assert claim.status == citations.NEEDS_REVIEW
    assert claim.unmatched_figures == ["410"]


def test_figure_cited_from_wrong_passage_needs_review():
    claim = citations.check_claim({"statement": "Phase 1 adds 24 MW.", "citations": ["S1"]}, RETRIEVED)
    assert claim.status == citations.NEEDS_REVIEW


def test_mixed_valid_and_invalid_citations_keeps_valid_only():
    claim = citations.check_claim({"statement": "Utilization was 87%.", "citations": ["S1", "S7"]}, RETRIEVED)
    assert claim.status == citations.VERIFIED
    assert claim.invalid_citations == ["S7"]


def test_evidence_status_rules():
    ok = citations.check_claims([{"statement": "Utilization was 87%.", "citations": ["S1"]}], RETRIEVED)
    bad = citations.check_claims([{"statement": "Revenue was huge.", "citations": ["S5"]}], RETRIEVED)
    assert citations.evidence_status(ok, "answered") == citations.EVIDENCE_AVAILABLE
    assert citations.evidence_status(ok, "partial") == citations.PARTIAL_EVIDENCE
    assert citations.evidence_status(bad, "answered") == citations.INSUFFICIENT_EVIDENCE
    assert citations.evidence_status([], "insufficient_evidence") == citations.INSUFFICIENT_EVIDENCE


# --- JSON repair & audit hashing ---------------------------------------------

def test_repair_json_handles_fences_and_trailing_commas():
    assert llm.repair_json('```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    assert llm.repair_json('Here you go: {"status": "answered"} thanks') == {"status": "answered"}
    with pytest.raises(llm.LLMError):
        llm.repair_json("not json at all")


def test_audit_hash_changes_when_record_changes():
    rec = {"action": "approve", "target": "risks", "details": {"comment": "ok"}}
    h1 = audit.compute_hash(audit.GENESIS_HASH, rec)
    assert h1 == audit.compute_hash(audit.GENESIS_HASH, dict(reversed(list(rec.items()))))
    assert h1 != audit.compute_hash(audit.GENESIS_HASH, {**rec, "action": "reject"})


def test_running_footer_removed_but_content_kept():
    pdf = SAMPLES / "ApexSphere_Q2_Financial_and_Expansion_Strategy.pdf"
    doc = parse_document(pdf.read_bytes(), pdf.name)
    assert all("demonstration only" not in c.text for c in doc.chunks)
    assert any("$38 million" in c.text and c.page == 3 for c in doc.chunks)
