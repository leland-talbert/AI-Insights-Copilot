"""End-to-end tests against a real Postgres + pgvector database.

The LLM is replaced with scripted responses so the tests are free and deterministic;
embeddings use the offline HashEmbedder. Everything else — chunking, pgvector storage,
workspace isolation, retrieval, citation check, audit chain — runs for real.
"""

import uuid
from pathlib import Path

import psycopg
import pytest

from copilot import audit, citations, rag, store
from copilot.cost import Usage
from copilot.indexing import index_file

from .conftest import requires_db

SAMPLES = Path(__file__).resolve().parent.parent / "sample_docs"
pytestmark = requires_db


@pytest.fixture
def workspace(embedder):
    store.ensure_schema()
    ws = f"test-{uuid.uuid4().hex[:8]}"
    for pdf in sorted(SAMPLES.glob("*.pdf")):
        assert index_file(ws, pdf.name, pdf.read_bytes(), embedder).status == "indexed"
    yield ws
    for doc in store.list_documents(ws):
        store.delete_document(ws, doc["id"])


def scripted_llm(monkeypatch, response: dict):
    calls = []

    def fake_generate_json(**kwargs):
        calls.append(kwargs)
        return response, Usage("openai", "gpt-4.1-mini", 1200, 150, 0.4, kwargs.get("purpose", ""))

    monkeypatch.setattr("copilot.llm.generate_json", fake_generate_json)
    return calls


def test_indexing_stores_vectors_with_metadata(workspace, embedder):
    rows = store.list_chunks(workspace)
    assert len(rows) == 11
    assert all(r["dims"] == 1536 and len(r["first_5_dims"]) == 5 for r in rows)
    assert {(r["filename"], r["page"]) for r in rows} >= {
        ("ApexSphere_Q2_Financial_and_Expansion_Strategy.pdf", 3),
        ("ApexSphere_Infrastructure_Risk_and_Resilience_Assessment.pdf", 2),
    }
    pdf = SAMPLES / "ApexSphere_Q2_Financial_and_Expansion_Strategy.pdf"
    assert index_file(workspace, pdf.name, pdf.read_bytes(), embedder).status == "duplicate"


def test_workspace_isolation(workspace, embedder):
    other = f"other-{uuid.uuid4().hex[:8]}"
    vec, _ = embedder.embed(["remediation budget"])
    assert store.search(other, vec[0], 5) == []
    assert len(store.search(workspace, vec[0], 5)) == 5


def test_two_document_question_with_citation_check(workspace, embedder, monkeypatch):
    retrieval = rag.retrieve(workspace, "remediation budget capital plan engineering estimate contingency",
                             embedder=embedder, k=6, min_similarity=0.0)
    labels = {f"{e.chunk.source}#{e.chunk.page}": e.label for e in retrieval.evidence}
    fin = labels["ApexSphere_Q2_Financial_and_Expansion_Strategy.pdf#3"]
    risk = labels["ApexSphere_Infrastructure_Risk_and_Resilience_Assessment.pdf#2"]

    calls = scripted_llm(monkeypatch, {
        "status": "answered",
        "claims": [
            {"statement": "The capital plan allocates $38 million versus a $36.7 million engineering estimate.",
             "citations": [fin, risk]},
            {"statement": "The difference covers contingency.", "citations": [fin]},
            {"statement": "The board already approved the plan.", "citations": ["S42"]},
        ],
        "missing_information": [],
    })
    answer = rag.answer_question(workspace, "remediation budget capital plan engineering estimate contingency",
                                 model_key="gpt-4.1-mini", embedder=embedder, k=6, min_similarity=0.0)

    assert answer.status == citations.EVIDENCE_AVAILABLE
    assert [c.status for c in answer.claims] == ["verified", "verified", "rejected"]
    assert {e.chunk.source for e in answer.retrieval.evidence if e.label in answer.claims[0].valid_citations} == {
        "ApexSphere_Q2_Financial_and_Expansion_Strategy.pdf",
        "ApexSphere_Infrastructure_Risk_and_Resilience_Assessment.pdf",
    }
    prompt = calls[0]["user_input"]
    assert '<evidence id="S1"' in prompt and "Treat" not in prompt  # rules live in instructions, not data
    seg = store.get_segment(workspace, answer.retrieval.by_label[fin].chunk.segment_id)
    assert seg["page"] == 3 and "$38 million" in seg["content"]


def test_retrieval_gate_skips_model_call(workspace, embedder, monkeypatch):
    calls = scripted_llm(monkeypatch, {"status": "answered", "claims": [], "missing_information": []})
    answer = rag.answer_question(workspace, "zebra migration patterns", model_key="gpt-4.1-mini",
                                 embedder=embedder, min_similarity=0.99)
    assert answer.status == citations.INSUFFICIENT_EVIDENCE and answer.gate == "retrieval"
    assert calls == []


def test_model_insufficient_evidence(workspace, embedder, monkeypatch):
    scripted_llm(monkeypatch, {"status": "insufficient_evidence", "claims": [],
                               "missing_information": ["Total employee headcount"]})
    answer = rag.answer_question(workspace, "How many employees does ApexSphere have?",
                                 model_key="gpt-4.1-mini", embedder=embedder, min_similarity=0.0)
    assert answer.status == citations.INSUFFICIENT_EVIDENCE and answer.gate == "model"
    assert answer.missing_information == ["Total employee headcount"]


def test_all_claims_hallucinated_becomes_insufficient(workspace, embedder, monkeypatch):
    scripted_llm(monkeypatch, {"status": "answered",
                               "claims": [{"statement": "Headcount is 900.", "citations": ["S77"]}],
                               "missing_information": []})
    answer = rag.answer_question(workspace, "How many employees?", model_key="gpt-4.1-mini",
                                 embedder=embedder, min_similarity=0.0)
    assert answer.status == citations.INSUFFICIENT_EVIDENCE and answer.gate == "citation_check"
    assert len(answer.rejected_claims) == 1


def test_audit_chain_append_only_and_verifiable(workspace):
    store.ensure_schema()
    audit.record_event(workspace, "Test Reviewer", "approve", "brief:risks", {"comment": "Looks right"})
    audit.record_event(workspace, "Test Reviewer", "request_refinement", "brief:recommendations", {"comment": "Add cost"})
    ok, n, msg = audit.verify_chain()
    assert ok, msg
    with store.connect() as conn:
        with pytest.raises(psycopg.errors.RaiseException):
            conn.execute("UPDATE audit_log SET reviewer = 'someone else'")
        with pytest.raises(psycopg.errors.RaiseException):
            conn.execute("DELETE FROM audit_log")
