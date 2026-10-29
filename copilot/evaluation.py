"""Evaluation harness: retrieval recall, faithfulness, and abstention on a fixed question set.

Metrics
- Retrieval recall:  share of expected (document, page) pairs found in the retrieved passages.
- Faithfulness:      share of displayed claims an LLM judge finds fully supported by their cited passages.
- Abstention:        answerable questions must be answered; unanswerable ones must return
                     "insufficient evidence".
- Citation rejects:  claims the deterministic citation check removed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import citations, config, llm, store
from .cost import Usage, summarize
from .embeddings import Embedder
from .indexing import index_file
from .rag import GroundedAnswer, answer_question
from .tracing import SPAN_KIND, set_io, tracer

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_index": {"type": "integer"},
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["claim_index", "supported", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

JUDGE_INSTRUCTIONS = """You are a strict fact-checking judge. For each numbered claim, decide whether the passages
cited for it fully support it. A claim is supported only if every fact, figure and relationship it states is
explicitly stated in, or directly implied by, its cited passages. Background knowledge does not count.
Return one verdict per claim."""


@dataclass
class CaseResult:
    id: str
    question: str
    answerable: bool
    status: str
    gate: str
    abstention_correct: bool
    retrieval_recall: float | None
    retrieved: list[str]
    expected: list[str]
    faithfulness: float | None
    claims_shown: int
    claims_rejected: int
    claims_needing_review: int
    answer: list[str]
    judge_notes: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    cost_usd: float = 0.0
    error: str = ""


def load_questions(path: Path | None = None) -> list[dict]:
    return json.loads(Path(path or config.EVAL_QUESTIONS_PATH).read_text())["questions"]


def ensure_sample_docs_indexed(workspace: str, embedder: Embedder, docs_dir: Path | None = None) -> list[str]:
    docs_dir = Path(docs_dir or config.SAMPLE_DOCS_DIR)
    messages = []
    for pdf in sorted(docs_dir.glob("*.pdf")):
        result = index_file(workspace, pdf.name, pdf.read_bytes(), embedder)
        messages.append(f"{pdf.name}: {result.status}" + (f" ({result.chunks} chunks)" if result.chunks else ""))
    return messages


def _ref(source: str, page: int | None) -> str:
    return f"{source}#p{page}"


def judge_faithfulness(answer: GroundedAnswer, judge_model: str) -> tuple[float | None, list[str], Usage | None]:
    shown = answer.shown_claims
    if not shown:
        return None, [], None
    by_label = answer.retrieval.by_label
    lines = []
    for i, claim in enumerate(shown):
        passages = "\n".join(f"  [{lab}] {by_label[lab].chunk.text}" for lab in claim.valid_citations)
        lines.append(f"Claim {i}: {claim.statement}\nCited passages:\n{passages}")
    data, usage = llm.generate_json(
        model_key=judge_model, instructions=JUDGE_INSTRUCTIONS, user_input="\n\n".join(lines),
        schema=JUDGE_SCHEMA, schema_name="faithfulness_verdicts", purpose="eval_judge",
    )
    verdicts = {v["claim_index"]: v for v in data.get("verdicts", [])}
    supported = sum(1 for i in range(len(shown)) if verdicts.get(i, {}).get("supported"))
    notes = [f"Claim {i}: {v['reason']}" for i, v in sorted(verdicts.items()) if not v.get("supported")]
    return supported / len(shown), notes, usage


def run_case(case: dict, *, workspace: str, model_key: str, judge_model: str, embedder: Embedder,
             k: int | None, min_similarity: float | None) -> CaseResult:
    expected = [_ref(e["source"], e.get("page")) for e in case.get("expected", [])]
    answerable = bool(case.get("answerable", True))
    with tracer.start_as_current_span(f"eval_case {case['id']}") as span:
        span.set_attribute(SPAN_KIND, "CHAIN")
        set_io(span, input_value=case["question"])
        try:
            answer = answer_question(workspace, case["question"], model_key=model_key, embedder=embedder,
                                     k=k, min_similarity=min_similarity)
        except Exception as exc:
            return CaseResult(case["id"], case["question"], answerable, "error", "", False, None, [], expected,
                              None, 0, 0, 0, [], error=str(exc))

        retrieved = [_ref(e.chunk.source, e.chunk.page) for e in answer.retrieval.evidence]
        recall = (sum(1 for ref in expected if ref in retrieved) / len(expected)) if expected else None
        abstained = answer.status == citations.INSUFFICIENT_EVIDENCE
        usages = list(answer.usage)

        faithfulness, notes = None, []
        if not abstained:
            faithfulness, notes, judge_usage = judge_faithfulness(answer, judge_model)
            if judge_usage:
                usages.append(judge_usage)

        result = CaseResult(
            id=case["id"], question=case["question"], answerable=answerable, status=answer.status,
            gate=answer.gate, abstention_correct=(not abstained) if answerable else abstained,
            retrieval_recall=recall, retrieved=retrieved, expected=expected, faithfulness=faithfulness,
            claims_shown=len(answer.shown_claims), claims_rejected=len(answer.rejected_claims),
            claims_needing_review=sum(c.status == citations.NEEDS_REVIEW for c in answer.claims),
            answer=[c.statement for c in answer.shown_claims], judge_notes=notes,
            latency_s=round(answer.latency_s, 2), cost_usd=round(summarize(usages)["cost_usd"], 6),
        )
        set_io(span, output_value={"status": result.status, "recall": recall, "faithfulness": faithfulness,
                                   "abstention_correct": result.abstention_correct})
        return result


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize_results(results: list[CaseResult]) -> dict:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    return {
        "questions": len(results),
        "retrieval_recall": _mean([r.retrieval_recall for r in answerable if r.retrieval_recall is not None]),
        "faithfulness": _mean([r.faithfulness for r in results if r.faithfulness is not None]),
        "answered_when_answerable": f"{sum(r.abstention_correct for r in answerable)}/{len(answerable)}",
        "abstained_when_unanswerable": f"{sum(r.abstention_correct for r in unanswerable)}/{len(unanswerable)}",
        "abstention_accuracy": _mean([1.0 if r.abstention_correct else 0.0 for r in results]),
        "claims_rejected_by_citation_check": sum(r.claims_rejected for r in results),
        "claims_flagged_for_review": sum(r.claims_needing_review for r in results),
        "errors": sum(1 for r in results if r.error),
        "mean_latency_s": _mean([r.latency_s for r in results]),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 4),
    }


def run_evaluation(
    *,
    model_key: str | None = None,
    judge_model: str | None = None,
    embedder: Embedder,
    workspace: str | None = None,
    k: int | None = None,
    min_similarity: float | None = None,
    questions_path: Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    save: bool = True,
) -> dict:
    model_key = model_key or config.DEFAULT_MODEL
    judge_model = judge_model or config.JUDGE_MODEL
    workspace = workspace or config.EVAL_WORKSPACE
    store.ensure_schema()
    index_log = ensure_sample_docs_indexed(workspace, embedder)
    cases = load_questions(questions_path)

    started = time.perf_counter()
    results: list[CaseResult] = []
    with tracer.start_as_current_span("evaluation_run") as span:
        span.set_attribute(SPAN_KIND, "CHAIN")
        for i, case in enumerate(cases, start=1):
            if progress:
                progress(i, len(cases), case["question"])
            results.append(run_case(case, workspace=workspace, model_key=model_key, judge_model=judge_model,
                                    embedder=embedder, k=k, min_similarity=min_similarity))
        summary = summarize_results(results)
        set_io(span, output_value=summary)

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": llm.resolve_model(model_key).model_id,
        "judge_model": llm.resolve_model(judge_model).model_id,
        "workspace": workspace,
        "top_k": k or config.TOP_K,
        "min_similarity": config.MIN_SIMILARITY if min_similarity is None else min_similarity,
        "duration_s": round(time.perf_counter() - started, 1),
        "indexing": index_log,
        "summary": summary,
        "cases": [asdict(r) for r in results],
    }
    if save:
        save_report(report)
    return report


def save_report(report: dict) -> Path:
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = report["run_at"].replace(":", "").replace("-", "")
    path = config.EVAL_RESULTS_DIR / f"eval_{stamp}.json"
    payload = json.dumps(report, indent=2)
    path.write_text(payload)
    (config.EVAL_RESULTS_DIR / "latest.json").write_text(payload)
    return path


def load_latest_report() -> dict | None:
    path = config.EVAL_RESULTS_DIR / "latest.json"
    return json.loads(path.read_text()) if path.exists() else None
