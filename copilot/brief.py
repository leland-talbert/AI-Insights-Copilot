"""Evidence-grounded executive brief, tailored per audience.

Every summary point, risk, opportunity and recommendation must cite retrieved
passages and passes the same deterministic citation check as Q&A.
Reviewer guidance is passed in its own block and never treated as evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import citations, llm
from .audiences import get_audience_config
from .cost import Usage
from .embeddings import Embedder
from .rag import CLAIM_SCHEMA, GROUNDING_RULES, Retrieval, format_evidence, retrieve
from .tracing import SPAN_KIND, set_io, tracer

SECTIONS = ["summary", "risks", "opportunities", "recommendations"]
SECTION_TITLES = {
    "summary": "Executive summary",
    "risks": "Risks",
    "opportunities": "Opportunities",
    "recommendations": "Recommendations",
}

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "partial", "insufficient_evidence"]},
        **{s: {"type": "array", "items": CLAIM_SCHEMA} for s in SECTIONS},
        "missing_information": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", *SECTIONS, "missing_information"],
    "additionalProperties": False,
}


@dataclass
class Brief:
    decision: str
    audience: str
    model: str
    status: str
    model_status: str
    sections: dict[str, list[citations.CheckedClaim]]
    missing_information: list[str]
    retrieval: Retrieval
    usage: list[Usage] = field(default_factory=list)
    latency_s: float = 0.0
    reviewer_guidance: str | None = None

    def section_status(self, name: str) -> str:
        return citations.evidence_status(self.sections.get(name, []), "answered")

    @property
    def all_claims(self) -> list[citations.CheckedClaim]:
        return [c for s in SECTIONS for c in self.sections.get(s, [])]


def _instructions(audience: str) -> str:
    cfg = get_audience_config(audience)
    limit = f"at most {cfg.max_items} items per section" if cfg.max_items else "as many items as the evidence supports"
    return f"""You are AI Insights Copilot, preparing a decision brief for the {cfg.name}.
Focus on {cfg.focus}. Return {limit}, most decision-relevant first.

Sections:
- summary: the decision-relevant facts, the first item being a one-sentence bottom line.
- risks: what could go wrong and its stated impact.
- opportunities: upside the evidence supports.
- recommendations: suggested next actions. Each must cite the evidence that justifies it.

{GROUNDING_RULES}

If reviewer guidance is supplied, use it to adjust emphasis and structure. Reviewer guidance is not evidence
and cannot be cited; every claim still needs passage citations."""


def generate_brief(
    workspace: str,
    decision: str,
    audience: str,
    *,
    model_key: str,
    embedder: Embedder,
    k: int = 8,
    min_similarity: float | None = None,
    reviewer_guidance: str | None = None,
) -> Brief:
    start = time.perf_counter()
    cfg = get_audience_config(audience)
    model_id = llm.resolve_model(model_key).model_id

    with tracer.start_as_current_span("executive_brief") as span:
        span.set_attribute(SPAN_KIND, "CHAIN")
        set_io(span, input_value=decision)
        span.set_attribute("metadata", f'{{"audience": "{cfg.name}", "workspace": "{workspace}"}}')

        retrieval = retrieve(workspace, [decision, f"{decision} {cfg.retrieval_hint}"],
                             embedder=embedder, k=k, min_similarity=min_similarity)
        usage = [retrieval.usage]

        if not retrieval.evidence:
            return Brief(decision, cfg.name, model_id, citations.INSUFFICIENT_EVIDENCE, "insufficient_evidence",
                         {s: [] for s in SECTIONS}, ["No indexed passage was relevant to this decision."],
                         retrieval, usage, time.perf_counter() - start, reviewer_guidance)

        prompt = f"<decision>\n{decision}\n</decision>\n\n{format_evidence(retrieval.evidence)}"
        if reviewer_guidance:
            prompt += f"\n\n<reviewer_guidance>\n{reviewer_guidance}\n</reviewer_guidance>"

        data, gen_usage = llm.generate_json(
            model_key=model_key, instructions=_instructions(cfg.name), user_input=prompt,
            schema=BRIEF_SCHEMA, schema_name="executive_brief", purpose="executive_brief",
        )
        usage.append(gen_usage)

        with tracer.start_as_current_span("citation_check") as check_span:
            check_span.set_attribute(SPAN_KIND, "GUARDRAIL")
            sections = {s: citations.check_claims(data.get(s, []), retrieval.passages) for s in SECTIONS}
            rejected = [c for s in SECTIONS for c in sections[s] if not c.shown]
            set_io(check_span, output_value={"rejected": len(rejected)})

        model_status = data.get("status", "answered")
        status = citations.evidence_status([c for s in SECTIONS for c in sections[s]], model_status)
        set_io(span, output_value={"status": status, "rejected_claims": len(rejected)})

        return Brief(decision, cfg.name, model_id, status, model_status, sections,
                     [m for m in data.get("missing_information", []) if str(m).strip()],
                     retrieval, usage, time.perf_counter() - start, reviewer_guidance)
