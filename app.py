"""AI Insights Copilot v2 — evidence-grounded executive decision support.

    streamlit run app.py
"""

from __future__ import annotations

import html
import json
import os
import re

import pandas as pd
import streamlit as st

from copilot import __version__, audit, citations, config, store
from copilot.audiences import AUDIENCES, get_audience_config
from copilot.brief import SECTION_TITLES, SECTIONS, Brief, generate_brief
from copilot.classic import analyze_document
from copilot.cost import CostLedger, Usage, summarize
from copilot.ingest import SUPPORTED_EXTENSIONS, collapse_whitespace
from copilot.indexing import index_file
from copilot.llm import LLMError
from copilot.rag import Evidence, GroundedAnswer, answer_question
from copilot.tracing import init_tracing, tracing_enabled

st.set_page_config(page_title="AI Insights Copilot", page_icon="◆", layout="wide")

STATUS_STYLE = {
    citations.EVIDENCE_AVAILABLE: ("#0f7b4a", "#e7f6ee"),
    citations.PARTIAL_EVIDENCE: ("#9a6200", "#fff4dc"),
    citations.INSUFFICIENT_EVIDENCE: ("#a8231c", "#fdecea"),
}
CLAIM_ICON = {citations.VERIFIED: "✓", citations.NEEDS_REVIEW: "⚠"}
SAMPLE_QUESTIONS = [
    "How does the remediation budget in the capital plan compare with the engineering cost estimate?",
    "Should ApexSphere prioritize expanding in Atlanta or resolving its existing infrastructure risks?",
    "How many employees does ApexSphere have in total?",
]
DEFAULT_DECISION = "Should ApexSphere prioritize expanding its Atlanta data center or resolving its existing infrastructure risks?"

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.status-pill {display:inline-block; padding:.28rem .8rem; border-radius:999px; font-weight:600; font-size:.9rem;}
.claim {padding:.55rem .8rem; border-left:3px solid #c9d2e0; margin:.35rem 0 .1rem 0; border-radius:4px;}
.claim.needs_review {border-left-color:#d99a00; background:rgba(217,154,0,.07);}
.claim .flag {font-size:.8rem; color:#9a6200; display:block; margin-top:.2rem;}
.passage {padding:.8rem 1rem; border-radius:6px; background:rgba(120,140,170,.10); line-height:1.55;}
.pagetext {padding:.8rem 1rem; border-radius:6px; border:1px solid rgba(120,140,170,.35); line-height:1.6; font-size:.92rem; max-height:420px; overflow:auto;}
.pagetext mark {background:#ffe58a; color:inherit; padding:0 .1rem;}
.muted {color:#6b7280; font-size:.86rem;}
</style>
""",
    unsafe_allow_html=True,
)


# --- resources ---------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _bootstrap() -> tuple[bool, str, bool]:
    tracing = init_tracing()
    try:
        store.ensure_schema()
        ok, msg = store.health_check()
    except Exception as exc:
        ok, msg = False, str(exc).splitlines()[0]
    return ok, msg, tracing


@st.cache_resource(show_spinner=False)
def get_embedder():
    from copilot.embeddings import OpenAIEmbedder

    return OpenAIEmbedder()


def ledger() -> CostLedger:
    if "ledger" not in st.session_state:
        st.session_state.ledger = CostLedger()
    return st.session_state.ledger


def track(usages: list[Usage]) -> None:
    ledger().add([u for u in usages if u])


# --- shared renderers --------------------------------------------------------

def status_pill(status: str, prefix: str = "") -> str:
    fg, bg = STATUS_STYLE[status]
    return f'<span class="status-pill" style="color:{fg};background:{bg}">{prefix}{status}</span>'


def short_name(filename: str, limit: int = 22) -> str:
    """Compact document label for citation buttons; drops a leading company-style word on long names."""
    words = re.sub(r"[_\-]+", " ", filename.rsplit(".", 1)[0]).split()
    if len(words) >= 4:
        words = words[1:]
    stem = " ".join(words)
    return stem if len(stem) <= limit else stem[: limit - 1].rstrip() + "…"


def highlight(page_text: str, passage: str) -> str:
    page_norm, passage_norm = collapse_whitespace(page_text), collapse_whitespace(passage)
    escaped_page = html.escape(page_norm)
    idx = page_norm.find(passage_norm)
    if idx < 0:
        return escaped_page
    return (html.escape(page_norm[:idx]) + "<mark>" + html.escape(passage_norm) + "</mark>"
            + html.escape(page_norm[idx + len(passage_norm):]))


def render_citation_viewer(workspace: str, evidence_map: dict[str, Evidence], state_key: str) -> None:
    label = st.session_state.get(state_key)
    if not label or label not in evidence_map:
        st.caption("Select a citation to open the exact passage and its page.")
        return
    ev = evidence_map[label]
    st.markdown(f"**[{label}] {ev.chunk.source}** · {ev.location} · similarity {ev.chunk.similarity:.3f}")
    st.markdown(f'<div class="passage">{html.escape(ev.chunk.text)}</div>', unsafe_allow_html=True)
    segment = store.get_segment(workspace, ev.chunk.segment_id)
    if segment:
        where = f"Page {segment['page']}" if segment["page"] is not None else (segment["section"] or "Full text")
        st.markdown(f"<p class='muted' style='margin-top:.8rem'>{html.escape(where)} of {html.escape(segment['filename'])} — "
                    "retrieved passage highlighted</p>", unsafe_allow_html=True)
        st.markdown(f'<div class="pagetext">{highlight(segment["content"], ev.chunk.text)}</div>', unsafe_allow_html=True)


def render_claims(claims: list[citations.CheckedClaim], evidence_map: dict[str, Evidence],
                  key_prefix: str, state_key: str) -> None:
    for i, claim in enumerate(claims):
        css = "claim needs_review" if claim.status == citations.NEEDS_REVIEW else "claim"
        flag = f"<span class='flag'>Needs review — {html.escape(claim.reason)}</span>" if claim.status == citations.NEEDS_REVIEW else ""
        st.markdown(f"<div class='{css}'>{CLAIM_ICON.get(claim.status, '')} {html.escape(claim.statement)}{flag}</div>",
                    unsafe_allow_html=True)
        n = len(claim.valid_citations)
        cols = st.columns(n if n >= 2 else 2)
        for j, lab in enumerate(claim.valid_citations):
            ev = evidence_map[lab]
            if cols[j].button(f"[{lab}] {short_name(ev.chunk.source)} {ev.location}", key=f"{key_prefix}-{i}-{lab}",
                              help=f"{ev.chunk.source} · {ev.location}"):
                st.session_state[state_key] = lab


def render_rejected(claims: list[citations.CheckedClaim]) -> None:
    rejected = [c for c in claims if not c.shown]
    if not rejected:
        return
    with st.expander(f"Removed by citation check ({len(rejected)})"):
        st.caption("These model statements were not shown because they did not cite a passage from the retrieved set.")
        for c in rejected:
            st.markdown(f"- ~~{c.statement}~~  \n  <span class='muted'>{html.escape(c.reason)}</span>",
                        unsafe_allow_html=True)


def evidence_table(evidence: list[Evidence]) -> pd.DataFrame:
    return pd.DataFrame([{
        "ID": e.label, "Document": e.chunk.source, "Location": e.location,
        "Similarity": round(e.chunk.similarity, 3), "Chunk": e.chunk.chunk_index, "Chars": len(e.chunk.text),
    } for e in evidence])


def run_metrics(usages: list[Usage], latency: float, model: str) -> None:
    s = summarize(usages)
    c = st.columns(4)
    c[0].metric("Response time", f"{latency:.2f}s")
    c[1].metric("Tokens in / out", f"{s['input_tokens']:,} / {s['output_tokens']:,}")
    c[2].metric("Est. cost", f"${s['cost_usd']:.4f}")
    c[3].metric("Model", model)


# --- sidebar -----------------------------------------------------------------

db_ok, db_msg, _ = _bootstrap()

with st.sidebar:
    st.markdown(f"### AI Insights Copilot\n<span class='muted'>v{__version__} · RAG with grounded retrieval</span>", unsafe_allow_html=True)
    workspace = st.text_input("Workspace", value=config.DEFAULT_WORKSPACE,
                              help="Document set. Retrieval never crosses workspaces.").strip() or config.DEFAULT_WORKSPACE
    reviewer = st.text_input("Reviewer name", value=st.session_state.get("reviewer", ""),
                             help="Recorded on review decisions. Not authenticated in this prototype.")
    st.session_state.reviewer = reviewer
    model_key = st.selectbox("Model", list(config.MODELS), index=list(config.MODELS).index(config.DEFAULT_MODEL),
                             format_func=lambda k: config.MODELS[k].label)
    with st.expander("Retrieval settings"):
        top_k = st.slider("Passages retrieved (k)", 3, 12, config.TOP_K)
        min_sim = st.slider("Relevance threshold", 0.0, 0.6, float(config.MIN_SIMILARITY), 0.01,
                            help="Passages below this cosine similarity are never shown to the model.")
    st.divider()
    st.markdown(f"{'🟢' if db_ok else '🔴'} Postgres · {db_msg}")
    st.markdown(f"{'🟢' if tracing_enabled() else '⚪'} Phoenix tracing"
                + (f" · [open]({config.PHOENIX_ENDPOINT})" if tracing_enabled() else " · off"))
    st.markdown(f"{'🟢' if os.getenv('OPENAI_API_KEY') else '🔴'} OpenAI API key")
    st.divider()
    st.metric("Session spend (est.)", f"${ledger().total_usd:.4f}")
    st.caption(f"{ledger().total_tokens:,} tokens this session · pricing assumptions in config.py")

st.markdown("## AI Insights Copilot")
st.caption("Executive decision support that shows its evidence. Every claim cites a retrieved passage, "
           "every citation is checked, and questions the documents can't answer get “insufficient evidence”.")

if not db_ok:
    st.error(f"Can't reach Postgres/pgvector at DATABASE_URL: {db_msg}. Start it with `docker compose up -d`.")
    st.stop()
if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY is not set. Add it to `.env` — embeddings use OpenAI even when generating locally.")

tab_lib, tab_ask, tab_brief, tab_classic, tab_gov, tab_eval = st.tabs(
    ["Library", "Ask", "Executive brief", "Classic analysis", "Governance", "Evaluation"])


# --- Library -----------------------------------------------------------------

with tab_lib:
    st.markdown("#### Document library")
    st.caption(f"PDFs are chunked page by page; every chunk keeps its document name and page number. "
               f"Limit {config.MAX_UPLOAD_MB:g} MB per file. Use fictional or approved documents only.")
    up_col, sample_col = st.columns([3, 1])
    with up_col:
        uploads = st.file_uploader("Upload documents", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
                                   accept_multiple_files=True, label_visibility="collapsed")
    with sample_col:
        load_samples = st.button("Load ApexSphere sample set", width="stretch",
                                 help="Three fictional documents used in the demo")
        index_clicked = st.button("Index uploaded files", type="primary", width="stretch", disabled=not uploads)

    batch: list[tuple[str, bytes]] = []
    if load_samples:
        batch = [(p.name, p.read_bytes()) for p in sorted(config.SAMPLE_DOCS_DIR.glob("*.pdf"))]
    elif index_clicked and uploads:
        batch = [(f.name, f.getvalue()) for f in uploads]

    if batch:
        rows = []
        with st.status(f"Indexing {len(batch)} file(s)…", expanded=True) as status_box:
            for name, data in batch:
                st.write(f"Chunking and embedding **{name}**")
                try:
                    result = index_file(workspace, name, data, get_embedder())
                except Exception as exc:
                    rows.append({"File": name, "Result": "error", "Pages": None, "Chunks": 0, "Note": str(exc)[:160]})
                    continue
                if result.usage:
                    track([result.usage])
                if result.status == "indexed":
                    audit.record_event(workspace, reviewer, "document_indexed", result.filename,
                                       {"pages": result.pages, "chunks": result.chunks})
                rows.append({"File": result.filename, "Result": result.status, "Pages": result.pages,
                             "Chunks": result.chunks, "Note": result.message})
            status_box.update(label="Indexing complete", state="complete", expanded=False)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    docs = store.list_documents(workspace)
    if not docs:
        st.info(f"No documents in workspace **{workspace}** yet.")
    else:
        st.markdown(f"**{len(docs)} document(s) in `{workspace}`**")
        st.dataframe(pd.DataFrame([{
            "Document": d["filename"], "Type": d["file_type"].upper(), "Pages": d["num_pages"],
            "Chunks": d["chunks"], "Size (KB)": round(d["size_bytes"] / 1024, 1),
            "Indexed": d["uploaded_at"].strftime("%Y-%m-%d %H:%M"),
        } for d in docs]), hide_index=True, width="stretch")

        chunk_col, del_col = st.columns([3, 1])
        with chunk_col:
            pick = st.selectbox("Inspect chunks", docs, format_func=lambda d: d["filename"])
        with del_col:
            st.write("")
            st.write("")
            if st.button("Delete this document", width="stretch"):
                store.delete_document(workspace, pick["id"])
                audit.record_event(workspace, reviewer, "document_deleted", pick["filename"], {})
                st.rerun()
        chunks = store.list_chunks(workspace, pick["id"])
        st.dataframe(pd.DataFrame([{
            "Page": c["page"] if c["page"] is not None else "n/a", "Section": c["section"] or "",
            "Chunk": c["chunk_index"], "Chars": c["chars"], "Text": collapse_whitespace(c["content"])[:220],
        } for c in chunks]), hide_index=True, width="stretch")

        with st.expander("Where is it stored? — chunks, metadata and vectors in pgvector"):
            sql = (f"SELECT d.filename, c.page, c.chunk_index, left(c.content, 60) AS preview,\n"
                   f"       vector_dims(c.embedding) AS dims, (c.embedding::real[])[1:5] AS first_5_dims\n"
                   f"FROM chunks c JOIN documents d ON d.id = c.document_id\n"
                   f"WHERE c.workspace = '{workspace}'\nORDER BY d.filename, c.chunk_index;")
            st.code(sql, language="sql")
            st.dataframe(pd.DataFrame([{
                "filename": c["filename"], "page": c["page"], "chunk_index": c["chunk_index"],
                "preview": collapse_whitespace(c["content"])[:60], "dims": c["dims"],
                "first_5_dims": [round(float(x), 4) for x in c["first_5_dims"]],
            } for c in store.list_chunks(workspace)]), hide_index=True, width="stretch")


# --- Ask ---------------------------------------------------------------------

with tab_ask:
    st.markdown("#### Ask the documents")
    st.caption("Answers are built only from retrieved passages. Click any citation to see the exact passage and page.")
    qcols = st.columns(len(SAMPLE_QUESTIONS))
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if qcols[i].button(q, key=f"sample-q-{i}", width="stretch"):
            st.session_state.ask_question = q
    question = st.text_area("Question", key="ask_question", height=80,
                            placeholder="e.g. How does the remediation budget compare with the engineering estimate?")
    if st.button("Ask", type="primary") and (question or "").strip():
        with st.spinner("Retrieving evidence and generating a grounded answer…"):
            try:
                ans = answer_question(workspace, question.strip(), model_key=model_key, embedder=get_embedder(),
                                      k=top_k, min_similarity=min_sim)
                track(ans.usage)
                st.session_state.answer = ans
                st.session_state.ask_citation = None
            except LLMError as exc:
                st.error(str(exc))

    ans: GroundedAnswer | None = st.session_state.get("answer")
    if ans:
        st.divider()
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown(f"**Q:** {ans.question}")
            st.markdown(status_pill(ans.status), unsafe_allow_html=True)
            if ans.status == citations.INSUFFICIENT_EVIDENCE:
                reason = {
                    "retrieval": "No indexed passage was relevant enough, so the model was not asked to answer.",
                    "model": "The retrieved passages do not contain the answer, so the copilot declined to guess.",
                    "citation_check": "The model produced claims, but none cited the retrieved evidence, so all were removed.",
                }.get(ans.gate, "")
                st.markdown(f"#### Insufficient evidence in the indexed documents\n{reason}")
                if ans.missing_information:
                    st.markdown("**Information that would be needed:**\n" + "\n".join(f"- {m}" for m in ans.missing_information))
            else:
                render_claims(ans.shown_claims, ans.retrieval.by_label, "ask", "ask_citation")
                if ans.missing_information:
                    st.markdown("**Not covered by the documents:** " + "; ".join(ans.missing_information))
            render_rejected(ans.claims)
        with right:
            st.markdown("##### Citation")
            render_citation_viewer(workspace, ans.retrieval.by_label, "ask_citation")
        with st.expander(f"Retrieved evidence ({len(ans.retrieval.evidence)} passages, "
                         f"{len(ans.retrieval.candidates) - len(ans.retrieval.evidence)} below threshold)"):
            if ans.retrieval.evidence:
                st.dataframe(evidence_table(ans.retrieval.evidence), hide_index=True, width="stretch")
        run_metrics(ans.usage, ans.latency_s, ans.model)


# --- Executive brief ---------------------------------------------------------

def submit_review(brief: Brief, decisions: dict[str, tuple[str, str]]) -> None:
    for section, (decision, comment) in decisions.items():
        if decision == "Pending" and not comment:
            continue
        audit.record_event(workspace, reviewer, decision.lower().replace(" ", "_"), f"brief:{section}", {
            "audience": brief.audience, "decision_question": brief.decision, "comment": comment,
            "model": brief.model, "evidence_status": brief.section_status(section),
            "claims": [c.statement for c in brief.sections[section] if c.shown],
            "citations": sorted({lab for c in brief.sections[section] for lab in c.valid_citations}),
        })


with tab_brief:
    st.markdown("#### Executive decision brief")
    b1, b2 = st.columns([1, 2.4])
    audience = b1.selectbox("Audience", AUDIENCES)
    decision = b2.text_input("Decision under review", value=DEFAULT_DECISION)
    cfg = get_audience_config(audience)
    st.caption(f"{cfg.name} view — focus: {cfg.focus}. "
               + (f"Top {cfg.max_items} items per section." if cfg.max_items else "All items shown."))

    def _run_brief(guidance: str | None = None):
        with st.spinner("Retrieving evidence across documents and drafting the brief…"):
            try:
                b = generate_brief(workspace, decision, audience, model_key=model_key, embedder=get_embedder(),
                                   k=max(top_k, 8), min_similarity=min_sim, reviewer_guidance=guidance)
                track(b.usage)
                st.session_state.brief = b
                st.session_state.brief_citation = None
                audit.record_event(workspace, reviewer, "brief_generated", f"brief:{b.audience}", {
                    "decision_question": b.decision, "model": b.model, "evidence_status": b.status,
                    "reviewer_guidance": guidance or "",
                    "rejected_claims": sum(not c.shown for c in b.all_claims)})
            except LLMError as exc:
                st.error(str(exc))

    if st.button("Generate brief", type="primary"):
        _run_brief()

    brief: Brief | None = st.session_state.get("brief")
    if brief:
        bcfg = get_audience_config(brief.audience)
        st.divider()
        head_l, head_r = st.columns([2, 1])
        head_l.markdown(f"**{brief.audience} brief** — {brief.decision}")
        head_r.markdown(status_pill(brief.status, "Decision readiness: "), unsafe_allow_html=True)
        if brief.reviewer_guidance:
            st.info(f"Regenerated with reviewer guidance: “{brief.reviewer_guidance}”")

        if brief.status == citations.INSUFFICIENT_EVIDENCE:
            st.markdown("#### Insufficient evidence for this decision")
            for m in brief.missing_information:
                st.markdown(f"- {m}")
        else:
            body, viewer = st.columns([1.15, 1])
            with body:
                for section in SECTIONS:
                    shown = [c for c in brief.sections[section] if c.shown][: bcfg.max_items or None]
                    st.markdown(f"##### {SECTION_TITLES[section]}  "
                                f"<span class='muted'>· {brief.section_status(section)}</span>", unsafe_allow_html=True)
                    if shown:
                        render_claims(shown, brief.retrieval.by_label, f"brief-{section}", "brief_citation")
                    else:
                        st.caption("No evidence-supported items.")
                if brief.missing_information:
                    st.markdown("**Gaps the documents don't cover:** " + "; ".join(brief.missing_information))
                if bcfg.show_rejected:
                    render_rejected(brief.all_claims)
            with viewer:
                st.markdown("##### Evidence")
                st.caption(f"{len({e.chunk.source for e in brief.retrieval.evidence})} source document(s), "
                           f"{len(brief.retrieval.evidence)} passages")
                render_citation_viewer(workspace, brief.retrieval.by_label, "brief_citation")

            if bcfg.show_retrieval_diagnostics:
                with st.expander("Retrieval diagnostics"):
                    st.markdown("Queries: " + " · ".join(f"`{q}`" for q in brief.retrieval.queries))
                    st.dataframe(evidence_table(brief.retrieval.evidence), hide_index=True, width="stretch")
            if bcfg.show_cost:
                run_metrics(brief.usage, brief.latency_s, brief.model)

            st.divider()
            st.markdown("##### Human review")
            st.caption("The copilot recommends; people decide. Decisions are written to the append-only audit trail.")
            decisions = {}
            rcols = st.columns(len(SECTIONS))
            for i, section in enumerate(SECTIONS):
                with rcols[i]:
                    choice = st.radio(SECTION_TITLES[section], ["Pending", "Approve", "Request refinement"],
                                      key=f"rev-{section}", horizontal=False)
                    note = st.text_area("Comment", key=f"rev-note-{section}", height=80, label_visibility="collapsed",
                                        placeholder="Reviewer comment")
                    decisions[section] = (choice, note.strip())
            r1, r2 = st.columns([1, 1])
            if r1.button("Submit review", type="primary", width="stretch"):
                if not reviewer.strip():
                    st.warning("Enter a reviewer name in the sidebar first.")
                else:
                    submit_review(brief, decisions)
                    st.success("Review recorded in the audit trail.")
            refine = [f"{SECTION_TITLES[s]}: {n or 'refine'}" for s, (d, n) in decisions.items() if d == "Request refinement"]
            if r2.button("Regenerate with reviewer guidance", width="stretch", disabled=not refine):
                _run_brief("; ".join(refine))
                st.rerun()

            export = {
                "audience": brief.audience, "decision": brief.decision, "model": brief.model,
                "decision_readiness": brief.status,
                "sections": {s: [{"statement": c.statement, "status": c.status, "citations": [
                    {"id": lab, "source": brief.retrieval.by_label[lab].chunk.source,
                     "page": brief.retrieval.by_label[lab].chunk.page} for lab in c.valid_citations]}
                    for c in brief.sections[s] if c.shown] for s in SECTIONS},
                "missing_information": brief.missing_information,
                "review": {s: {"decision": d, "comment": n} for s, (d, n) in decisions.items()},
            }
            st.download_button("Export brief (JSON)", json.dumps(export, indent=2),
                               file_name="executive_brief.json", mime="application/json")


# --- Classic analysis --------------------------------------------------------

with tab_classic:
    st.markdown("#### Classic single-document analysis (v1)")
    st.caption("The original mode: the whole document goes to the model in one call. Kept for comparison — "
               "its evaluation scores are model-generated estimates, and its conclusions are not tied to citations.")
    docs = store.list_documents(workspace)
    if not docs:
        st.info("Index a document in the Library first.")
    else:
        c1, c2 = st.columns([2, 1])
        doc_pick = c1.selectbox("Document", docs, format_func=lambda d: d["filename"], key="classic-doc")
        classic_audience = c2.selectbox("Audience", AUDIENCES, key="classic-aud")
        context = st.text_input("Business context (optional)", key="classic-ctx")
        if st.button("Analyze document", type="primary"):
            loaded = store.get_document_text(workspace, doc_pick["id"])
            if loaded:
                with st.spinner("Analyzing…"):
                    try:
                        data, usage, truncated = analyze_document(loaded[1], classic_audience, context, model_key=model_key)
                        track([usage])
                        st.session_state.classic = (loaded[0], classic_audience, data, usage, truncated)
                    except LLMError as exc:
                        st.error(str(exc))
        if st.session_state.get("classic"):
            fname, aud, data, usage, truncated = st.session_state.classic
            ccfg = get_audience_config(aud)
            st.markdown(f"**{fname}** · {aud}")
            if truncated:
                st.warning("Document was truncated to fit the single-call limit — a limitation the grounded modes avoid.")
            st.markdown(data.get("executive_summary", ""))
            for key in ["risks", "opportunities", "recommendations"]:
                items = data.get(key, [])[: ccfg.max_items or None]
                st.markdown(f"**{key.title()}**\n" + "\n".join(f"- {x}" for x in items))
            ev = data.get("evaluation", {})
            st.markdown("**Model self-evaluation** <span class='muted'>(model-generated, not calibrated)</span>",
                        unsafe_allow_html=True)
            e = st.columns(3)
            e[0].metric("Certainty", ev.get("certainty", "–"))
            e[1].metric("Consistency", ev.get("consistency", "–"))
            e[2].metric("PII risk", ev.get("pii_risk", "–"))
            if ev.get("safety_concerns"):
                st.markdown("Safety concerns: " + "; ".join(ev["safety_concerns"]))
            st.caption(f"{usage.input_tokens:,} in / {usage.output_tokens:,} out tokens · ${usage.cost_usd:.4f}")


# --- Governance --------------------------------------------------------------

with tab_gov:
    st.markdown("#### Governance controls")
    g = st.columns(3)
    g[0].markdown("**Grounding**\n\n- Answers use retrieved passages only\n- Relevance threshold before any model call\n"
                  "- Documents treated as untrusted data")
    g[1].markdown("**Verification**\n\n- Citations outside the retrieved set are rejected\n"
                  "- Figures must appear in cited text\n- Evidence status from explicit checks")
    g[2].markdown("**Oversight**\n\n- Approve / refine per section\n- Hash-chained, append-only audit trail\n"
                  "- Workspace-scoped retrieval")

    st.markdown("##### Citation check (runs on every answer)")
    with st.expander("View the code that rejects citations outside the retrieved set"):
        import inspect

        st.code(inspect.getsource(citations.check_claim), language="python")

    st.markdown("##### Review audit trail")
    a1, a2 = st.columns([1, 3])
    if a1.button("Verify hash chain"):
        ok, n, msg = audit.verify_chain()
        (a2.success if ok else a2.error)(msg)
    events = audit.list_events(workspace)
    if events:
        df = pd.DataFrame([{
            "#": e["id"], "When (UTC)": e["created_at"].strftime("%Y-%m-%d %H:%M:%S"), "Reviewer": e["reviewer"],
            "Action": e["action"], "Target": e["target"],
            "Comment": e["record"].get("details", {}).get("comment", ""), "Hash": e["hash"][:12] + "…",
        } for e in events])
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("Export audit trail (JSON)", json.dumps(events, default=str, indent=2),
                           file_name=f"audit_{workspace}.json", mime="application/json")
    else:
        st.caption("No review events yet in this workspace.")
    st.caption("Reviewer names are self-reported in this prototype. Production would bind them to SSO identities.")


# --- Evaluation --------------------------------------------------------------

with tab_eval:
    from copilot.evaluation import load_latest_report, run_evaluation

    st.markdown("#### Evaluation")
    st.caption(f"{len(json.loads(config.EVAL_QUESTIONS_PATH.read_text())['questions'])} fixed questions against the "
               f"ApexSphere set in workspace `{config.EVAL_WORKSPACE}`: retrieval recall, faithfulness "
               f"(LLM judge: {config.JUDGE_MODEL}), and whether the copilot abstains on unanswerable questions.")
    if st.button("Run evaluation", type="primary"):
        bar = st.progress(0.0, text="Starting…")
        try:
            report = run_evaluation(model_key=model_key, embedder=get_embedder(), k=top_k, min_similarity=min_sim,
                                    progress=lambda i, n, q: bar.progress((i - 1) / n, text=f"{i}/{n}: {q}"))
            bar.progress(1.0, text="Done")
        except LLMError as exc:
            st.error(str(exc))

    report = load_latest_report()
    if report:
        s = report["summary"]
        m = st.columns(5)
        m[0].metric("Retrieval recall", f"{s['retrieval_recall']:.0%}" if s["retrieval_recall"] is not None else "–")
        m[1].metric("Faithfulness", f"{s['faithfulness']:.0%}" if s["faithfulness"] is not None else "–")
        m[2].metric("Answered (answerable)", s["answered_when_answerable"])
        m[3].metric("Abstained (unanswerable)", s["abstained_when_unanswerable"])
        m[4].metric("Run cost", f"${s['total_cost_usd']:.3f}")
        st.caption(f"Run {report['run_at']} · model {report['model']} · k={report['top_k']} · "
                   f"threshold {report['min_similarity']} · {s['claims_rejected_by_citation_check']} claim(s) removed "
                   f"by citation check · {s['claims_flagged_for_review']} flagged for review")
        st.dataframe(pd.DataFrame([{
            "ID": c["id"], "Question": c["question"], "Answerable": "yes" if c["answerable"] else "no",
            "Result": c["status"], "Correct behavior": "✓" if c["abstention_correct"] else "✗",
            "Retrieval recall": "–" if c["retrieval_recall"] is None else f"{c['retrieval_recall']:.0%}",
            "Faithfulness": "–" if c["faithfulness"] is None else f"{c['faithfulness']:.0%}",
            "Latency (s)": c["latency_s"],
        } for c in report["cases"]]), hide_index=True, width="stretch")
        with st.expander("Answers and judge notes"):
            for c in report["cases"]:
                st.markdown(f"**{c['id']}. {c['question']}** — {c['status']}")
                for a in c["answer"]:
                    st.markdown(f"- {a}")
                for n in c["judge_notes"]:
                    st.markdown(f"  - <span class='muted'>Judge: {html.escape(n)}</span>", unsafe_allow_html=True)
    else:
        st.info("No evaluation run yet. Click **Run evaluation**, or run `python evals/run_eval.py`.")
