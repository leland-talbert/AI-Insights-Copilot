"""Generate the three fictional ApexSphere PDFs used in the demo and the evaluation suite.

All companies, people, sites and figures are invented. Page content is fixed so
evals/questions.json can name the exact page that answers each question.

    python scripts/make_sample_docs.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT_DIR = Path(__file__).resolve().parent.parent / "sample_docs"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=17, spaceAfter=4, textColor=colors.HexColor("#14213d"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#14213d"))
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10.2, leading=14.2, spaceAfter=6)
META = ParagraphStyle("Meta", parent=BODY, fontSize=8.5, textColor=colors.HexColor("#6b7280"))

DOCS = {
    "ApexSphere_Q1_Data_Center_Operations_Report.pdf": {
        "title": "ApexSphere Q1 Data Center Operations Report",
        "meta": "Prepared by Site Operations · Reporting period: January–March · Classification: Internal (fictional)",
        "pages": [
            ("1. Executive summary", [
                "The Atlanta campus (ATL-1) ran at an average utilization of 87% of critical IT capacity during Q1, "
                "well above the 75% operating target set by the Infrastructure Committee. Sustained operation above "
                "target reduces the headroom available for maintenance windows and failover.",
                "Availability remained strong at 99.982% across all customer-facing halls. Power usage effectiveness "
                "(PUE) averaged 1.48 against a target of 1.35, driven mainly by the age of the ATL-1 cooling plant.",
                "Site Operations recommends that leadership treat ATL-1 capacity and resilience as a single decision: "
                "new demand cannot be absorbed safely until headroom is restored.",
            ], [["Metric", "Q1 actual", "Target"], ["ATL-1 utilization", "87%", "75%"],
                ["Availability", "99.982%", "99.98%"], ["PUE", "1.48", "1.35"]]),
            ("2. Capacity and demand", [
                "ATL-1 has 18 MW of critical IT capacity, of which 15.7 MW was in use at quarter end. The Dallas campus "
                "(DAL-2) ran at 61% utilization and has 4.1 MW of unsold capacity, but most Atlanta customers require "
                "low-latency connectivity to the Southeast metro and cannot be relocated to Dallas.",
                "The sales team holds a customer waitlist of 2.4 MW for ATL-1, including two hyperscale tenants. "
                "Without additional capacity, roughly 1.1 MW of that demand is expected to go to competitors within "
                "two quarters.",
            ], None),
            ("3. Incidents and staffing", [
                "There were three Severity-2 incidents in Q1 and no Severity-1 incidents. The most significant occurred "
                "on February 11, when UPS Bank B at ATL-1 failed over unexpectedly and Hall 3 experienced 14 minutes of "
                "degraded cooling. No customer workloads went offline, but two tenants filed formal inquiries.",
                "Mean time to restore (MTTR) across all incidents was 3.2 hours. ATL-1 currently has 6 open critical "
                "facilities technician positions, and overtime for the facilities team rose 22% compared with Q4.",
            ], None),
            ("4. Cooling and sustainability", [
                "Chiller Plant 2 at ATL-1 was installed in 2009 and has reached the end of its rated service life. "
                "Its failure would remove about 30% of the campus's cooling capacity during summer peaks.",
                "Water usage effectiveness (WUE) averaged 1.9 liters per kWh. The sustainability team expects that "
                "replacing Chiller Plant 2 would lower ATL-1 PUE to about 1.38.",
            ], None),
        ],
    },
    "ApexSphere_Q2_Financial_and_Expansion_Strategy.pdf": {
        "title": "ApexSphere Q2 Financial and Expansion Strategy",
        "meta": "Prepared by the Office of the CFO · Board pre-read · Classification: Confidential (fictional)",
        "pages": [
            ("1. Financial performance", [
                "ApexSphere reported Q2 revenue of $142.6 million, up 11% year over year, with an adjusted EBITDA "
                "margin of 38%. Colocation revenue grew faster than interconnection revenue, reflecting strong demand "
                "for high-density capacity in the Southeast.",
                "Customer churn remained low at 1.2% of monthly recurring revenue.",
            ], [["Metric", "Q2"], ["Revenue", "$142.6M"], ["YoY growth", "11%"], ["Adj. EBITDA margin", "38%"]]),
            ("2. Atlanta expansion proposal (ATL-2)", [
                "Management proposes building ATL-2, a new Atlanta facility adjacent to ATL-1. Phase 1 would add "
                "24 MW of critical IT capacity at an estimated capital cost of $310 million, with a target in-service "
                "date of Q3 2028.",
                "Customers have already committed to pre-lease 40% of Phase 1 capacity. At full lease-up, management "
                "projects a payback period of 6.5 years.",
            ], None),
            ("3. Capital allocation", [
                "The proposed 2027 capital budget is $420 million. It includes a $38 million infrastructure remediation "
                "allocation for ATL-1, which covers the engineering plan's estimate plus contingency.",
                "Net leverage is currently 3.9x against a debt covenant maximum of 4.5x. Funding ATL-2 Phase 1 and the "
                "remediation program in the same year would bring projected net leverage to 4.3x at peak.",
            ], None),
            ("4. Risks to the plan", [
                "The utility, Peachtree Regional Power, has quoted a 30-month lead time for the new ATL-2 substation "
                "interconnect, which makes the Q3 2028 date dependent on an early commitment.",
                "Construction cost inflation is running at about 7% a year, and any slip in the schedule increases the "
                "capital estimate. A serious outage at ATL-1 during the expansion could also put the 40% pre-lease "
                "commitments at risk.",
            ], None),
        ],
    },
    "ApexSphere_Infrastructure_Risk_and_Resilience_Assessment.pdf": {
        "title": "ApexSphere Infrastructure Risk and Resilience Assessment",
        "meta": "Prepared by Engineering & Risk · Scope: ATL-1 · Classification: Internal (fictional)",
        "pages": [
            ("1. Risk summary", [
                "The overall risk rating for ATL-1 power infrastructure is High. UPS Bank B batteries are at 82% of "
                "their rated service life and were the root cause of the February 11 failover.",
                "Hall 3 is served by a single utility feed, so a utility-side fault would leave it on generator power "
                "with no redundant grid path. Combined with the end-of-life Chiller Plant 2, these issues create "
                "correlated failure modes during summer peak load.",
            ], [["Finding", "Rating"], ["UPS Bank B battery life", "High"],
                ["Hall 3 single utility feed", "High"], ["Chiller Plant 2 end of life", "Medium-High"]]),
            ("2. Remediation plan", [
                "Engineering recommends three projects: replace the UPS Bank B batteries ($6.2 million, about 4 months); "
                "add a second utility feed to Hall 3 ($21.0 million, about 14 months); and replace Chiller Plant 2 "
                "($9.5 million, about 9 months).",
                "The total engineering estimate is $36.7 million before contingency. The battery replacement can begin "
                "immediately; the second feed depends on the same utility engineering team as any new substation work.",
            ], None),
            ("3. Exposure if remediation is deferred", [
                "Engineering estimates an 18% probability of a site-wide outage at ATL-1 over the next 24 months if "
                "remediation is deferred, falling to 4% once all three projects are complete.",
                "The estimated cost of a site-wide outage is $2.1 million per hour, including SLA credits and emergency "
                "response, before any reputational impact. Engineering advises sequencing remediation ahead of, or in "
                "parallel with, any major expansion that draws on the same utility and construction resources.",
            ], None),
        ],
    },
}


def _footer(title: str):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawString(0.9 * inch, 0.6 * inch, f"{title} — fictional scenario for demonstration only")
        canvas.drawRightString(LETTER[0] - 0.9 * inch, 0.6 * inch, f"Page {doc.page}")
        canvas.restoreState()
    return draw


def build(filename: str, spec: dict) -> Path:
    path = OUT_DIR / filename
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.9 * inch, bottomMargin=0.9 * inch, title=spec["title"],
                            author="ApexSphere (fictional)", subject="AI Insights Copilot demo document")
    story = []
    for i, (heading, paragraphs, table) in enumerate(spec["pages"]):
        if i == 0:
            story += [Paragraph(spec["title"], H1), Paragraph(spec["meta"], META), Spacer(1, 10)]
        story.append(Paragraph(heading, H2))
        story += [Paragraph(p, BODY) for p in paragraphs]
        if table:
            t = Table(table, hAlign="LEFT", colWidths=[2.4 * inch] + [1.3 * inch] * (len(table[0]) - 1))
            t.setStyle(TableStyle([
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9.5),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 9.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c4ccd8")),
                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story += [Spacer(1, 6), t]
        if i < len(spec["pages"]) - 1:
            story.append(PageBreak())
    doc.build(story, onFirstPage=_footer(spec["title"]), onLaterPages=_footer(spec["title"]))
    return path


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    for name, spec in DOCS.items():
        print("wrote", build(name, spec))
