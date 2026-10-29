"""Audience-specific configuration: what each decision-maker sees and what retrieval emphasizes."""

from __future__ import annotations

from dataclasses import dataclass

AUDIENCES = ["CEO / Board", "CTO", "COO", "CFO", "Analyst / Practitioner"]


@dataclass(frozen=True)
class AudienceConfig:
    name: str
    focus: str
    retrieval_hint: str  # added as a second retrieval query to pull role-relevant passages
    max_items: int | None
    show_passages: bool
    show_retrieval_diagnostics: bool
    show_cost: bool
    show_rejected: bool


_CONFIGS = {
    "CEO / Board": AudienceConfig(
        "CEO / Board",
        "strategic decisions, enterprise exposure, and what must be decided now",
        "strategic risk enterprise exposure growth decision",
        3, False, False, False, False,
    ),
    "CTO": AudienceConfig(
        "CTO",
        "technical risk, infrastructure reliability, architecture and remediation feasibility",
        "infrastructure reliability technical failure power cooling remediation",
        5, True, True, False, True,
    ),
    "COO": AudienceConfig(
        "COO",
        "execution, operational resilience, dependencies, staffing and schedule",
        "operations incidents staffing schedule resilience dependencies",
        5, True, False, False, False,
    ),
    "CFO": AudienceConfig(
        "CFO",
        "cost, capital allocation, financial exposure, payback and ROI",
        "capex budget cost revenue payback financial exposure leverage",
        5, True, False, True, False,
    ),
    "Analyst / Practitioner": AudienceConfig(
        "Analyst / Practitioner",
        "detailed examination and validation of every finding",
        "detailed evidence figures assumptions",
        None, True, True, True, True,
    ),
}


def get_audience_config(audience: str) -> AudienceConfig:
    return _CONFIGS.get(audience, _CONFIGS["Analyst / Practitioner"])
