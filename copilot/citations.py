"""Deterministic citation check — runs on every model answer before anyone sees it.

Rule 1. A citation is valid only if it names a passage that was actually
        retrieved for this request. Invented or out-of-set IDs are rejected.
Rule 2. A claim with no valid citation is removed from the answer.
Rule 3. Every figure in a claim must appear in the passages it cites;
        otherwise the claim is flagged for human review.

No model is involved in this check, so it cannot be talked out of a decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

VERIFIED = "verified"
NEEDS_REVIEW = "needs_review"
REJECTED = "rejected"

EVIDENCE_AVAILABLE = "Evidence available"
PARTIAL_EVIDENCE = "Partial evidence"
INSUFFICIENT_EVIDENCE = "Insufficient evidence"

_LABEL_RE = re.compile(r"\[?\s*(S\d+)\s*\]?", re.IGNORECASE)
_INLINE_LABEL_RE = re.compile(r"\[\s*S\d+(?:\s*,\s*S\d+)*\s*\]", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class CheckedClaim:
    statement: str
    cited: list[str]
    valid_citations: list[str]
    invalid_citations: list[str]
    unmatched_figures: list[str] = field(default_factory=list)
    status: str = VERIFIED
    reason: str = ""

    @property
    def shown(self) -> bool:
        return self.status != REJECTED


def normalize_label(raw: str) -> str:
    match = _LABEL_RE.fullmatch(str(raw).strip())
    return match.group(1).upper() if match else str(raw).strip()


def extract_figures(text: str) -> set[str]:
    text = _INLINE_LABEL_RE.sub(" ", text)
    figures = set()
    for raw in _NUMBER_RE.findall(text):
        value = raw.replace(",", "").rstrip(".")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        figures.add(value)
    return figures


def check_claim(claim: Mapping, retrieved: Mapping[str, str]) -> CheckedClaim:
    """retrieved maps evidence label (e.g. 'S3') -> passage text for THIS request only."""
    statement = str(claim.get("statement", "")).strip()
    cited: list[str] = []
    for raw in claim.get("citations") or []:
        label = normalize_label(raw)
        if label not in cited:
            cited.append(label)

    valid = [c for c in cited if c in retrieved]
    invalid = [c for c in cited if c not in retrieved]
    result = CheckedClaim(statement, cited, valid, invalid)

    if not valid:
        result.status = REJECTED
        result.reason = (
            "No citation provided." if not cited
            else f"Cited {', '.join(invalid)}, which {'is' if len(invalid) == 1 else 'are'} not in the retrieved evidence."
        )
        return result

    cited_text = " ".join(retrieved[c] for c in valid)
    missing = sorted(extract_figures(statement) - extract_figures(cited_text))
    if missing:
        result.unmatched_figures = missing
        result.status = NEEDS_REVIEW
        result.reason = f"Figure(s) {', '.join(missing)} not found in the cited passage(s)."
    elif invalid:
        result.reason = f"Dropped invalid citation(s): {', '.join(invalid)}."
    return result


def check_claims(claims: Iterable[Mapping], retrieved: Mapping[str, str]) -> list[CheckedClaim]:
    return [check_claim(c, retrieved) for c in claims if str(c.get("statement", "")).strip()]


def evidence_status(checked: list[CheckedClaim], model_status: str) -> str:
    """Status shown to executives — derived from explicit checks, not a model confidence score."""
    shown = [c for c in checked if c.shown]
    if model_status == "insufficient_evidence" or not shown:
        return INSUFFICIENT_EVIDENCE
    # Rejected claims are already removed from view; flagged figures downgrade the status.
    if model_status == "partial" or any(c.status != VERIFIED for c in shown):
        return PARTIAL_EVIDENCE
    return EVIDENCE_AVAILABLE
