"""
Shared data models for the PR restaurant ↔ legal-entity matching pipeline.

Using dataclasses here gives:
- Editor / mypy-friendly attribute access
- Clear shapes for scores and candidates (fewer KeyError surprises)
- Single place to document fields
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NameScores:
    token_sort: float
    token_set: float
    partial: float
    wratio: float
    token_overlap: float
    distinctive_coverage: float
    combined: float
    r_clean: str = ""
    c_clean: str = ""


@dataclass
class AddrScores:
    city_match: float
    zip_match: float
    street_score: float
    combined: float
    r_addr: str = ""
    e_addr: str = ""


@dataclass
class AuxScores:
    status_score: float
    purpose_score: float
    is_active: bool
    combined: float


@dataclass
class ScoredCandidate:
    """
    One registry candidate scored against a restaurant row.

    ``detail`` holds the raw PR ``/info`` payload when available — used for
    LLM prompts and debugging; keep optional to avoid huge logs by default.
    """

    corp_name: str
    registration_index: str
    status: str
    final_score: float
    name_scores: NameScores
    addr_scores: AddrScores
    aux_scores: AuxScores
    detail: dict[str, Any] | None = None

    def summary_dict(self) -> dict[str, Any]:
        """Compact dict for review CSV / logging."""
        return {
            "corp_name": self.corp_name,
            "registration_index": self.registration_index,
            "final_score": round(self.final_score, 1),
            "name_score": round(self.name_scores.combined, 1),
            "addr_score": round(self.addr_scores.combined, 1),
            "status": self.status,
        }


@dataclass
class CandidatePreview:
    """Top-k candidate snapshot attached to MatchResult for human review."""

    corp_name: str
    registration_index: str
    final_score: float
    name_score: float
    addr_score: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "corp_name": self.corp_name,
            "registration_index": self.registration_index,
            "final_score": self.final_score,
            "name_score": self.name_score,
            "addr_score": self.addr_score,
            "status": self.status,
        }
