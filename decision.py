"""
Decision Module — Phase 3

Classifies each restaurant into:
  - HIGH_CONFIDENCE: auto-populate CSV (legal name + PR link)
  - NEEDS_REVIEW: borderline, output for manual review
  - NO_MATCH: leave blank

Uses multi-threshold approach calibrated on labeled rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from models import CandidatePreview, ScoredCandidate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — calibrated from first 116 rows (37 labeled + 79 confirmed no-match)
# Target: <5% false positive rate
# ---------------------------------------------------------------------------
T_SCORE_HIGH = 78.0
T_MARGIN_HIGH = 10.0
T_NAME_HIGH = 70.0
T_ADDR_HIGH = 50.0
T_COVERAGE_HIGH = 0.80

T_SCORE_REVIEW = 45.0
T_MARGIN_REVIEW = 2.0

# ---------------------------------------------------------------------------
# Match statuses
# ---------------------------------------------------------------------------
HIGH_CONFIDENCE = "match"
NEEDS_REVIEW = "review"
NO_MATCH = "no_match"

MatchStatus = Literal["match", "review", "no_match"]


@dataclass
class MatchResult:
    """Result of the decision layer for one restaurant."""

    restaurant_name: str
    status: MatchStatus
    legal_name: str = ""
    registration_index: str = ""
    pr_link: str = ""
    confidence: float = 0.0
    margin: float = 0.0
    name_score: float = 0.0
    addr_score: float = 0.0
    reason: str = ""
    match_source: str = "deterministic"
    top_candidates: list[dict[str, Any]] = field(default_factory=list)


def _build_pr_link(registration_index: str) -> str:
    if not registration_index:
        return ""
    return f"https://rcp.estado.pr.gov/en/file?c={registration_index}"


def decide_from_llm(
    restaurant_row: Mapping[str, Any],
    ranked_candidates: Sequence[ScoredCandidate],
    llm_result: Mapping[str, Any],
) -> MatchResult:
    """Build MatchResult from an LLM match decision."""
    name = str(restaurant_row.get("Name") or "")
    idx_any = llm_result.get("match_index")
    confidence = llm_result.get("confidence")
    reason = str(llm_result.get("reason") or "")
    source = str(llm_result.get("source") or "llm")

    if idx_any is None or confidence not in ("high",):
        return MatchResult(
            restaurant_name=name,
            status=NO_MATCH,
            reason=f"LLM: no match — {reason}",
            match_source=source,
        )

    idx = int(idx_any)
    if idx < 0 or idx >= len(ranked_candidates):
        return MatchResult(
            restaurant_name=name,
            status=NO_MATCH,
            reason=f"LLM: invalid index {idx}",
            match_source=source,
        )

    matched = ranked_candidates[idx]

    return MatchResult(
        restaurant_name=name,
        status=HIGH_CONFIDENCE,
        legal_name=matched.corp_name,
        registration_index=matched.registration_index,
        pr_link=_build_pr_link(matched.registration_index),
        confidence=round(matched.final_score, 1),
        margin=0.0,
        name_score=round(matched.name_scores.combined, 1),
        addr_score=round(matched.addr_scores.combined, 1),
        reason=f"LLM ({confidence} confidence): {reason}",
        match_source=source,
        top_candidates=[],
    )


def decide(
    restaurant_row: Mapping[str, Any],
    ranked_candidates: Sequence[ScoredCandidate],
) -> MatchResult:
    """
    Apply thresholds to ranked candidates and classify.

    ranked_candidates: output of scorer.rank_candidates() — sorted desc by final_score.
    """
    name = str(restaurant_row.get("Name") or "")

    if not ranked_candidates:
        return MatchResult(
            restaurant_name=name,
            status=NO_MATCH,
            reason="no candidates found",
        )

    top = ranked_candidates[0]
    second_score = ranked_candidates[1].final_score if len(ranked_candidates) > 1 else 0.0
    margin = top.final_score - second_score
    name_s = top.name_scores.combined
    addr_s = top.addr_scores.combined
    coverage = top.name_scores.distinctive_coverage

    top_cands = [
        CandidatePreview(
            corp_name=c.corp_name,
            registration_index=c.registration_index,
            final_score=round(c.final_score, 1),
            name_score=round(c.name_scores.combined, 1),
            addr_score=round(c.addr_scores.combined, 1),
            status=c.status,
        ).to_dict()
        for c in ranked_candidates[:3]
    ]

    base = MatchResult(
        restaurant_name=name,
        status=NO_MATCH,
        legal_name=top.corp_name,
        registration_index=top.registration_index,
        pr_link=_build_pr_link(top.registration_index),
        confidence=round(top.final_score, 1),
        margin=round(margin, 1),
        name_score=round(name_s, 1),
        addr_score=round(addr_s, 1),
        top_candidates=top_cands,
    )

    is_active = top.aux_scores.is_active

    if (
        top.final_score >= T_SCORE_HIGH
        and margin >= T_MARGIN_HIGH
        and name_s >= T_NAME_HIGH
        and addr_s >= T_ADDR_HIGH
        and coverage >= T_COVERAGE_HIGH
        and is_active
    ):
        base.status = HIGH_CONFIDENCE
        base.reason = "all gates pass"
        return base

    if top.final_score >= T_SCORE_REVIEW and margin >= T_MARGIN_REVIEW:
        base.status = NEEDS_REVIEW
        reasons: list[str] = []
        if top.final_score < T_SCORE_HIGH:
            reasons.append(f"score {top.final_score:.1f} < {T_SCORE_HIGH}")
        if margin < T_MARGIN_HIGH:
            reasons.append(f"margin {margin:.1f} < {T_MARGIN_HIGH}")
        if name_s < T_NAME_HIGH:
            reasons.append(f"name {name_s:.1f} < {T_NAME_HIGH}")
        if addr_s < T_ADDR_HIGH:
            reasons.append(f"addr {addr_s:.1f} < {T_ADDR_HIGH}")
        if coverage < T_COVERAGE_HIGH:
            reasons.append(f"coverage {coverage:.2f} < {T_COVERAGE_HIGH}")
        if not is_active:
            reasons.append(f"not active ({top.status})")
        base.reason = "; ".join(reasons) if reasons else "borderline"
        return base

    reasons2: list[str] = []
    if top.final_score < T_SCORE_REVIEW:
        reasons2.append(f"score {top.final_score:.1f} < {T_SCORE_REVIEW}")
    if margin < T_MARGIN_REVIEW:
        reasons2.append(f"margin {margin:.1f} < {T_MARGIN_REVIEW}")
    base.reason = "; ".join(reasons2) if reasons2 else "below all thresholds"
    base.legal_name = ""
    base.pr_link = ""
    return base


def classify_batch(results: list[dict[str, Any]]) -> list[MatchResult]:
    """
    Apply ``decide()`` to a list of ``process_batch()`` output dicts.

    Each element of ``results`` must have keys ``"restaurant"`` (the original
    CSV row dict) and optionally ``"ranked"`` (a list of ``ScoredCandidate``).
    Returns one ``MatchResult`` per input element, in the same order.
    """
    decisions: list[MatchResult] = []
    for res in results:
        row = res["restaurant"]
        ranked = res.get("ranked", [])
        decisions.append(decide(row, ranked))
    return decisions


def print_summary(decisions: Sequence[MatchResult]) -> None:
    """Log a concise summary of match/review/no_match counts to the root logger."""
    total = len(decisions)
    if total == 0:
        logger.info("No decisions to summarize.")
        return
    matched = sum(1 for d in decisions if d.status == HIGH_CONFIDENCE)
    review = sum(1 for d in decisions if d.status == NEEDS_REVIEW)
    no_match = sum(1 for d in decisions if d.status == NO_MATCH)

    logger.info("=" * 60)
    logger.info("DECISION SUMMARY")
    logger.info("=" * 60)
    logger.info("Total rows:      %s", total)
    logger.info("High confidence: %4d (%.1f%%)", matched, matched / total * 100)
    logger.info("Needs review:    %4d (%.1f%%)", review, review / total * 100)
    logger.info("No match:        %4d (%.1f%%)", no_match, no_match / total * 100)
    logger.info("=" * 60)
