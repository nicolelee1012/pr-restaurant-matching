"""
Holdout evaluation: Puerto_Rico_hidden_set.csv

Labeled rows   (Legal Name filled in) — pipeline should recover the correct entity.
Unlabeled rows (Legal Name blank)     — human confirmed no good match exists;
                                        pipeline must NOT auto-match these.

Runs the full pipeline: deterministic scoring + LLM fallback.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import re
from pathlib import Path

from decision import HIGH_CONFIDENCE, NEEDS_REVIEW, MatchResult, decide, decide_from_llm
from llm_matcher import llm_match_batch
from pr_registry import process_batch
from scorer import rank_candidates

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
HOLDOUT_PATH = _HERE / "Puerto_Rico_hidden_set.csv"


def _normalize(s: str) -> str:
    """Collapse whitespace and upper-case for comparison."""
    return re.sub(r"\s+", " ", s.strip()).upper()


async def main() -> None:
    with open(HOLDOUT_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    labeled = [r for r in rows if r.get("Legal Name", "").strip()]
    unlabeled = [r for r in rows if not r.get("Legal Name", "").strip()]
    logger.info(
        "Holdout set: %s total — %s labeled, %s unlabeled (confirmed no match)",
        len(rows), len(labeled), len(unlabeled),
    )

    results = await process_batch(rows, fetch_details=True)

    tp = fp_wrong = fp_spurious = tn = 0
    llm_inputs: list[tuple[dict, list]] = []
    llm_meta: list[tuple[bool, str | None, str]] = []  # (is_labeled, true_legal, name)
    issues: list[str] = []

    # ── Deterministic pass ───────────────────────────────────────────────────
    det_decisions: list[tuple[dict, list, MatchResult]] = []

    for res in results:
        row = res["restaurant"]
        ranked = rank_candidates(row, res["candidates"]) if res["candidates"] else []
        d = decide(row, ranked)
        det_decisions.append((row, ranked, d))

        is_labeled = bool(row.get("Legal Name", "").strip())
        true_legal = _normalize(row.get("Legal Name", "")) if is_labeled else None
        name = row["Name"]

        if d.status == HIGH_CONFIDENCE:
            if is_labeled:
                if _normalize(d.legal_name) == true_legal:
                    tp += 1
                else:
                    fp_wrong += 1
                    issues.append(
                        f"DET FP (wrong):    {name[:45]:<47} "
                        f"got={d.legal_name[:35]:<37} expected={true_legal}"
                    )
            else:
                fp_spurious += 1
                issues.append(
                    f"DET FP (spurious): {name[:45]:<47} got={d.legal_name[:35]}"
                )
        else:
            if not is_labeled:
                tn += 1
            elif ranked:
                llm_inputs.append((row, ranked))
                llm_meta.append((is_labeled, true_legal, name))
            else:
                issues.append(f"MISS (no candidates): {name}")

    logger.info(
        "Deterministic: %s TP, %s FP → %s sent to LLM",
        tp, fp_wrong + fp_spurious, len(llm_inputs),
    )

    # ── LLM fallback ────────────────────────────────────────────────────────
    if llm_inputs:
        llm_results = await llm_match_batch(llm_inputs)

        for (row, ranked), llm_result, (is_labeled, true_legal, name) in zip(
            llm_inputs, llm_results, llm_meta
        ):
            llm_d = decide_from_llm(row, ranked, llm_result)
            if llm_d.status == HIGH_CONFIDENCE:
                if is_labeled:
                    if _normalize(llm_d.legal_name) == true_legal:
                        tp += 1
                    else:
                        fp_wrong += 1
                        issues.append(
                            f"LLM FP (wrong):    {name[:45]:<47} "
                            f"got={llm_d.legal_name[:35]:<37} expected={true_legal}"
                        )
                else:
                    fp_spurious += 1
                    issues.append(
                        f"LLM FP (spurious): {name[:45]:<47} got={llm_d.legal_name[:35]}"
                    )
            else:
                if not is_labeled:
                    tn += 1
                else:
                    issues.append(f"MISS: {name:<47} expected={true_legal}")

    # ── Summary ──────────────────────────────────────────────────────────────
    total_fp = fp_wrong + fp_spurious
    total_matched = tp + total_fp
    total_labeled = len(labeled)
    total_unlabeled = len(unlabeled)

    print("=" * 65)
    print("GOLDEN SET RESULTS — Deterministic + LLM")
    print("=" * 65)
    print(f"Total rows:        {len(rows)}  ({total_labeled} labeled, {total_unlabeled} no-match)")
    print(f"True Positives:    {tp} / {total_labeled}")
    print(f"False Positives:   {total_fp}  (wrong_match={fp_wrong}, spurious={fp_spurious})")
    print(f"True Negatives:    {tn} / {total_unlabeled}")
    print()
    print(f"Recall:            {tp / total_labeled * 100:.1f}%")
    if total_matched:
        print(f"Precision:         {tp / total_matched * 100:.1f}%")
    print(f"FP rate:           {total_fp / len(rows) * 100:.1f}%  (target ≤5%)")
    print("=" * 65)

    if issues:
        print(f"\nDETAILS ({len(issues)} issues):")
        for issue in issues:
            print(f"  {issue}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
