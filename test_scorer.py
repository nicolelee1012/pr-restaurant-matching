"""Test scorer on labeled rows — deterministic pass + optional LLM fallback."""
from __future__ import annotations

import asyncio
import os

from decision import HIGH_CONFIDENCE, decide_from_llm
from models import RestaurantRow, ScoredCandidate
from pr_registry import load_restaurants, process_batch
from scorer import rank_candidates

_USE_LLM = bool(os.environ.get("OPENAI_API_KEY"))

if _USE_LLM:
    try:
        from llm_matcher import llm_match_batch
    except ImportError:
        _USE_LLM = False


def _is_match(corp_name: str, reg_idx: str, expected_legal: str, expected_idx: str) -> bool:
    return (
        (expected_idx and reg_idx == expected_idx)
        or corp_name.upper() == expected_legal
    )


async def main():
    restaurants = load_restaurants()
    labeled = [r for r in restaurants if r.legal_name]

    print(f"Processing {len(labeled)} labeled rows...\n")
    results = await process_batch(labeled, fetch_details=True)

    correct_at_1 = 0
    correct_in_top3 = 0
    total_with_candidates = 0
    total = len(results)

    # Per-row state for LLM pass
    ranked_map: dict[int, list[ScoredCandidate]] = {}
    row_map: dict[int, RestaurantRow] = {}
    expected_map: dict[int, tuple[str, str]] = {}
    det_correct: dict[int, bool] = {}
    failures: list = []

    for i, res in enumerate(results):
        row = res.restaurant
        name = row.name
        expected_legal = row.legal_name.upper()
        expected_idx = row.pr_link.split("?c=")[-1] if "?c=" in row.pr_link else ""
        expected_map[i] = (expected_legal, expected_idx)
        row_map[i] = row

        if not res.candidates:
            failures.append((name, expected_legal, "NO CANDIDATES", 0, []))
            det_correct[i] = False
            continue

        total_with_candidates += 1
        ranked = rank_candidates(row, res.candidates)
        ranked_map[i] = ranked

        top = ranked[0]
        is_top1 = _is_match(top.corp_name, top.registration_index, expected_legal, expected_idx)
        is_top3 = any(
            _is_match(r.corp_name, r.registration_index, expected_legal, expected_idx)
            for r in ranked[:3]
        )

        if is_top1:
            correct_at_1 += 1
        if is_top3:
            correct_in_top3 += 1

        det_correct[i] = is_top1

        if not is_top1:
            actual_rank = next(
                (j + 1 for j, r in enumerate(ranked)
                 if _is_match(r.corp_name, r.registration_index, expected_legal, expected_idx)),
                None,
            )
            failures.append(
                (
                    name,
                    expected_legal,
                    f"Ranked #{actual_rank}" if actual_rank else "NOT IN CANDIDATES",
                    top.final_score,
                    [
                        (r.corp_name, f"{r.final_score:.1f}", r.name_scores.combined, r.addr_scores.combined)
                        for r in ranked[:5]
                    ],
                )
            )
        else:
            margin = top.final_score - ranked[1].final_score if len(ranked) > 1 else 999
            print(
                f"[OK] {name:45s} -> {top.corp_name:45s}  score={top.final_score:.1f}  margin={margin:.1f}"
            )

    print(f"\n{'='*80}")
    print("DETERMINISTIC RESULTS (scorer only):")
    print(f"  Correct at #1:     {correct_at_1}/{total} ({correct_at_1/total*100:.1f}%)")
    print(f"  Correct in top 3:  {correct_in_top3}/{total} ({correct_in_top3/total*100:.1f}%)")
    print(f"  With candidates:   {total_with_candidates}/{total}")
    print(f"{'='*80}")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for name, expected, status, top_score, top5 in failures:
            print(f"\n  {name}")
            print(f"    Expected: {expected}")
            print(f"    Status: {status}")
            if top5:
                print("    Top candidates:")
                for cname, score, nscore, ascore in top5:
                    print(f"      {cname:45s} final={score} name={nscore:.1f} addr={ascore:.1f}")

    # ── LLM fallback pass ─────────────────────────────────────────────────────
    if not _USE_LLM:
        print("\n(Set OPENAI_API_KEY to also run LLM fallback pass)")
        return

    print(f"\n{'='*80}")
    print("LLM FALLBACK PASS")
    print(f"{'='*80}")

    # Send all non-#1 rows that have candidates to LLM
    llm_indices = [i for i, ok in det_correct.items() if not ok and i in ranked_map]
    llm_inputs = [(row_map[i], ranked_map[i]) for i in llm_indices]

    if not llm_inputs:
        print("Nothing to send to LLM — all rows correct at #1.")
        return

    print(f"Sending {len(llm_inputs)} rows to LLM...\n")
    llm_results = await llm_match_batch(llm_inputs)

    llm_promoted = 0
    for i, llm_result in zip(llm_indices, llm_results):
        row = row_map[i]
        ranked = ranked_map[i]
        expected_legal, expected_idx = expected_map[i]
        llm_d = decide_from_llm(row, ranked, llm_result)
        if llm_d.status == HIGH_CONFIDENCE:
            got = llm_d.legal_name.upper()
            got_idx = llm_d.registration_index
            correct = _is_match(got, got_idx, expected_legal, expected_idx)
            tag = "[LLM-OK]" if correct else "[LLM-FP]"
            print(f"{tag} {row.name:45s} -> {llm_d.legal_name:45s}  ({llm_result.reason[:60]})")
            if correct:
                llm_promoted += 1
        else:
            print(f"[MISS]   {row.name:45s}  ({llm_result.reason[:60]})")

    total_correct_with_llm = correct_at_1 + llm_promoted
    print(f"\n{'='*80}")
    print("FINAL RESULTS (deterministic + LLM):")
    print(f"  Recall:   {total_correct_with_llm}/{total} ({total_correct_with_llm/total*100:.1f}%)")
    print(f"  LLM rescued: {llm_promoted} rows")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
