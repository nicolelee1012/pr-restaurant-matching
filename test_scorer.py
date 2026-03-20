"""Test scorer on labeled rows."""
from __future__ import annotations

import asyncio

from pr_registry import load_restaurants, process_batch
from scorer import rank_candidates


async def main():
    restaurants = load_restaurants()
    labeled = [r for r in restaurants if r.get("Legal Name", "").strip()]

    print(f"Processing {len(labeled)} labeled rows...\n")
    results = await process_batch(labeled, fetch_details=True)

    correct_at_1 = 0
    correct_in_top3 = 0
    total_with_candidates = 0
    total = len(results)
    failures: list = []

    for res in results:
        row = res.restaurant
        name = row.name
        expected_legal = row.legal_name.upper()
        expected_idx = row.pr_link.split("?c=")[-1] if "?c=" in row.pr_link else ""

        if not res.candidates:
            failures.append((name, expected_legal, "NO CANDIDATES", 0, []))
            continue

        total_with_candidates += 1
        ranked = rank_candidates(row, res.candidates)

        top = ranked[0]
        is_top1 = (
            top.registration_index == expected_idx
            or top.corp_name.upper() == expected_legal
        )

        is_top3 = False
        for r in ranked[:3]:
            if r.registration_index == expected_idx or r.corp_name.upper() == expected_legal:
                is_top3 = True
                break

        if is_top1:
            correct_at_1 += 1
        if is_top3:
            correct_in_top3 += 1

        if not is_top1:
            actual_rank = None
            for i, r in enumerate(ranked):
                if r.registration_index == expected_idx or r.corp_name.upper() == expected_legal:
                    actual_rank = i + 1
                    break

            failures.append(
                (
                    name,
                    expected_legal,
                    f"Ranked #{actual_rank}" if actual_rank else "NOT IN CANDIDATES",
                    top.final_score,
                    [
                        (
                            r.corp_name,
                            f"{r.final_score:.1f}",
                            r.name_scores.combined,
                            r.addr_scores.combined,
                        )
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
    print("RESULTS:")
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


if __name__ == "__main__":
    asyncio.run(main())
