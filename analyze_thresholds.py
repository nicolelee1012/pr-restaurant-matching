"""Analyze score distributions on labeled data to pick decision thresholds."""
from __future__ import annotations

import asyncio

from pr_registry import load_restaurants, process_batch
from scorer import rank_candidates


async def main():
    restaurants = load_restaurants()
    labeled = [r for r in restaurants if r.get("Legal Name", "").strip()]

    print(f"Processing {len(labeled)} labeled rows (all cached)...\n")
    results = await process_batch(labeled, fetch_details=True)

    rows = []
    for res in results:
        row = res["restaurant"]
        name = row["Name"]
        expected_legal = row["Legal Name"].strip().upper()
        pr_link = row.get("Puerto Rico Link", "")
        expected_idx = pr_link.split("?c=")[-1] if "?c=" in pr_link else ""

        if not res["candidates"]:
            rows.append({
                "name": name, "expected": expected_legal,
                "top_score": 0, "margin": 0, "addr_score": 0,
                "name_score": 0, "correct_at_1": False, "n_candidates": 0,
            })
            continue

        ranked = rank_candidates(row, res["candidates"])
        top = ranked[0]
        second = ranked[1].final_score if len(ranked) > 1 else 0
        margin = top.final_score - second

        is_correct = (
            top.registration_index == expected_idx
            or top.corp_name.upper() == expected_legal
        )

        rows.append({
            "name": name,
            "expected": expected_legal,
            "matched": top.corp_name,
            "top_score": top.final_score,
            "name_score": top.name_scores.combined,
            "addr_score": top.addr_scores.combined,
            "margin": margin,
            "correct_at_1": is_correct,
            "n_candidates": len(ranked),
        })

    # Print all rows sorted by top_score
    print(f"{'Restaurant':<45} {'Correct':>7} {'Score':>6} {'Margin':>7} {'Name':>6} {'Addr':>6}")
    print("-" * 90)
    for r in sorted(rows, key=lambda x: x["top_score"]):
        tag = "YES" if r["correct_at_1"] else "NO"
        print(f"{r['name'][:44]:<45} {tag:>7} {r['top_score']:>6.1f} {r['margin']:>7.1f} {r['name_score']:>6.1f} {r['addr_score']:>6.1f}")

    # Summary statistics
    correct = [r for r in rows if r["correct_at_1"]]
    wrong = [r for r in rows if not r["correct_at_1"]]

    print(f"\n{'='*90}")
    print(f"CORRECT matches ({len(correct)}):")
    scores = [r["top_score"] for r in correct]
    margins = [r["margin"] for r in correct]
    print(f"  Score:  min={min(scores):.1f}  median={sorted(scores)[len(scores)//2]:.1f}  max={max(scores):.1f}")
    print(f"  Margin: min={min(margins):.1f}  median={sorted(margins)[len(margins)//2]:.1f}")

    print(f"\nWRONG matches ({len(wrong)}):")
    if wrong:
        scores_w = [r["top_score"] for r in wrong]
        margins_w = [r["margin"] for r in wrong]
        print(f"  Score:  min={min(scores_w):.1f}  median={sorted(scores_w)[len(scores_w)//2]:.1f}  max={max(scores_w):.1f}")
        print(f"  Margin: min={min(margins_w):.1f}  median={sorted(margins_w)[len(margins_w)//2]:.1f}")
        for r in wrong:
            print(f"  - {r['name'][:40]}: score={r['top_score']:.1f} margin={r['margin']:.1f} (got: {r['matched'][:35]})")

    # Threshold sweep
    print(f"\n{'='*90}")
    print("THRESHOLD SWEEP (score >= T)")
    print(f"{'Threshold':>10} {'Matched':>8} {'Correct':>8} {'Wrong':>6} {'Precision':>10} {'Coverage':>9}")
    print("-" * 60)
    for t in [50, 55, 60, 65, 70, 75, 80, 85, 90]:
        matched = [r for r in rows if r["top_score"] >= t]
        c = sum(1 for r in matched if r["correct_at_1"])
        w = sum(1 for r in matched if not r["correct_at_1"])
        prec = c / len(matched) * 100 if matched else 0
        cov = len(matched) / len(rows) * 100
        print(f"{t:>10} {len(matched):>8} {c:>8} {w:>6} {prec:>9.1f}% {cov:>8.1f}%")

    # Threshold sweep with margin
    print(f"\n{'='*90}")
    print("THRESHOLD SWEEP (score >= T AND margin >= M)")
    print(f"{'T':>5} {'M':>5} {'Matched':>8} {'Correct':>8} {'Wrong':>6} {'Precision':>10} {'Coverage':>9}")
    print("-" * 60)
    for t in [55, 60, 65, 70]:
        for m in [2, 5, 8, 10]:
            matched = [r for r in rows if r["top_score"] >= t and r["margin"] >= m]
            c = sum(1 for r in matched if r["correct_at_1"])
            w = sum(1 for r in matched if not r["correct_at_1"])
            prec = c / len(matched) * 100 if matched else 0
            cov = len(matched) / len(rows) * 100
            print(f"{t:>5} {m:>5} {len(matched):>8} {c:>8} {w:>6} {prec:>9.1f}% {cov:>8.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
