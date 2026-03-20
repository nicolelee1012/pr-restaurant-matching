"""
Full Pipeline Runner — Phase 5

Processes all restaurants through:
  1. Candidate retrieval (PR Registry via Zyte)
  2. Candidate scoring (RapidFuzz name + address similarity)
  3. Decision logic (threshold-based classification)
  4. Output CSV generation

Usage:
  python run_pipeline.py                    # process all unlabeled rows
  python run_pipeline.py --start 0 --limit 100  # process a subset
  python run_pipeline.py --all              # process everything including labeled
"""

import argparse
import asyncio
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from decision import HIGH_CONFIDENCE, NEEDS_REVIEW, NO_MATCH, MatchResult, decide, print_summary
from models import BatchResult, RestaurantRow
from pr_registry import load_restaurants, process_batch
from scorer import rank_candidates

logger = logging.getLogger(__name__)


OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_scoring_and_decisions(results: list[BatchResult]) -> list[tuple[RestaurantRow, MatchResult]]:
    """Score candidates and apply decision logic for each restaurant."""
    decisions = []
    for res in results:
        ranked = rank_candidates(res.restaurant, res.candidates) if res.candidates else []
        d = decide(res.restaurant, ranked)
        decisions.append((res.restaurant, d))
    return decisions


def write_output_csv(decisions: list[tuple[dict, MatchResult]], output_path: Path):
    """
    Write the final output CSV with Legal Name + Puerto Rico Link populated
    for high-confidence matches.
    """
    if not decisions:
        return

    # Use original CSV columns + our new fields
    sample_row = decisions[0][0]
    original_cols = list(sample_row.keys())

    # Ensure our output columns exist
    extra_cols = ["Match Status", "Match Confidence", "Match Margin"]
    fieldnames = original_cols + [c for c in extra_cols if c not in original_cols]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row, d in decisions:
            out = row.to_csv_row()

            if d.status == HIGH_CONFIDENCE:
                out["Legal Name"] = d.legal_name
                out["Puerto Rico Link"] = d.pr_link
            else:
                # NEEDS_REVIEW and NO_MATCH: leave blank, don't auto-populate
                out["Legal Name"] = ""
                out["Puerto Rico Link"] = ""

            out["Match Status"] = d.status
            out["Match Confidence"] = d.confidence
            out["Match Margin"] = d.margin

            writer.writerow(out)

    logger.info("Wrote %s rows to %s", len(decisions), output_path)


def write_review_csv(decisions: list[tuple[dict, MatchResult]], output_path: Path):
    """Write a separate CSV for rows that need manual review."""
    review_rows = [(row, d) for row, d in decisions if d.status == NEEDS_REVIEW]
    if not review_rows:
        logger.info("No review rows to write.")
        return

    fieldnames = [
        "Name", "Full address", "City", "Postal code",
        "Match Status", "Reason",
        "Candidate 1 Name", "Candidate 1 Score", "Candidate 1 Index",
        "Candidate 2 Name", "Candidate 2 Score", "Candidate 2 Index",
        "Candidate 3 Name", "Candidate 3 Score", "Candidate 3 Index",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row, d in review_rows:
            out = {
                "Name": row.name,
                "Full address": row.full_address,
                "City": row.city,
                "Postal code": row.postal_code,
                "Match Status": d.status,
                "Reason": d.reason,
            }
            for i, cand in enumerate(d.top_candidates[:3], 1):
                out[f"Candidate {i} Name"] = cand.corp_name
                out[f"Candidate {i} Score"] = cand.final_score
                out[f"Candidate {i} Index"] = cand.registration_index

            writer.writerow(out)

    logger.info("Wrote %s review rows to %s", len(review_rows), output_path)


def write_metrics(decisions: list[tuple[dict, MatchResult]], output_path: Path):
    """Write a JSON metrics summary."""
    total = len(decisions)
    matched = sum(1 for _, d in decisions if d.status == HIGH_CONFIDENCE)
    review = sum(1 for _, d in decisions if d.status == NEEDS_REVIEW)
    no_match = sum(1 for _, d in decisions if d.status == NO_MATCH)

    # Score distributions for matched
    matched_scores = [d.confidence for _, d in decisions if d.status == HIGH_CONFIDENCE]
    review_scores = [d.confidence for _, d in decisions if d.status == NEEDS_REVIEW]

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_rows": total,
        "high_confidence": matched,
        "needs_review": review,
        "no_match": no_match,
        "pct_matched": round(matched / total * 100, 1) if total else 0,
        "pct_review": round(review / total * 100, 1) if total else 0,
        "pct_no_match": round(no_match / total * 100, 1) if total else 0,
        "avg_match_confidence": round(sum(matched_scores) / len(matched_scores), 1) if matched_scores else 0,
        "avg_review_confidence": round(sum(review_scores) / len(review_scores), 1) if review_scores else 0,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Wrote metrics to %s", output_path)
    return metrics


async def run_pipeline(restaurants: list[dict], batch_size: int = 50, start: int = 0):
    """
    Run the full pipeline in batches to manage memory and provide progress updates.

    Processing flow per batch:
      1. For each restaurant, search PR registry with multiple name variants (via Zyte)
      2. Fetch detailed entity info for top candidates
      3. Score all candidates using RapidFuzz (name + address + auxiliary)
      4. Apply decision thresholds to classify as match/review/no_match
    """
    total = len(restaurants)
    all_decisions = []

    # Process in batches
    for batch_start in range(start, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = restaurants[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"BATCH {batch_start//batch_size + 1}: rows {batch_start+1}-{batch_end} of {total}")
        print(f"{'='*60}")

        t0 = time.time()

        # Step 1+2: Candidate retrieval + detail fetching
        # batch is already sliced, so start=0; batch_start is only for display
        results = await process_batch(batch, start=0, fetch_details=True)

        # Step 3+4: Scoring + decision
        batch_decisions = run_scoring_and_decisions(results)
        all_decisions.extend(batch_decisions)

        elapsed = time.time() - t0
        matched = sum(1 for _, d in batch_decisions if d.status == HIGH_CONFIDENCE)
        review = sum(1 for _, d in batch_decisions if d.status == NEEDS_REVIEW)
        no_match_count = sum(1 for _, d in batch_decisions if d.status == NO_MATCH)

        logger.info(
            "Batch done in %.0fs — match: %s, review: %s, no_match: %s",
            elapsed, matched, review, no_match_count,
        )

        # Save intermediate results after each batch (crash recovery)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_path = OUTPUT_DIR / f"intermediate_{batch_end}.csv"
        write_output_csv(all_decisions, intermediate_path)
        logger.info("Intermediate save: %s", intermediate_path)

    return all_decisions


async def main():
    parser = argparse.ArgumentParser(description="PR TAM Matching Pipeline")
    parser.add_argument("--start", type=int, default=0, help="Start row index")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--all", action="store_true", help="Process all rows (including labeled)")
    args = parser.parse_args()

    restaurants = load_restaurants()

    if not args.all:
        # Only process unlabeled rows (skip the 50 already done by hand)
        restaurants = [r for r in restaurants if not r.legal_name]

    if args.limit:
        restaurants = restaurants[args.start:args.start + args.limit]
        start = 0
    else:
        start = args.start
        restaurants = restaurants[start:]
        start = 0  # reset since we sliced

    logger.info("Pipeline starting: %s restaurants to process", len(restaurants))
    logger.info("Batch size: %s", args.batch_size)

    t0 = time.time()

    # Run the full pipeline
    all_decisions = await run_pipeline(restaurants, batch_size=args.batch_size, start=0)

    total_time = time.time() - t0

    # Write final outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Main output CSV
    main_csv = OUTPUT_DIR / f"pr_tam_output_{timestamp}.csv"
    write_output_csv(all_decisions, main_csv)

    # Review CSV
    review_csv = OUTPUT_DIR / f"pr_tam_review_{timestamp}.csv"
    write_review_csv(all_decisions, review_csv)

    # Metrics
    metrics_path = OUTPUT_DIR / f"pr_tam_metrics_{timestamp}.json"
    metrics = write_metrics(all_decisions, metrics_path)

    # Print final summary
    print_summary([d for _, d in all_decisions])
    logger.info("Total time: %.1f minutes", total_time / 60)
    logger.info("Outputs:")
    logger.info("  Main CSV:   %s", main_csv)
    logger.info("  Review CSV: %s", review_csv)
    logger.info("  Metrics:    %s", metrics_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
