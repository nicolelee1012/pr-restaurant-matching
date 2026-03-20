"""
Full pipeline run — deterministic scoring + LLM fallback.

Outputs (written incrementally, flushed after every batch):
  - pr_output_matches.csv   — high-confidence auto-matches
  - pr_output_review.csv    — needs human review
  - pr_output_no_match.csv  — no match found
  - pr_output_full.csv      — all rows with status + diagnostics

Usage:
  python run_full_pipeline.py
  python run_full_pipeline.py --input "My Data.csv" --output-dir results/
  python run_full_pipeline.py --batch-size 100
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import time
from pathlib import Path
from typing import Any

from decision import (
    HIGH_CONFIDENCE,
    NEEDS_REVIEW,
    NO_MATCH,
    MatchResult,
    decide,
    decide_from_llm,
)
from llm_matcher import llm_match_batch
from models import RestaurantRow
from pr_registry import load_restaurants, process_batch
from scorer import rank_candidates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent

DEFAULT_INPUT_CSV = _HERE / "Puerto Rico Data_ v1109.csv"
DEFAULT_OUTPUT_DIR = _HERE
DEFAULT_BATCH_SIZE = 200

EXTRA_COLS = [
    "Legal Name",
    "Puerto Rico Link",
    "match_status",
    "match_confidence",
    "match_margin",
    "match_name_score",
    "match_addr_score",
    "match_reason",
    "match_source",
    "top2_name",
    "top2_score",
    "top3_name",
    "top3_score",
]


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------
def build_output_row(original_row: RestaurantRow, d: MatchResult) -> dict[str, Any]:
    # Start from the full original CSV row (preserves every column we didn't touch)
    row = original_row.to_csv_row()
    row["Legal Name"] = d.legal_name or ""
    row["Puerto Rico Link"] = d.pr_link or ""
    row["match_status"] = d.status
    row["match_confidence"] = d.confidence
    row["match_margin"] = d.margin
    row["match_name_score"] = d.name_score
    row["match_addr_score"] = d.addr_score
    row["match_reason"] = d.reason
    row["match_source"] = d.match_source
    cands = d.top_candidates or []
    row["top2_name"] = cands[1]["corp_name"] if len(cands) > 1 else ""
    row["top2_score"] = cands[1]["final_score"] if len(cands) > 1 else ""
    row["top3_name"] = cands[2]["corp_name"] if len(cands) > 2 else ""
    row["top3_score"] = cands[2]["final_score"] if len(cands) > 2 else ""
    return row


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
async def main(
    input_csv: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    with open(input_csv, encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    if not all_rows:
        logger.error("Input CSV is empty: %s", input_csv)
        return

    original_cols = list(all_rows[0].keys())
    all_cols = original_cols + [c for c in EXTRA_COLS if c not in original_cols]

    labeled_rows = [r for r in all_rows if r.legal_name]
    unlabeled_rows = [r for r in all_rows if not r.legal_name]

    logger.info("Total rows:                   %s", len(all_rows))
    logger.info("Already labeled (kept as-is): %s", len(labeled_rows))
    logger.info("To process:                   %s", len(unlabeled_rows))
    logger.info("Batch size:                   %s", batch_size)
    logger.info("Estimated batches:            %s", len(unlabeled_rows) // batch_size + 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_full = output_dir / "pr_output_full.csv"
    out_match = output_dir / "pr_output_matches.csv"
    out_review = output_dir / "pr_output_review.csv"
    out_no_match = output_dir / "pr_output_no_match.csv"

    match_count = review_count = no_match_count = llm_promoted = 0

    # Open all output files up front — allows incremental flushing so results
    # are visible on disk after every batch (crash-safe).
    with (
        open(out_full, "w", newline="", encoding="utf-8") as f_full,
        open(out_match, "w", newline="", encoding="utf-8") as f_match,
        open(out_review, "w", newline="", encoding="utf-8") as f_review,
        open(out_no_match, "w", newline="", encoding="utf-8") as f_no_match,
    ):
        w_full = csv.DictWriter(f_full, fieldnames=all_cols, extrasaction="ignore")
        w_match = csv.DictWriter(f_match, fieldnames=all_cols, extrasaction="ignore")
        w_review = csv.DictWriter(f_review, fieldnames=all_cols, extrasaction="ignore")
        w_no_match = csv.DictWriter(f_no_match, fieldnames=all_cols, extrasaction="ignore")

        for w in [w_full, w_match, w_review, w_no_match]:
            w.writeheader()

        # Pass pre-labeled rows straight through to the full output.
        for r in labeled_rows:
            out: dict[str, Any] = r.to_csv_row()
            out.update({
                "match_status": "pre-labeled",
                "match_confidence": "",
                "match_margin": "",
                "match_name_score": "",
                "match_addr_score": "",
                "match_reason": "human labeled",
                "match_source": "human",
                "top2_name": "",
                "top2_score": "",
                "top3_name": "",
                "top3_score": "",
            })
            w_full.writerow(out)
        f_full.flush()

        t0 = time.time()
        total = len(unlabeled_rows)
        processed = 0

        for batch_start in range(0, total, batch_size):
            batch = unlabeled_rows[batch_start : batch_start + batch_size]
            results = await process_batch(batch, fetch_details=True)

            # ── Step 1: deterministic pass ──────────────────────────────────
            batch_decisions: list[tuple[dict[str, Any], list, MatchResult]] = []
            llm_needed: list[tuple[int, dict[str, Any], list]] = []

            for i, res in enumerate(results):
                row = res["restaurant"]
                ranked = rank_candidates(row, res["candidates"]) if res["candidates"] else []
                d = decide(row, ranked)
                batch_decisions.append((row, ranked, d))
                if d.status != HIGH_CONFIDENCE and ranked:
                    llm_needed.append((i, row, ranked))

            # ── Step 2: LLM fallback for ambiguous rows ─────────────────────
            if llm_needed:
                llm_inputs = [(row, ranked) for _, row, ranked in llm_needed]
                llm_results = await llm_match_batch(llm_inputs)

                for (orig_idx, row, ranked), llm_result in zip(llm_needed, llm_results):
                    llm_d = decide_from_llm(row, ranked, llm_result)
                    if llm_d.status == HIGH_CONFIDENCE:
                        batch_decisions[orig_idx] = (row, ranked, llm_d)

            # ── Step 3: write decisions ─────────────────────────────────────
            for row, _ranked, d in batch_decisions:
                out_row = build_output_row(row, d)
                w_full.writerow(out_row)

                if d.status == HIGH_CONFIDENCE:
                    w_match.writerow(out_row)
                    match_count += 1
                    if d.match_source in ("llm", "llm_cache"):
                        llm_promoted += 1
                elif d.status == NEEDS_REVIEW:
                    w_review.writerow(out_row)
                    review_count += 1
                else:
                    w_no_match.writerow(out_row)
                    no_match_count += 1

            for f in [f_full, f_match, f_review, f_no_match]:
                f.flush()

            processed += len(batch)
            elapsed = time.time() - t0
            rate = processed / elapsed
            remaining = (total - processed) / rate if rate > 0 else 0
            logger.info(
                "[%s/%s] %.1f%% — %s matched (%s via LLM), %s review, %s no_match — ETA %.1f min",
                processed, total, processed / total * 100,
                match_count, llm_promoted,
                review_count, no_match_count,
                remaining / 60,
            )

    total_time = time.time() - t0
    logger.info("=" * 60)
    logger.info("DONE in %.1f min", total_time / 60)
    logger.info("  Matched:    %s  (%.1f%%)", match_count, match_count / total * 100)
    if match_count:
        logger.info(
            "    of which LLM-promoted: %s  (%.1f%% of matches)",
            llm_promoted, llm_promoted / match_count * 100,
        )
    logger.info("  Review:     %s  (%.1f%%)", review_count, review_count / total * 100)
    logger.info("  No match:   %s  (%.1f%%)", no_match_count, no_match_count / total * 100)
    logger.info("  Outputs in: %s", output_dir)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PR restaurant → legal-entity matching pipeline (deterministic + LLM)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to input CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSVs (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Rows per processing batch (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main(args.input, args.output_dir, args.batch_size))
