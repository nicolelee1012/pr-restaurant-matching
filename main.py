"""
Main entry point — delegates to run_full_pipeline.

Usage:
    python main.py                          # run with defaults
    python main.py --input data.csv         # custom input
    python main.py --output-dir results/    # custom output dir
    python main.py --batch-size 100         # tune concurrency
    python main.py --log-level DEBUG        # verbose logging

Environment variables required:
    ZYTE_API_KEY      — Zyte proxy credentials
    OPENAI_API_KEY    — OpenAI API key for LLM fallback
"""

import asyncio
import logging

from run_full_pipeline import _parse_args, main

if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main(args.input, args.output_dir, args.batch_size))
