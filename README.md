# Puerto Rico restaurant → legal entity matching

Production-oriented pipeline: registry search (via Zyte), scoring, threshold decisions, optional LLM fallback.

## Setup

```bash
cd /Users/nicolelee/Desktop/datalane
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required environment

| Variable | Purpose |
|----------|---------|
| `ZYTE_API_KEY` | **Required** for PR registry HTTP calls through Zyte Extract API |
| `OPENAI_API_KEY` | Required only when using `llm_matcher` / full pipeline with LLM |

Never commit API keys. The code does not embed fallback secrets.

## Running

Always run from this directory (or set `PYTHONPATH` here) so imports resolve:

```bash
export ZYTE_API_KEY='…'
python run_pipeline.py --help
python eval_holdout.py
```

## Logging

Registry client uses the `logging` module (logger `pr_registry`). From a script:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Use `DEBUG` to see per-query search variants.

## Code layout

| Module | Role |
|--------|------|
| `models.py` | Dataclasses: `ScoredCandidate`, score breakdowns |
| `exceptions.py` | `ConfigurationError` and other explicit errors |
| `pr_registry.py` | Zyte + PR API, caching, batch processing |
| `scorer.py` | Name / address / aux scores → `ScoredCandidate` |
| `decision.py` | Thresholds → `MatchResult` |
| `llm_matcher.py` | OpenAI fallback (optional) |

## Type safety

Score outputs are `ScoredCandidate` instances (not loose dicts), so attribute access is checked by editors and mypy if you run it:

```bash
pip install mypy
mypy models.py scorer.py decision.py --ignore-missing-imports
```
