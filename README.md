# Puerto Rico Restaurant → Legal Entity Matching

Production pipeline that matches ~10,000 Puerto Rico restaurant locations (Google Maps / scrap.io) to their corresponding PR Corporate Registry legal entities — populating **Legal Name** and **Puerto Rico Link** columns.

**Results on 40-row holdout set (13 labeled, 27 confirmed no-match):**
| Metric | Value |
|--------|-------|
| Recall | 92.3% |
| Precision | 100% |
| False Positive Rate | 0.0% (target ≤ 5%) |

---

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/nicolelee1012/pr-restaurant-matching.git
cd pr-restaurant-matching

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API keys

Copy the example file and fill in both keys:

```bash
cp .env.example .env
```

Open `.env` and set:

```
ZYTE_API_KEY=your_zyte_key_here
OPENAI_API_KEY=your_openai_key_here
```

> **Where to get the keys:**
> - **Zyte API key** — [app.zyte.com](https://app.zyte.com) → Settings → API Keys. The pipeline proxies all PR Registry requests through Zyte to avoid rate limiting.
> - **OpenAI API key** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys). Used only for the LLM fallback (GPT-4o-mini) on ambiguous rows.

Then export them before running any script:

```bash
export ZYTE_API_KEY=$(grep ZYTE_API_KEY .env | cut -d= -f2)
export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d= -f2)
```

Or on Windows (PowerShell):

```powershell
$env:ZYTE_API_KEY  = "your_zyte_key_here"
$env:OPENAI_API_KEY = "your_openai_key_here"
```

> ⚠️ **Never commit `.env`** — it's already in `.gitignore`. The codebase contains zero embedded secrets.

---

## Running the pipeline

### Full run (all ~10K unlabeled rows)

```bash
python main.py
```

Options:

```bash
python main.py --help

python main.py --input "Puerto Rico Data_ v1109.csv"   # custom input
python main.py --output-dir results/                    # custom output folder
python main.py --batch-size 100                         # tune parallelism
python main.py --log-level DEBUG                        # verbose output
```

Outputs written incrementally to (flushed after every batch, so safe to interrupt):

| File | Contents |
|------|----------|
| `pr_output_matches.csv` | High-confidence auto-matches |
| `pr_output_review.csv` | Needs human review |
| `pr_output_no_match.csv` | No registry match found |
| `pr_output_full.csv` | All rows with status + scoring diagnostics |

### Evaluate on holdout set

```bash
python eval_holdout.py
```

Runs the full pipeline (deterministic + LLM) against `Puerto_Rico_hidden_set.csv` and prints precision / recall / FP rate.

### Analyze score thresholds (labeled rows only)

```bash
python analyze_thresholds.py
```

---

## Architecture

```
Input CSV
   │
   ▼
pr_registry.py   ←── Zyte proxy ──→  PR Corporate Registry API
  • 6–8 search variants per restaurant
  • Concurrent via asyncio.gather + Semaphore(20)
  • Local disk cache (MD5-keyed JSON) — skip already-fetched queries
   │
   ▼ up to 50 candidates
scorer.py
  • RapidFuzz: token_sort, token_set, partial, WRatio, overlap, distinctive_coverage
  • Name score (55%) + Address score (30%) + Auxiliary score (15%)
  • distinctive_coverage gate: fraction of restaurant's key tokens present in legal name
   │
   ▼ ranked ScoredCandidates
decision.py
  • 5-gate HIGH_CONFIDENCE policy (score, name, addr, margin, coverage + ACTIVE status)
  • Outputs: match / review / no_match
   │
   ├── HIGH_CONFIDENCE ──→ write to matches CSV
   │
   └── review / no_match (with candidates)
         │
         ▼
      llm_matcher.py
        • GPT-4o-mini, temp=0, top-20 candidates in prompt
        • Cached in cache/llm/ — never re-calls OpenAI for same input
        • decide_from_llm() upgrades row to HIGH_CONFIDENCE if LLM is confident
```

## Code layout

| Module | Role |
|--------|------|
| `utils.py` | Shared text normalization (`strip_accents`, `clean_text`) |
| `models.py` | Typed dataclasses: `ScoredCandidate`, score breakdowns |
| `exceptions.py` | `ConfigurationError` and other explicit errors |
| `pr_registry.py` | Zyte proxy + PR API search/info, caching, async batch processing |
| `scorer.py` | Name / address / aux scores → `ScoredCandidate` |
| `decision.py` | Multi-threshold classifier → `MatchResult` |
| `llm_matcher.py` | OpenAI GPT-4o-mini fallback for ambiguous rows |
| `run_full_pipeline.py` | Orchestrator: batching, LLM integration, incremental CSV output |
| `main.py` | CLI entry point (delegates to `run_full_pipeline`) |
| `eval_holdout.py` | Holdout set evaluation (deterministic + LLM) |
| `analyze_thresholds.py` | Score distribution analysis on labeled rows |

## Caching

All API responses are cached locally in `cache/`:

```
cache/
  search/   ← PR Registry search results (keyed by query string MD5)
  info/     ← PR Registry entity detail (keyed by registration index)
  llm/      ← OpenAI responses (keyed by restaurant name + candidates MD5)
```

The cache means re-runs are fast and free — API calls only happen for new queries.

## Type safety

All scoring outputs use typed dataclasses (`ScoredCandidate`, `MatchResult`) — no loose dicts in the hot path. Run mypy for static checking:

```bash
pip install mypy
mypy models.py scorer.py decision.py --ignore-missing-imports
```
