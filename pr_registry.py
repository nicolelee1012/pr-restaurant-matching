"""
Puerto Rico Corporate Registry Search Module

Searches the PR corporate registry (rceapi.estado.pr.gov) via Zyte proxy
to find legal entity matches for restaurant names.

Two endpoints:
  1. POST /api/corporation/search  - search by name
  2. GET  /api/corporation/info/{registrationIndex} - get entity details
"""

from __future__ import annotations

import asyncio
import csv
import gzip
import json
import hashlib
import logging
import os
import re
import ssl
from base64 import b64decode
from pathlib import Path
from typing import Any

try:
    import brotli as _brotli
    _HAS_BROTLI = True
except ImportError:
    _brotli = None  # type: ignore[assignment]
    _HAS_BROTLI = False

import certifi
from aiohttp import BasicAuth, ClientSession, ClientTimeout, ClientResponse, TCPConnector

from exceptions import ConfigurationError
from models import BatchResult, Candidate, RegistryEntityDetail, RestaurantRow
from utils import CORP_SUFFIXES, TOO_COMMON_WORDS, strip_accents, strip_noise_phrases

logger = logging.getLogger(__name__)

# Build SSL context using certifi's CA bundle
SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ZYTE_ENDPOINT = "https://api.zyte.com/v1/extract"


def get_zyte_api_key() -> str:
    """Require Zyte API key from the environment (no embedded secrets)."""
    key = (os.environ.get("ZYTE_API_KEY") or "").strip()
    if not key:
        raise ConfigurationError(
            "ZYTE_API_KEY is not set. Export it before running the pipeline, e.g.\n"
            "  export ZYTE_API_KEY='…'\n"
        )
    return key

SEARCH_URL = "https://rceapi.estado.pr.gov/api/corporation/search"
INFO_URL_TEMPLATE = "https://rceapi.estado.pr.gov/api/corporation/info/{index}"

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
(CACHE_DIR / "search").mkdir(exist_ok=True)
(CACHE_DIR / "info").mkdir(exist_ok=True)

CSV_PATH = Path(__file__).parent / "Puerto Rico Data_ v1109.csv"

# Concurrency: maximized for speed
SEMAPHORE_LIMIT = 20  # parallel Zyte requests

# ---------------------------------------------------------------------------
# Name normalization utilities
# ---------------------------------------------------------------------------
def normalize_name(name: str) -> str:
    """
    Normalize a restaurant name for registry search queries.

    Noise phrases are stripped via ``strip_noise_phrases()`` from utils —
    the single canonical list shared with the scorer.  Multi-word phrases
    ("bar & grill", "sports bar") are handled correctly because we use
    str.replace() rather than word-by-word iteration.
    """
    # Strip location suffix first ("Ikebana Sushi Bar - Guaynabo" → "Ikebana Sushi Bar")
    name = re.sub(r"\s*[-–—]\s*.*$", "", name).strip()
    # Apply shared noise-phrase removal + accent stripping
    result = strip_noise_phrases(name)
    # Allow apostrophes through (n', 's) but remove everything else non-alphanumeric
    result = re.sub(r"[^a-z0-9' ]", " ", result)
    return re.sub(r"\s+", " ", result).strip()


def generate_search_variants(name: str) -> list[str]:
    """Generate multiple search query variants for a restaurant name."""
    variants = set()

    # Original name (cleaned of location suffixes after dash)
    clean = re.sub(r"\s*[-–—]\s*.*$", "", name).strip()
    if clean:
        variants.add(clean)

    # Normalized version (noise words stripped)
    normed = normalize_name(name)
    if normed:
        variants.add(normed)

    # First N words (brand name only)
    words = normed.split()
    if len(words) >= 2:
        variants.add(" ".join(words[:2]))
    if len(words) >= 3:
        variants.add(" ".join(words[:3]))

    # Single first word (core brand) - important for cases like "Condal"
    # Only use single-word search if the name is multi-word (otherwise we already have it)
    # and the word is distinctive enough (>= 4 chars)
    # TOO_COMMON_WORDS is defined in utils — update it there if needed
    if words and len(words[0]) >= 4 and words[0] not in TOO_COMMON_WORDS:
        variants.add(words[0])
    # If first word is too common, try the second word as brand
    if words and words[0] in TOO_COMMON_WORDS and len(words) >= 2 and len(words[1]) >= 4:
        variants.add(words[1])

    # Remove 's, 'n type contractions
    no_apos = re.sub(r"'[a-z]", "", normed)
    if no_apos != normed and no_apos.strip():
        variants.add(no_apos.strip())

    # Also try the original without common suffixes like Inc, LLC, Corp
    clean_orig = strip_accents(
        re.sub(r"\s*[-–—]\s*.*$", "", name).strip().lower()
    )
    clean_orig = re.sub(r"[^a-z0-9' ]", " ", clean_orig)
    clean_orig = re.sub(r"\s+", " ", clean_orig).strip()
    if clean_orig and clean_orig not in variants:
        variants.add(clean_orig)

    # Try plural/singular variants of first word (supermercado/supermercados)
    if words:
        w0 = words[0]
        rest = " ".join(words[1:]) if len(words) > 1 else ""
        if w0.endswith("s") and len(w0) >= 5:
            alt = w0[:-1] + ("" if not rest else " " + rest)
            variants.add(alt.strip())
        elif len(w0) >= 4:
            alt = w0 + "s" + ("" if not rest else " " + rest)
            variants.add(alt.strip())

    return [v for v in variants if len(v) >= 2]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_key(prefix: str, query: str) -> Path:
    h = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()
    return CACHE_DIR / prefix / f"{h}.json"


def cache_get(prefix: str, query: str) -> list[dict[str, Any]] | dict[str, Any] | None:
    path = _cache_key(prefix, query)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupted cache file — removing: %s", path)
            path.unlink(missing_ok=True)
            return None
    return None


def cache_set(prefix: str, query: str, data: list[dict[str, Any]] | dict[str, Any]) -> None:
    path = _cache_key(prefix, query)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Zyte-proxied HTTP helpers
# ---------------------------------------------------------------------------
async def _parse_zyte_response(resp: ClientResponse) -> dict[str, Any]:
    """Read and parse a Zyte API response, handling brotli/gzip."""
    raw = await resp.read()
    content_encoding = resp.headers.get("Content-Encoding", "")
    if "br" in content_encoding:
        if not _HAS_BROTLI:
            logger.warning(
                "Received brotli-encoded response but 'brotli' package is not installed. "
                "Install it with: pip install brotli"
            )
        else:
            raw = _brotli.decompress(raw)
    elif "gzip" in content_encoding:
        raw = gzip.decompress(raw)
    return json.loads(raw)


async def _zyte_request(
    session: ClientSession,
    url: str,
    sem: asyncio.Semaphore,
    api_key: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    """
    Make a GET or POST request through the Zyte proxy with exponential-backoff retries.

    ``api_key`` is passed in rather than looked up on every call so the env-var
    read happens once per session (in the callers ``search_registry`` /
    ``get_entity_info``) instead of N × M times.
    """
    payload: dict[str, Any] = {
        "url": url,
        "httpResponseBody": True,
        "httpRequestMethod": method,
    }
    if method == "POST" and body is not None:
        payload["httpRequestText"] = json.dumps(body)
        payload["customHttpRequestHeaders"] = [
            {"name": "Content-Type", "value": "application/json"},
            {"name": "Accept", "value": "application/json"},
        ]

    for attempt in range(retries):
        try:
            async with sem:
                async with session.post(
                    ZYTE_ENDPOINT,
                    auth=BasicAuth(api_key, ""),
                    json=payload,
                ) as resp:
                    data = await _parse_zyte_response(resp)
                    if "httpResponseBody" not in data:
                        logger.warning(
                            "Zyte %s missing httpResponseBody: %s",
                            method,
                            json.dumps(data)[:500],
                        )
                        return {}
                    return json.loads(b64decode(data["httpResponseBody"]))
        except Exception:
            if attempt < retries - 1:
                await asyncio.sleep(2**attempt)
            else:
                logger.exception(
                    "Zyte %s failed after %s attempts for %s", method, retries, url
                )
                return {}
    return {}  # unreachable, but satisfies type checkers


# ---------------------------------------------------------------------------
# PR Registry API wrappers
# ---------------------------------------------------------------------------
async def search_registry(
    session: ClientSession,
    query: str,
    sem: asyncio.Semaphore,
    api_key: str,
    only_active: bool = False,
) -> list[dict[str, Any]]:
    """Search the PR corporate registry for a name query. Returns list of records."""
    cached = cache_get("search", query)
    if cached is not None:
        return cached  # type: ignore[return-value]

    body = {
        "cancellationMode": False,
        "comparisonType": 1,
        "corpName": query,
        "isWorkFlowSearch": False,
        "limit": 250,
        "matchType": 4,  # contains
        "method": None,
        "onlyActive": only_active,
        "registryNumber": None,
        "advanceSearch": None,
    }

    result = await _zyte_request(session, SEARCH_URL, sem, api_key, method="POST", body=body)

    records: list[dict[str, Any]] = []
    if result.get("success") and result.get("response"):
        records = result["response"].get("records", [])

    cache_set("search", query, records)
    return records


async def get_entity_info(
    session: ClientSession,
    registration_index: str,
    sem: asyncio.Semaphore,
    api_key: str,
) -> dict[str, Any]:
    """Get detailed info for a specific entity by registration index."""
    cached = cache_get("info", registration_index)
    if cached is not None:
        return cached  # type: ignore[return-value]

    url = INFO_URL_TEMPLATE.format(index=registration_index)
    result = await _zyte_request(session, url, sem, api_key, method="GET")

    info: dict[str, Any] = {}
    if result.get("success") and result.get("response"):
        info = result["response"]

    cache_set("info", registration_index, info)
    return info


# ---------------------------------------------------------------------------
# Candidate generation for a single restaurant
# ---------------------------------------------------------------------------
def _name_token_overlap(restaurant_name: str, corp_name: str) -> float:
    """Quick token overlap score between restaurant name and corp name."""
    r_tokens = set(normalize_name(restaurant_name).split())
    c_tokens = set(strip_accents(corp_name.lower()).split())
    # Remove corporate suffixes and common stopwords (single source of truth: utils.py)
    stopwords = {"de", "del"} | TOO_COMMON_WORDS
    c_tokens -= CORP_SUFFIXES | stopwords
    r_tokens -= stopwords
    if not r_tokens or not c_tokens:
        return 0.0
    # Exact overlap
    overlap = r_tokens & c_tokens
    # Fuzzy: also count prefix matches (supermercado ~ supermercados)
    unmatched_r = r_tokens - overlap
    unmatched_c = c_tokens - overlap
    fuzzy_matches = 0
    for rt in unmatched_r:
        for ct in unmatched_c:
            # Prefix match: one starts with the other (min 4 chars)
            if len(rt) >= 4 and len(ct) >= 4:
                if rt.startswith(ct) or ct.startswith(rt):
                    fuzzy_matches += 1
                    break
    return (len(overlap) + fuzzy_matches * 0.8) / min(len(r_tokens), len(c_tokens))


async def find_candidates(
    session: ClientSession,
    restaurant_name: str,
    sem: asyncio.Semaphore,
    api_key: str,
) -> list[dict[str, Any]]:
    """
    Search PR registry with multiple name variants.
    Returns a deduplicated candidate list pre-filtered by name relevance.

    Pre-filter cap is deliberately generous (50) so that entities whose legal
    name differs significantly from the DBA name (e.g. "Kash N' Karry" →
    "SUPERMERCADOS HATILLO INC") are not discarded before the LLM sees them.
    """
    variants = generate_search_variants(restaurant_name)
    seen_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []

    for variant in variants:
        logger.debug("search variant query=%r", variant)
        records = await search_registry(session, variant, sem, api_key)
        logger.debug("search variant hits=%s", len(records))

        for rec in records:
            eid = rec.get("businessEntityId")
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                candidates.append(rec)

    # Pre-filter: if we have too many candidates, keep those with
    # meaningful name overlap (cap raised to 50 to avoid missing DBA→legal
    # divergences that only share a city/region word like "Hatillo").
    if len(candidates) > 50:
        scored = [
            (_name_token_overlap(restaurant_name, c.get("corpName", "")), c)
            for c in candidates
        ]
        scored.sort(key=lambda x: -x[0])
        filtered = [c for score, c in scored if score > 0]
        if len(filtered) < 50:
            seen = {c["businessEntityId"] for c in filtered}
            for _, c in scored:
                if c["businessEntityId"] not in seen:
                    filtered.append(c)
                    seen.add(c["businessEntityId"])
                if len(filtered) >= 50:
                    break
        candidates = filtered

    return candidates


# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------
def load_restaurants(csv_path: str | None = None) -> list[RestaurantRow]:
    """
    Load restaurant data from CSV and return typed ``RestaurantRow`` objects.

    Column name mapping lives in ``RestaurantRow.COLUMN_MAP`` — update there
    if the upstream spreadsheet schema ever changes.
    """
    path = csv_path or str(CSV_PATH)
    with open(path, encoding="utf-8-sig") as f:
        return [RestaurantRow.from_csv_row(row) for row in csv.DictReader(f)]


# ---------------------------------------------------------------------------
# Main: search for all restaurants and fetch entity details
# ---------------------------------------------------------------------------
async def process_batch(
    restaurants: list[RestaurantRow],
    start: int = 0,
    limit: int | None = None,
    fetch_details: bool = True,
) -> list[BatchResult]:
    """
    For each restaurant, search the PR registry and collect candidates.
    Optionally fetch detailed entity info for each candidate.

    Returns list of dicts::

        {
            "restaurant": <original CSV row>,
            "candidates": [
                {"search_record": <search API record>, "detail": <info payload | None>},
                ...
            ]
        }

    ``start`` / ``limit`` let callers process a slice without pre-slicing the
    input list; pass ``start=0, limit=None`` (the defaults) to process all rows.
    """
    subset = restaurants[start:]
    if limit is not None:
        subset = subset[:limit]

    # Resolve the API key once per batch, not once per request.
    api_key = get_zyte_api_key()
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)

    timeout = ClientTimeout(total=120)
    connector = TCPConnector(ssl=SSL_CTX, limit=50)
    async with ClientSession(
        timeout=timeout,
        connector=connector,
        auto_decompress=False,
    ) as session:

        async def process_one(i: int, row: RestaurantRow) -> BatchResult:
            name = row.name
            if not name:
                return BatchResult(restaurant=row, candidates=[])

            logger.info("[%s/%s] Searching: %s", start + i + 1, start + len(subset), name)

            candidates_raw = await find_candidates(session, name, sem, api_key)
            logger.info("  -> %s candidates found", len(candidates_raw))

            # Fetch details for the top 30 candidates concurrently.
            # 30 (up from 20) ensures that DBA→legal-name divergences ranked
            # between positions 20–30 still receive address/status data for
            # scoring and LLM prompts.
            top_recs = candidates_raw[:30]
            candidates: list[Candidate] = []

            if fetch_details:
                async def fetch_detail(rec: dict[str, Any]) -> Candidate:
                    reg_idx = rec.get("registrationIndex")
                    if reg_idx:
                        raw_detail = await get_entity_info(session, str(reg_idx), sem, api_key)
                        return Candidate(
                            search_record=rec,
                            detail=RegistryEntityDetail.from_dict(raw_detail),
                        )
                    return Candidate(search_record=rec, detail=None)

                candidates = list(await asyncio.gather(*[fetch_detail(rec) for rec in top_recs]))
            else:
                candidates = [Candidate(search_record=rec) for rec in top_recs]

            return BatchResult(restaurant=row, candidates=candidates)

        tasks = [process_one(i, row) for i, row in enumerate(subset)]
        results: list[BatchResult] = list(await asyncio.gather(*tasks))

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
async def main():
    """Run a small test: process first 5 restaurants without Legal Name."""
    restaurants = load_restaurants()

    # Filter to rows without Legal Name (the ones we need to find)
    unlabeled = [r for r in restaurants if not r.legal_name]
    # Also get labeled ones for testing
    labeled = [r for r in restaurants if r.legal_name]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info("Total restaurants: %s", len(restaurants))
    logger.info("  Already labeled: %s", len(labeled))
    logger.info("  Need matching:   %s", len(unlabeled))

    logger.info("=" * 60)
    logger.info("TEST: Searching for all %s labeled restaurants", len(labeled))
    logger.info("=" * 60)
    results = await process_batch(labeled, start=0, limit=None, fetch_details=True)

    found = 0
    not_found = 0
    found_list = []
    missed_list = []

    for res in results:
        row = res.restaurant
        name = row.name
        expected_legal = row.legal_name.upper()
        expected_index = row.pr_link.split("?c=")[-1] if "?c=" in row.pr_link else ""

        candidate_names = [c.corp_name.upper() for c in res.candidates]
        candidate_indices = [c.registration_index for c in res.candidates]

        # Check if expected entity is in candidates (by index or name)
        matched = (expected_index and expected_index in candidate_indices) or (
            expected_legal in candidate_names
        )

        if matched:
            found += 1
            found_list.append(name)
        else:
            not_found += 1
            missed_list.append((name, expected_legal, expected_index, len(res.candidates)))

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS: %s/%s correct entities found in candidates", found, found + not_found)
    logger.info("  Recall: %.1f%%", found / (found + not_found) * 100 if (found + not_found) else 0)
    logger.info("=" * 60)

    if missed_list:
        logger.info("MISSED (%s):", len(missed_list))
        for name, legal, idx, n_cands in missed_list:
            logger.info("  %s — expected %s (%s), n_cands=%s", name, legal, idx, n_cands)


if __name__ == "__main__":
    asyncio.run(main())
