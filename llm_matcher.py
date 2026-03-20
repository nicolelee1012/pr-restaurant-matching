"""
LLM Matcher — OpenAI-powered fallback for ambiguous cases.

Used for rows that the deterministic pipeline sends to NEEDS_REVIEW or NO_MATCH
where candidates exist. GPT-4o-mini evaluates the top candidates and decides
whether any is the correct legal entity for the restaurant.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from openai import AsyncOpenAI

from exceptions import ConfigurationError
from models import LLMMatchResponse, RestaurantRow, ScoredCandidate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 300

CACHE_DIR = Path(__file__).parent / "cache" / "llm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_LLM_SEMAPHORE: asyncio.Semaphore | None = None


def get_openai_api_key() -> str:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ConfigurationError(
            "OPENAI_API_KEY is not set. Export it before using the LLM matcher."
        )
    return key


def _get_semaphore() -> asyncio.Semaphore:
    global _LLM_SEMAPHORE
    if _LLM_SEMAPHORE is None:
        _LLM_SEMAPHORE = asyncio.Semaphore(20)
    return _LLM_SEMAPHORE


def _cache_key(restaurant_name: str, candidates: Sequence[ScoredCandidate]) -> str:  # noqa: E501
    payload = json.dumps(
        {
            "name": restaurant_name,
            "candidates": [c.corp_name for c in candidates],
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode(), usedforsecurity=False).hexdigest()


def _cache_get(key: str) -> dict[str, Any] | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupted LLM cache file — removing: %s", path)
            path.unlink(missing_ok=True)
            return None
    return None


def _cache_set(key: str, value: Mapping[str, Any]) -> None:
    path = CACHE_DIR / f"{key}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(value), f, ensure_ascii=False)
    except OSError as exc:
        logger.warning("Failed to write LLM cache %s: %s", path, exc)


def _build_prompt(restaurant: RestaurantRow, candidates: Sequence[ScoredCandidate]) -> str:
    name = restaurant.name
    street = restaurant.street or restaurant.full_address
    city = restaurant.city
    postal = restaurant.postal_code
    address_str = ", ".join(filter(None, [street, city, postal, "Puerto Rico"]))

    lines = [
        "You are helping match a restaurant to its Puerto Rico legal entity (incorporation record).",
        "",
        f'Restaurant name (doing business as): "{name}"',
        f"Restaurant address: {address_str}",
        "",
        "Below are candidate legal entities from the Puerto Rico corporate registry.",
        "Your job: identify which candidate (if any) is the SAME BUSINESS as the restaurant above.",
        "",
        "Rules:",
        "- The legal name may differ SIGNIFICANTLY from the DBA name — this is expected and common:",
        "  e.g. 'Casa Bacardí' → 'BACARDI CORPORATION', 'Kash N Karry' → 'SUPERMERCADOS HATILLO INC'",
        "  e.g. 'La Factoría' → 'FACTORIA LA SANSE INC', 'Martin's BBQ' → 'FRANQUICIAS DE MARTIN'S BBQ INC'",
        "- Match on the OPERATING ENTITY — the company that runs this specific restaurant/store location",
        "- Prioritize: (1) brand/concept name match, (2) city/neighborhood match, (3) address proximity",
        "- Franchise holding companies and regional subsidiaries are acceptable matches",
        "- Chain stores: the parent LLC/Corp is a valid match even if address differs from this location",
        "- If multiple candidates could match, prefer: exact brand name > same city > closest address",
        "- Only reject if no candidate plausibly operates this business",
        "",
        "Candidates:",
    ]

    for i, c in enumerate(candidates):
        # All API field names resolved through typed RegistryEntityDetail properties
        detail = c.detail
        addr_str = (detail.addr_str if detail else "") or "address unknown"
        purpose = f' | purpose: "{detail.purpose}"' if detail and detail.purpose else ""
        lines.append(f"  [{i}] {c.corp_name} | {c.status} | {addr_str}{purpose}")

    lines += [
        "",
        'Respond with ONLY valid JSON in this exact format:',
        '{"match_index": 0, "confidence": "high", "reason": "brief explanation"}',
        "or if no match:",
        '{"match_index": null, "confidence": null, "reason": "brief explanation"}',
        "",
        'confidence must be "high" — only return a match if you are confident (≥80%).',
        "Do not return medium or low-confidence matches; return null instead.",
    ]

    return "\n".join(lines)


async def llm_match(
    restaurant: RestaurantRow,
    candidates: Sequence[ScoredCandidate],
    client: AsyncOpenAI,
) -> LLMMatchResponse:
    if not candidates:
        return LLMMatchResponse.no_match("no candidates")

    cache_key = _cache_key(restaurant.name, candidates)
    cached = _cache_get(cache_key)
    if cached:
        return LLMMatchResponse.from_dict(cached, source="llm_cache")

    prompt = _build_prompt(restaurant, candidates)

    async with _get_semaphore():
        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise entity resolution assistant. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            if not response.choices:
                raise ValueError("OpenAI returned an empty choices list")
            raw = (response.choices[0].message.content or "").strip()

            # Robustly extract the JSON object regardless of markdown wrappers.
            json_start = raw.find("{")
            json_end = raw.rfind("}")
            if json_start != -1 and json_end > json_start:
                raw = raw[json_start : json_end + 1]
            result: dict[str, Any] = json.loads(raw)

            # Validate index before caching
            match_index = result.get("match_index")
            if match_index is not None and (
                not isinstance(match_index, int)
                or match_index < 0
                or match_index >= len(candidates)
            ):
                result["match_index"] = None
                result["confidence"] = None
                result["reason"] = f"invalid index from LLM: {result}"

            _cache_set(cache_key, result)
            return LLMMatchResponse.from_dict(result, source="llm")

        except Exception as e:
            logger.exception("LLM match failed")
            return LLMMatchResponse.no_match(f"LLM error: {e}", source="llm_error")


async def llm_match_batch(
    rows_with_candidates: list[tuple[RestaurantRow, list[ScoredCandidate]]],
) -> list[LLMMatchResponse]:
    """
    Run LLM matching concurrently for a list of (restaurant, candidates) pairs.

    Creates one shared AsyncOpenAI client for the batch and closes it when done.
    Returns one ``LLMMatchResponse`` per input row in the same order.
    """
    client = AsyncOpenAI(api_key=get_openai_api_key())
    try:
        tasks = [
            llm_match(restaurant, candidates[:20], client)
            for restaurant, candidates in rows_with_candidates
        ]
        return list(await asyncio.gather(*tasks))
    finally:
        await client.close()
