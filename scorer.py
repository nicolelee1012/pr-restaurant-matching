"""
Candidate Scoring Module

Scores PR registry candidates against restaurant data using:
  - Name similarity (RapidFuzz) — primary signal
  - Address similarity (city, zip, street) — guardrail
  - Auxiliary signals (entity status, purpose) — tiebreaker
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from rapidfuzz import fuzz

from models import AddrScores, AuxScores, NameScores, ScoredCandidate
from utils import clean_text, strip_accents  # noqa: F401  (re-exported for callers)


# ---------------------------------------------------------------------------
# Name scoring
# ---------------------------------------------------------------------------
RESTAURANT_NOISE = {
    # Generic establishment-type words — single tokens only (multi-word strings
    # are never matched by the word-by-word iteration in _strip_restaurant_noise,
    # so only list individual words here).
    "restaurant", "restaurante",
    "bar", "grill", "steakhouse", "steak",
    "bakery", "panaderia", "cafeteria", "cafe",
    "sushi", "burger", "kitchen",
    "pizza", "pizzeria", "bbq",
    "seafood", "lounge", "tavern", "pub",
    "diner", "bistro", "trattoria",
    "cantina", "taqueria", "cocina",
    "shop", "place", "house",
    "sport", "sports", "food", "truck",
}

CORP_SUFFIXES = {
    "inc",
    "inc.",
    "corp",
    "corp.",
    "llc",
    "llp",
    "incorporated",
    "corporation",
    "company",
    "co",
    "co.",
    "incorporado",
    "incorporada",
    "sociedad",
    "enterprises",
    "enterprise",
    "group",
    "holdings",
}


def _strip_restaurant_noise(name: str) -> str:
    words = clean_text(name).split()
    filtered = [w for w in words if w not in RESTAURANT_NOISE]
    return " ".join(filtered) if filtered else clean_text(name)


def _strip_corp_suffixes(name: str) -> str:
    words = clean_text(name).split()
    filtered = [w for w in words if w not in CORP_SUFFIXES]
    return " ".join(filtered) if filtered else clean_text(name)


def _strip_location_suffix(name: str) -> str:
    return re.split(r"\s*[-–—]\s*", name)[0].strip()


def score_name(restaurant_name: str, corp_name: str) -> NameScores:
    """
    Score name similarity between restaurant DBA name and corporate legal name.
    """
    r_raw = clean_text(_strip_location_suffix(restaurant_name))
    r_clean = _strip_restaurant_noise(_strip_location_suffix(restaurant_name))
    c_raw = clean_text(corp_name)
    c_clean = _strip_corp_suffixes(corp_name)

    token_sort = max(
        fuzz.token_sort_ratio(r_raw, c_raw),
        fuzz.token_sort_ratio(r_clean, c_clean),
    )
    token_set = max(
        fuzz.token_set_ratio(r_raw, c_raw),
        fuzz.token_set_ratio(r_clean, c_clean),
    )
    partial = max(
        fuzz.partial_ratio(r_raw, c_raw),
        fuzz.partial_ratio(r_clean, c_clean),
    )
    wratio = max(
        fuzz.WRatio(r_raw, c_raw),
        fuzz.WRatio(r_clean, c_clean),
    )

    r_tokens = {t for t in r_clean.split() if len(t) >= 3}
    c_tokens = {t for t in c_clean.split() if len(t) >= 3}

    if r_tokens and c_tokens:
        overlap = len(r_tokens & c_tokens)
        unmatched_r = r_tokens - c_tokens
        unmatched_c = c_tokens - r_tokens
        prefix_matches = 0
        for rt in unmatched_r:
            for ct in unmatched_c:
                if len(rt) >= 4 and len(ct) >= 4:
                    if rt.startswith(ct) or ct.startswith(rt):
                        prefix_matches += 1
                        break
        token_overlap = (overlap + prefix_matches * 0.8) / max(len(r_tokens), 1)
    else:
        token_overlap = 0.0

    token_overlap_score = min(token_overlap * 100, 100)

    if r_tokens and c_tokens:
        matched_r = 0.0
        for rt in r_tokens:
            if rt in c_tokens:
                matched_r += 1
            else:
                for ct in c_tokens:
                    if len(rt) >= 4 and len(ct) >= 4:
                        if rt.startswith(ct) or ct.startswith(rt):
                            matched_r += 0.8
                            break
                        if fuzz.ratio(rt, ct) >= 85:
                            matched_r += 0.9
                            break
        distinctive_coverage = matched_r / len(r_tokens)
    else:
        distinctive_coverage = 0.0

    sequence = fuzz.ratio(r_clean, c_clean)

    combined = (
        0.15 * token_sort
        + 0.15 * token_set
        + 0.10 * partial
        + 0.15 * wratio
        + 0.25 * token_overlap_score
        + 0.20 * sequence
    )

    return NameScores(
        token_sort=float(token_sort),
        token_set=float(token_set),
        partial=float(partial),
        wratio=float(wratio),
        token_overlap=float(token_overlap_score),
        distinctive_coverage=float(distinctive_coverage),
        combined=float(combined),
        r_clean=r_clean,
        c_clean=c_clean,
    )


# ---------------------------------------------------------------------------
# Address scoring
# ---------------------------------------------------------------------------


def _normalize_zip(z: str) -> str:
    z_str = re.sub(r"[^0-9]", "", str(z).strip())
    if len(z_str) <= 3:
        z_str = z_str.zfill(5)
    return z_str[:5]


def score_address(restaurant_row: Mapping[str, Any], entity_detail: Mapping[str, Any] | None) -> AddrScores:
    if not entity_detail:
        return AddrScores(0, 0, 0, 0)

    r_city = clean_text(str(restaurant_row.get("City") or ""))
    r_zip = _normalize_zip(str(restaurant_row.get("Postal code") or ""))
    r_street = clean_text(str(restaurant_row.get("Street") or ""))

    corp_addr: Mapping[str, Any] | dict = entity_detail.get("corpStreetAddress") or {}
    if not corp_addr or str(corp_addr.get("city") or "").lower() == "unknown":
        main_loc = entity_detail.get("mainLocation") or {}
        if isinstance(main_loc, dict) and main_loc:
            sa = main_loc.get("streetAddress")
            if isinstance(sa, dict):
                corp_addr = sa

    e_city = clean_text(str(corp_addr.get("city") or ""))
    e_zip = _normalize_zip(str(corp_addr.get("zip") or ""))
    e_street = clean_text(str(corp_addr.get("address1") or ""))

    if r_city and e_city:
        city_sim = fuzz.ratio(r_city, e_city)
        city_match = 100.0 if city_sim >= 80 else float(city_sim)
    else:
        city_match = 0.0

    if r_zip and e_zip and r_zip != "00000" and e_zip != "00000":
        if r_zip == e_zip:
            zip_match = 100.0
        elif r_zip[:3] == e_zip[:3]:
            zip_match = 60.0
        else:
            zip_match = 0.0
    else:
        zip_match = 0.0

    if r_street and e_street:
        street_score = float(
            max(
                fuzz.token_sort_ratio(r_street, e_street),
                fuzz.partial_ratio(r_street, e_street),
            )
        )
    else:
        street_score = 0.0

    combined = 0.40 * city_match + 0.35 * zip_match + 0.25 * street_score

    return AddrScores(
        city_match=city_match,
        zip_match=zip_match,
        street_score=street_score,
        combined=combined,
        r_addr=f"{r_street}, {r_city} {r_zip}",
        e_addr=f"{e_street}, {e_city} {e_zip}",
    )


# ---------------------------------------------------------------------------
# Auxiliary scoring
# ---------------------------------------------------------------------------
RESTAURANT_PURPOSE_KEYWORDS = {
    "restaurant", "restaurante", "comida", "food", "pizza", "sushi",
    "bar", "cafe", "coffee", "bakery", "panaderia", "reposteria",
    "pollo", "chicken", "burger", "hamburger", "taco", "tacos",
    "ice cream", "helado", "donut", "creamery", "catering",
    "grill", "cocina", "kitchen", "steak", "bbq", "barbecue",
    "seafood", "mariscos", "soda fountain", "diner", "bistro",
    "cantina", "cerveceria", "brewery", "ceviche", "empanada",
    "frituras", "lechon", "lechonera", "sandwich", "sandwiches",
    "vegan", "vegetarian", "tropical", "bebidas", "drinks",
    "smoothie", "juice", "pasteleria", "confiteria", "chocolat",
    "aliment", "gastronom", "culin",
}

NON_RESTAURANT_PURPOSE_KEYWORDS = {
    "alquiler de propiedades", "property rental", "real estate",
    "bienes raices", "bienes inmuebles", "construccion", "construction",
    "insurance", "seguros", "consulting", "consultoria",
    "legal", "abogado", "attorney", "accounting", "contabilidad",
    "medical", "medico", "dental", "hospital", "health", "salud",
    "pharmacy", "farmacia", "veterinar", "pediatr",
    "education", "escuela", "school", "university",
    "technology", "software", "hardware", "telecom",
    "manufacturing", "manufactura", "industrial",
    "transportation", "transporte", "trucking",
    "cleaning", "limpieza", "janitorial",
    "auto parts", "automotive", "garage", "mechanic",
}


def score_auxiliary(entity_detail: Mapping[str, Any] | None) -> AuxScores:
    if not entity_detail:
        return AuxScores(0, 0, False, 0)

    corp = entity_detail.get("corporation") or {}
    status = str(corp.get("statusEn") or "").upper()

    status_scores = {
        "ACTIVE": 100,
        "CANCELLED": 10,
        "DISSOLVED": 5,
        "MERGED": 20,
        "REVOKED": 5,
    }
    status_score = float(status_scores.get(status, 0))
    is_active = status == "ACTIVE"

    purpose = clean_text(str(corp.get("purpose") or ""))
    purpose_score = 50.0

    if purpose and purpose not in ("unknown", "n a", ""):
        pl = purpose.lower()
        has_r = any(kw in pl for kw in RESTAURANT_PURPOSE_KEYWORDS)
        has_nr = any(kw in pl for kw in NON_RESTAURANT_PURPOSE_KEYWORDS)
        if has_r and not has_nr:
            purpose_score = 100.0
        elif has_nr and not has_r:
            purpose_score = 5.0
        elif has_r and has_nr:
            purpose_score = 50.0
        else:
            purpose_score = 50.0

    combined = 0.60 * status_score + 0.40 * purpose_score

    return AuxScores(
        status_score=status_score,
        purpose_score=purpose_score,
        is_active=is_active,
        combined=combined,
    )


# ---------------------------------------------------------------------------
# Overall candidate score
# ---------------------------------------------------------------------------


def score_candidate(
    restaurant_row: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> ScoredCandidate:
    search_rec = candidate["search_record"]
    raw_detail = candidate.get("detail")
    detail: dict[str, Any] | None
    if raw_detail is None:
        detail = None
    elif isinstance(raw_detail, dict):
        detail = raw_detail
    else:
        detail = None

    restaurant_name = str(restaurant_row.get("Name") or "")
    corp_name = str(search_rec.get("corpName") or "")

    name_scores = score_name(restaurant_name, corp_name)
    addr_scores = score_address(restaurant_row, detail)
    aux_scores = score_auxiliary(detail)

    name_s = name_scores.combined
    addr_s = addr_scores.combined
    aux_s = aux_scores.combined
    is_active = aux_scores.is_active
    purpose_s = aux_scores.purpose_score

    final_score = 0.55 * name_s + 0.30 * addr_s + 0.15 * aux_s

    if name_s >= 85:
        final_score += 5.0

    if not is_active:
        final_score *= 0.60

    if purpose_s <= 10:
        final_score *= 0.70

    return ScoredCandidate(
        corp_name=corp_name,
        registration_index=str(search_rec.get("registrationIndex") or ""),
        status=str(search_rec.get("statusEn") or ""),
        final_score=float(final_score),
        name_scores=name_scores,
        addr_scores=addr_scores,
        aux_scores=aux_scores,
        detail=detail,
    )


def rank_candidates(
    restaurant_row: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
) -> list[ScoredCandidate]:
    scored = [score_candidate(restaurant_row, c) for c in candidates]
    scored.sort(key=lambda x: -x.final_score)
    return scored
