"""
Candidate Scoring Module

Scores PR registry candidates against restaurant data using:
  - Name similarity (RapidFuzz) — primary signal
  - Address similarity (city, zip, street) — guardrail
  - Auxiliary signals (entity status, purpose) — tiebreaker
"""

from __future__ import annotations

import re
from rapidfuzz import fuzz

from models import AddrScores, AuxScores, Candidate, NameScores, RegistryEntityDetail, RestaurantRow, ScoredCandidate
from utils import clean_text, strip_corp_suffixes, strip_noise_words

# ---------------------------------------------------------------------------
# Score weights — calibrated on first 116 labeled rows (37 matches + 79 no-match)
# All sets of weights within a component sum to 1.0.
# ---------------------------------------------------------------------------

# Final composite weights
_W_NAME: float = 0.55   # name similarity dominates
_W_ADDR: float = 0.30   # address provides a geographic guardrail
_W_AUX: float = 0.15    # entity status / purpose as tiebreaker

# Address sub-weights
_W_CITY: float = 0.40
_W_ZIP: float = 0.35
_W_STREET: float = 0.25

# Auxiliary sub-weights
_W_STATUS: float = 0.60
_W_PURPOSE: float = 0.40

# Score adjustments
_BONUS_HIGH_NAME: float = 5.0          # added when name_score ≥ threshold
_THRESH_BONUS_NAME: float = 85.0       # name_score threshold to earn bonus
_PENALTY_INACTIVE: float = 0.60        # multiplier for non-ACTIVE entities
_PENALTY_NON_RESTAURANT: float = 0.70  # multiplier when purpose is clearly non-food
_THRESH_NON_RESTAURANT: float = 10.0   # purpose_score at or below this triggers penalty

# Fuzzy matching thresholds
_FUZZY_RATIO_THRESHOLD: float = 85.0   # fuzz.ratio score to count as a token match


# ---------------------------------------------------------------------------
# Name scoring
# ---------------------------------------------------------------------------

def _strip_location_suffix(name: str) -> str:
    return re.split(r"\s*[-–—]\s*", name)[0].strip()


def score_name(restaurant_name: str, corp_name: str) -> NameScores:
    """
    Score name similarity between restaurant DBA name and corporate legal name.
    """
    r_raw = clean_text(_strip_location_suffix(restaurant_name))
    r_clean = strip_noise_words(_strip_location_suffix(restaurant_name))
    c_raw = clean_text(corp_name)
    c_clean = strip_corp_suffixes(corp_name)

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
                        if fuzz.ratio(rt, ct) >= _FUZZY_RATIO_THRESHOLD:
                            matched_r += 0.9
                            break
        distinctive_coverage = matched_r / len(r_tokens)
    else:
        distinctive_coverage = 0.0

    sequence = fuzz.ratio(r_clean, c_clean)

    # Name sub-weights: token_sort + token_set + partial + wratio + overlap + sequence = 1.0
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


def score_address(restaurant_row: RestaurantRow, entity_detail: RegistryEntityDetail | None) -> AddrScores:
    if not entity_detail:
        return AddrScores(city_match=0, zip_match=0, street_score=0, combined=0)

    r_city = clean_text(restaurant_row.city)
    r_zip = _normalize_zip(restaurant_row.postal_code)
    r_street = clean_text(restaurant_row.street)  # intentionally blank when no street column

    # Address field names and fallback logic centralised in RegistryEntityDetail
    e_city = clean_text(entity_detail.city)
    e_zip = _normalize_zip(entity_detail.zip)
    e_street = clean_text(entity_detail.street)

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

    combined = _W_CITY * city_match + _W_ZIP * zip_match + _W_STREET * street_score

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


def score_auxiliary(entity_detail: RegistryEntityDetail | None) -> AuxScores:
    if not entity_detail:
        return AuxScores(status_score=0, purpose_score=0, is_active=False, combined=0)

    # Status and purpose field names centralised in RegistryEntityDetail
    status = entity_detail.status

    status_scores = {
        "ACTIVE": 100,
        "CANCELLED": 10,
        "DISSOLVED": 5,
        "MERGED": 20,
        "REVOKED": 5,
    }
    status_score = float(status_scores.get(status, 0))
    is_active = status == "ACTIVE"

    purpose = clean_text(entity_detail.purpose)
    purpose_score = 50.0

    if purpose:
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

    combined = _W_STATUS * status_score + _W_PURPOSE * purpose_score

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
    restaurant_row: RestaurantRow,
    candidate: Candidate,
) -> ScoredCandidate:
    """Score a single typed Candidate against a RestaurantRow."""
    name_scores = score_name(restaurant_row.name, candidate.corp_name)
    addr_scores = score_address(restaurant_row, candidate.detail)
    aux_scores = score_auxiliary(candidate.detail)

    name_s = name_scores.combined
    addr_s = addr_scores.combined
    aux_s = aux_scores.combined

    final_score = _W_NAME * name_s + _W_ADDR * addr_s + _W_AUX * aux_s

    if name_s >= _THRESH_BONUS_NAME:
        final_score += _BONUS_HIGH_NAME
    if not aux_scores.is_active:
        final_score *= _PENALTY_INACTIVE
    if aux_scores.purpose_score <= _THRESH_NON_RESTAURANT:
        final_score *= _PENALTY_NON_RESTAURANT

    return ScoredCandidate(
        corp_name=candidate.corp_name,
        registration_index=candidate.registration_index,
        status=candidate.entity_status,
        final_score=float(final_score),
        name_scores=name_scores,
        addr_scores=addr_scores,
        aux_scores=aux_scores,
        detail=candidate.detail,
    )


def rank_candidates(
    restaurant_row: RestaurantRow,
    candidates: list[Candidate],
) -> list[ScoredCandidate]:
    """Score and sort all candidates for a restaurant, best first."""
    scored = [score_candidate(restaurant_row, c) for c in candidates]
    scored.sort(key=lambda x: -x.final_score)
    return scored
