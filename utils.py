"""
Shared text utilities used by both pr_registry and scorer.

Centralising here prevents the two modules from drifting out of sync.

**Single source of truth for noise-word and corp-suffix lists.**
Both pr_registry (search query generation) and scorer (name similarity)
strip the same categories of words — if you add a word here it is
automatically applied in both contexts.
"""

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Noise phrases — ordered longest-first so multi-word phrases are removed
# before their constituent words would be. str.replace() handles phrases
# correctly; word-by-word iteration cannot.
# ---------------------------------------------------------------------------
NOISE_PHRASES: list[str] = [
    # Multi-word first (must come before single words that overlap)
    "bar & grill", "bar and grill", "bar & restaurant",
    "sports bar", "sport bar",
    "steak house", "sushi bar", "food truck", "ice cream",
    # Single-word generic descriptors
    "restaurant", "restaurante",
    "steakhouse", "steak",
    "bakery", "panaderia", "cafeteria",
    "cafe", "café",
    "burger", "kitchen",
    "pizza", "pizzeria", "bbq",
    "seafood", "lounge", "tavern", "pub",
    "diner", "bistro", "trattoria",
    "cantina", "taqueria", "cocina",
    "shop", "place", "house",
    "bar", "grill",
    "sports", "sport", "food", "truck",
    "sushi",
]

# Words treated as "too common" to use as a standalone search query.
# Single-word searches using these would return thousands of unrelated results.
TOO_COMMON_WORDS: frozenset[str] = frozenset({
    "the", "los", "las", "del", "san", "new", "old", "big",
    "east", "west", "north", "south", "casa", "don", "el", "la",
})

# Corporate-entity suffixes stripped when comparing legal names.
CORP_SUFFIXES: frozenset[str] = frozenset({
    "inc", "inc.", "corp", "corp.", "llc", "llp",
    "incorporated", "corporation", "company", "co", "co.",
    "incorporado", "incorporada", "sociedad",
    "enterprises", "enterprise", "group", "holdings",
})


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """Remove diacritics/accents (é → e, ñ → n, etc.)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """Lowercase, strip accents, collapse punctuation to spaces."""
    text = strip_accents(text.lower().strip())
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_noise_phrases(name: str) -> str:
    """
    Remove restaurant noise phrases from *name* using substring replacement.

    Used by **pr_registry** for generating search query variants.

    Uses ``str.replace()`` so multi-word phrases ("bar & grill", "sports bar")
    are matched correctly — word-by-word iteration cannot match across token
    boundaries.  Phrases in ``NOISE_PHRASES`` are ordered longest-first so
    longer matches are consumed before shorter sub-phrases overlap them.
    """
    result = clean_text(name)
    for phrase in NOISE_PHRASES:
        result = result.replace(phrase, " ")
    return re.sub(r"\s+", " ", result).strip()


# Pre-built set of single-word noise tokens for fast word-boundary lookup.
# Multi-word phrases from NOISE_PHRASES are intentionally excluded here
# (word-by-word iteration cannot match them anyway).
_NOISE_WORDS: frozenset[str] = frozenset(
    p for p in NOISE_PHRASES if " " not in p
)


def strip_noise_words(name: str) -> str:
    """
    Remove single-word restaurant noise tokens from *name* word-by-word.

    Used by **scorer** for name-similarity comparisons.  Word-by-word
    removal guarantees word-boundary matching — ``str.replace("bar", "")``
    would corrupt words like "embarques".  Multi-word phrases ("bar & grill")
    are handled naturally because their constituent single words ("bar",
    "grill") are each stripped individually.

    The underlying ``_NOISE_WORDS`` set is derived from ``NOISE_PHRASES``
    (defined once in this module) so both functions stay in sync.
    """
    words = clean_text(name).split()
    filtered = [w for w in words if w not in _NOISE_WORDS]
    return " ".join(filtered) if filtered else clean_text(name)


def strip_corp_suffixes(name: str) -> str:
    """Remove corporate-entity suffixes (INC, LLC, CORP, …) word-by-word."""
    words = clean_text(name).split()
    filtered = [w for w in words if w not in CORP_SUFFIXES]
    return " ".join(filtered) if filtered else clean_text(name)
