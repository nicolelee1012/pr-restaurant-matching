"""
Shared text utilities used by both pr_registry and scorer.

Centralising here prevents the two modules from drifting out of sync.
"""

from __future__ import annotations

import re
import unicodedata


def strip_accents(text: str) -> str:
    """Remove diacritics/accents (é → e, ñ → n, etc.)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """Lowercase, strip accents, collapse punctuation to spaces."""
    text = strip_accents(text.lower().strip())
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
