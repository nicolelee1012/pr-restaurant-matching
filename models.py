"""
Shared data models for the PR restaurant ↔ legal-entity matching pipeline.

Using dataclasses here gives:
- Editor / mypy-friendly attribute access
- Clear shapes for scores and candidates (fewer KeyError surprises)
- Single place to document fields
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class RestaurantRow:
    """
    Typed representation of one row from the input CSV.

    **This is the single source of truth for CSV column names.**
    If the upstream spreadsheet renames a column (e.g. "Postal code" →
    "ZIP Code"), update ``COLUMN_MAP`` here — nothing else needs to change.

    ``_raw`` preserves every original column so ``to_csv_row()`` can round-
    trip the full row to the output CSV without dropping unknown columns.
    """

    # ── Column name → attribute name mapping ────────────────────────────────
    # Update this dict if the input CSV schema ever changes.
    COLUMN_MAP: ClassVar[dict[str, str]] = {
        "Name":             "name",
        "Legal Name":       "legal_name",
        "Puerto Rico Link": "pr_link",
        "City":             "city",
        "Postal code":      "postal_code",
        "Full address":     "full_address",
        "Street":           "street",
    }

    # ── Typed attributes ─────────────────────────────────────────────────────
    name: str
    legal_name: str = ""
    pr_link: str = ""
    city: str = ""
    postal_code: str = ""
    full_address: str = ""
    street: str = ""

    # Raw dict kept for output round-tripping; excluded from repr / equality.
    _raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # ── Constructors / serialisers ────────────────────────────────────────────
    @classmethod
    def from_csv_row(cls, row: dict[str, Any]) -> RestaurantRow:
        """
        Map a raw CSV dict to a typed ``RestaurantRow``.

        Missing columns silently produce empty strings — no ``KeyError`` at
        runtime even if the CSV lacks an optional column.
        """
        return cls(
            name=str(row.get("Name") or "").strip(),
            legal_name=str(row.get("Legal Name") or "").strip(),
            pr_link=str(row.get("Puerto Rico Link") or "").strip(),
            city=str(row.get("City") or "").strip(),
            postal_code=str(row.get("Postal code") or "").strip(),
            full_address=str(row.get("Full address") or "").strip(),
            street=str(row.get("Street") or "").strip(),
            _raw=dict(row),
        )

    def to_csv_row(self) -> dict[str, Any]:
        """
        Reconstruct a raw dict for CSV output.

        Starts from ``_raw`` (preserves every original column) then overlays
        the canonical typed fields so edits made in code are reflected.
        """
        out = dict(self._raw)
        out["Legal Name"] = self.legal_name
        out["Puerto Rico Link"] = self.pr_link
        return out


@dataclass
class RegistryEntityDetail:
    """
    Typed wrapper for the PR Registry ``/api/corporation/info`` payload.

    **Single source of truth for entity detail API field names.**
    All string keys (``"corpStreetAddress"``, ``"statusEn"``, ``"purpose"``, …)
    are accessed only through the typed properties below.  If the upstream API
    renames or restructures a field, update the property here — scorer, LLM
    matcher, and any other consumer automatically see the fix.
    """

    _raw: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RegistryEntityDetail:
        """Wrap a raw API dict; accepts None gracefully (returns empty wrapper)."""
        return cls(_raw=d or {})

    # ── Address ───────────────────────────────────────────────────────────────
    def _best_address(self) -> dict[str, Any]:
        """
        Return the most useful address dict from the payload.

        Preference: ``corpStreetAddress`` when its city is not "unknown",
        otherwise fall back to ``mainLocation.streetAddress``.
        If the upstream API restructures the address hierarchy, update here only.
        """
        corp_addr: dict[str, Any] = self._raw.get("corpStreetAddress") or {}
        if corp_addr and str(corp_addr.get("city") or "").lower() != "unknown":
            return corp_addr
        main_loc = self._raw.get("mainLocation") or {}
        if isinstance(main_loc, dict) and main_loc:
            sa = main_loc.get("streetAddress")
            if isinstance(sa, dict):
                return sa
        return corp_addr

    @property
    def city(self) -> str:
        return str(self._best_address().get("city") or "")

    @property
    def zip(self) -> str:
        return str(self._best_address().get("zip") or "")

    @property
    def street(self) -> str:
        return str(self._best_address().get("address1") or "")

    @property
    def addr_str(self) -> str:
        """Formatted address string for display / prompts."""
        parts = [self.street, self.city, self.zip]
        return ", ".join(p for p in parts if p)

    # ── Entity status / purpose ───────────────────────────────────────────────
    @property
    def status(self) -> str:
        """Entity status in upper-case (e.g. ``"ACTIVE"``, ``"DISSOLVED"``)."""
        corp = self._raw.get("corporation") or {}
        return str(corp.get("statusEn") or "").upper()

    @property
    def purpose(self) -> str:
        """Stated corporate purpose string, empty when absent / unknown."""
        corp = self._raw.get("corporation") or {}
        p = corp.get("purpose") or ""
        return "" if str(p).lower() in ("unknown", "n/a", "n a") else str(p)

    def __bool__(self) -> bool:
        """Allow ``if entity_detail:`` to behave as ``if raw_dict:``."""
        return bool(self._raw)


@dataclass
class Candidate:
    """
    One registry search result paired with its fetched detail payload.

    Registry API field names (``"corpName"``, ``"registrationIndex"``, …) are
    accessed only through the typed properties below.  If the upstream API ever
    renames a field, update the property — nothing else needs to change.
    """

    search_record: dict[str, Any]          # raw /search API record
    detail: RegistryEntityDetail | None = None  # typed /info payload, if fetched

    # ── Typed accessors for registry search-record fields ───────────────────
    @property
    def corp_name(self) -> str:
        return str(self.search_record.get("corpName") or "")

    @property
    def registration_index(self) -> str:
        return str(self.search_record.get("registrationIndex") or "")

    @property
    def entity_status(self) -> str:
        return str(self.search_record.get("statusEn") or "")

    @property
    def business_entity_id(self) -> str:
        return str(self.search_record.get("businessEntityId") or "")


@dataclass
class BatchResult:
    """
    Output of ``process_batch()`` for a single restaurant.

    Replaces the raw ``{"restaurant": …, "candidates": […]}`` dict so
    callers use typed attribute access instead of string key lookups.
    """

    restaurant: RestaurantRow
    candidates: list[Candidate]


@dataclass
class LLMMatchResponse:
    """
    Parsed response from the GPT-4o-mini fallback matcher.

    All LLM JSON field names (``"match_index"``, ``"confidence"``, …) are
    extracted once in ``from_dict()`` — the rest of the codebase sees only
    typed attributes.
    """

    match_index: int | None       # index into the ranked candidates list
    confidence: str | None        # "high" or None
    reason: str
    source: str = "llm"

    @classmethod
    def from_dict(cls, d: dict[str, Any], source: str = "llm") -> LLMMatchResponse:
        """Parse a raw LLM JSON response dict."""
        raw_idx = d.get("match_index")
        return cls(
            match_index=int(raw_idx) if raw_idx is not None else None,
            confidence=d.get("confidence"),
            reason=str(d.get("reason") or ""),
            source=source,
        )

    @classmethod
    def no_match(cls, reason: str = "", source: str = "llm") -> LLMMatchResponse:
        """Convenience constructor for a no-match result."""
        return cls(match_index=None, confidence=None, reason=reason, source=source)


@dataclass
class NameScores:
    token_sort: float
    token_set: float
    partial: float
    wratio: float
    token_overlap: float
    distinctive_coverage: float
    combined: float
    r_clean: str = ""
    c_clean: str = ""


@dataclass
class AddrScores:
    city_match: float
    zip_match: float
    street_score: float
    combined: float
    r_addr: str = ""
    e_addr: str = ""


@dataclass
class AuxScores:
    status_score: float
    purpose_score: float
    is_active: bool
    combined: float


@dataclass
class ScoredCandidate:
    """
    One registry candidate scored against a restaurant row.

    ``detail`` holds the raw PR ``/info`` payload when available — used for
    LLM prompts and debugging; keep optional to avoid huge logs by default.
    """

    corp_name: str
    registration_index: str
    status: str
    final_score: float
    name_scores: NameScores
    addr_scores: AddrScores
    aux_scores: AuxScores
    detail: RegistryEntityDetail | None = None

    def summary_dict(self) -> dict[str, Any]:
        """Compact dict for review CSV / logging."""
        return {
            "corp_name": self.corp_name,
            "registration_index": self.registration_index,
            "final_score": round(self.final_score, 1),
            "name_score": round(self.name_scores.combined, 1),
            "addr_score": round(self.addr_scores.combined, 1),
            "status": self.status,
        }


@dataclass
class CandidatePreview:
    """Top-k candidate snapshot attached to MatchResult for human review."""

    corp_name: str
    registration_index: str
    final_score: float
    name_score: float
    addr_score: float
    status: str

