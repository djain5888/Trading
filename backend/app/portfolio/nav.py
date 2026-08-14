"""AMFI mutual-fund NAV source: parse NAVAll.txt, cache it, run offline.

AMFI publishes every scheme's daily NAV as a semicolon-delimited text file at
``https://portal.amfiindia.com/spages/NAVAll.txt``. The file interleaves
category header lines (no ``;``) with data rows:

    Scheme Code;ISIN Payout;ISIN Reinvest;Scheme Name;Net Asset Value;Date

We parse it into a snapshot keyed by AMFI scheme code, cache the raw text
locally, and can run entirely from that cache. A snapshot older than a
freshness bound fails loudly rather than silently valuing the book on stale
NAVs. The network fetch is an injected callable so the engine stays offline and
fully testable.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import get_logger
from app.portfolio.importers.base import parse_date

logger = get_logger(__name__)

#: The canonical AMFI all-schemes NAV endpoint.
AMFI_NAVALL_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
#: A cached snapshot older than this (calendar days) is considered stale.
DEFAULT_MAX_STALE_DAYS = 5
#: Plan/option words that distinguish AMFI variants but not the scheme itself.
#: Dropping them lets a portfolio's plain scheme name match any variant.
_PLAN_NOISE = frozenset(
    {
        "direct",
        "regular",
        "plan",
        "growth",
        "idcw",
        "dividend",
        "payout",
        "reinvestment",
        "reinvest",
        "option",
        "div",
    }
)
#: Minimum Jaccard token overlap for a fuzzy scheme-name match to be accepted.
DEFAULT_NAME_MATCH_THRESHOLD = 0.6


def scheme_name_tokens(name: str) -> frozenset[str]:
    """Normalise a scheme name to a set of distinctive tokens.

    Lower-cases, strips punctuation, and drops plan/option noise words
    ("Direct", "Regular", "Growth", "IDCW", ...) so a portfolio's plain
    "Parag Parikh Flexi Cap Fund" matches the AMFI "... - Direct Plan - Growth".
    """
    cleaned = re.sub(r"[^a-z0-9]+", " ", name.lower())
    return frozenset(
        token for token in cleaned.split() if token and token not in _PLAN_NOISE
    )


def _variant_rank(entry: NavEntry) -> tuple[int, str]:
    """Deterministic preference among equal-scoring variants (Direct Growth first)."""
    raw = entry.name.lower()
    if "direct" in raw and "growth" in raw:
        tier = 0
    elif "growth" in raw:
        tier = 1
    else:
        tier = 2
    return (tier, entry.scheme_code)


class NavEntry(BaseModel):
    """One scheme's NAV on a date."""

    model_config = ConfigDict(frozen=True)

    scheme_code: str = Field(description="AMFI scheme code.")
    name: str = Field(description="Scheme name.")
    nav: Decimal = Field(gt=0, description="Net asset value per unit.")
    date: datetime.date = Field(description="NAV date.")


class NavSnapshot(BaseModel):
    """A parsed set of NAV entries keyed by scheme code."""

    model_config = ConfigDict(frozen=True)

    as_of: date = Field(description="Latest NAV date in the snapshot.")
    entries: dict[str, NavEntry] = Field(description="Scheme code -> entry.")

    def get(self, scheme_code: str) -> NavEntry | None:
        """Return the NAV entry for a scheme code, or ``None``."""
        return self.entries.get(scheme_code.strip())

    def match_by_name(
        self, name: str, *, threshold: float = DEFAULT_NAME_MATCH_THRESHOLD
    ) -> NavEntry | None:
        """Fuzzy-match a scheme name to an AMFI entry by normalised token overlap.

        Returns the best entry whose Jaccard token similarity meets ``threshold``.
        Ties (typically the Direct/Regular × Growth/IDCW variants that collapse
        to the same tokens) are broken deterministically, preferring the Direct
        Growth variant so valuation is stable run to run. ``None`` if nothing
        clears the threshold.
        """
        query = scheme_name_tokens(name)
        if not query:
            return None
        best_score = 0.0
        best: list[NavEntry] = []
        for entry in self.entries.values():
            tokens = scheme_name_tokens(entry.name)
            union = len(query | tokens)
            if union == 0:
                continue
            score = len(query & tokens) / union
            if score > best_score + 1e-9:
                best_score, best = score, [entry]
            elif abs(score - best_score) <= 1e-9:
                best.append(entry)
        if best_score < threshold or not best:
            return None
        return min(best, key=_variant_rank)

    def is_stale(self, on: date, max_days: int = DEFAULT_MAX_STALE_DAYS) -> bool:
        """Return whether the snapshot is older than ``max_days`` before ``on``."""
        return (on - self.as_of).days > max_days


def parse_navall(text: str) -> NavSnapshot:
    """Parse AMFI NAVAll.txt into a :class:`NavSnapshot`.

    Raises:
        ValueError: If no valid NAV rows are found.
    """
    entries: dict[str, NavEntry] = {}
    latest: date | None = None
    for line in text.splitlines():
        if ";" not in line:
            continue  # category header or blank
        parts = line.split(";")
        if len(parts) < 6 or parts[0].strip().lower() == "scheme code":
            continue
        code = parts[0].strip()
        name = parts[3].strip()
        nav = _decimal(parts[4])
        when = parse_date(parts[5])
        if not code or nav is None or nav <= 0 or when is None:
            continue
        entries[code] = NavEntry(scheme_code=code, name=name, nav=nav, date=when)
        latest = when if latest is None or when > latest else latest

    if latest is None:
        raise ValueError("No valid NAV rows found in AMFI file.")
    return NavSnapshot(as_of=latest, entries=entries)


def _decimal(raw: str) -> Decimal | None:
    """Parse a NAV cell to Decimal, tolerating blanks and 'N.A.'."""
    text = raw.strip()
    if not text or text.upper() in {"N.A.", "NA", "-"}:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def load_navs(
    cache_path: Path,
    *,
    on: date,
    fetcher: Callable[[], str] | None = None,
    max_stale_days: int = DEFAULT_MAX_STALE_DAYS,
    offline: bool = False,
) -> NavSnapshot:
    """Load a NAV snapshot, refreshing the cache from the network when allowed.

    Order of operations:
      1. If a ``fetcher`` is given and we are not offline, download and cache.
      2. Otherwise read the cached file.
      3. Fail loudly if there is no cache, or the snapshot is stale.

    Args:
        cache_path: Where the raw NAVAll.txt is cached.
        on: The valuation date staleness is measured against.
        fetcher: Callable returning fresh NAVAll.txt text (network is injected).
        max_stale_days: Maximum tolerated age of the snapshot.
        offline: Force cache-only operation (no fetch even if a fetcher exists).

    Raises:
        FileNotFoundError: If offline/fetch-less and no cache exists.
        ValueError: If the snapshot cannot be parsed or is stale.
    """
    if fetcher is not None and not offline:
        try:
            text = fetcher()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
            logger.info("Refreshed AMFI NAV cache at %s.", cache_path)
        except Exception as exc:  # noqa: BLE001 - fall back to cache, then fail loudly
            logger.warning("AMFI NAV fetch failed (%s); falling back to cache.", exc)

    if not cache_path.exists():
        raise FileNotFoundError(
            f"No AMFI NAV cache at {cache_path} and no successful fetch — "
            "cannot value mutual funds."
        )
    snapshot = parse_navall(cache_path.read_text(encoding="utf-8"))
    if snapshot.is_stale(on, max_stale_days):
        raise ValueError(
            f"AMFI NAV snapshot is stale: as of {snapshot.as_of}, valuing on {on} "
            f"(> {max_stale_days} days). Refresh NAVAll.txt before valuing."
        )
    return snapshot


def default_cache_path() -> Path:
    """Return the default on-disk NAV cache path."""
    import os

    root = os.environ.get("TITAN_PORTFOLIO_DIR", "./data/portfolio")
    return Path(root) / "amfi_navall.txt"


def freshness_window(as_of: date, max_days: int = DEFAULT_MAX_STALE_DAYS) -> date:
    """Return the earliest date a snapshot as of ``as_of`` is still fresh for."""
    return as_of + timedelta(days=max_days)
