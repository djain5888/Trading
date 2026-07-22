"""The default watchlist, benchmark index and symbol -> sector metadata.

Titan's context engines (regime, sector strength, relative strength) need a
universe to operate on: a default watchlist so ``titan morning`` runs with no
``-s`` flags, a benchmark index for regime/RS, and a symbol -> sector map so the
sector engine can bucket members. Sensible NSE defaults live here so the app
works out of the box; a JSON file (``WATCHLIST_FILE``) can override them.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Groww's NSE symbol for the NIFTY 50 index (override via the watchlist file).
_DEFAULT_INDEX = "NIFTY"

#: Seeded symbol -> sector map for the default watchlist. Every default symbol
#: is mapped so the sector engine can bucket it; each sector has >= 2 members.
_DEFAULT_SECTORS: dict[str, str] = {
    "RELIANCE": "ENERGY",
    "ONGC": "ENERGY",
    "TCS": "IT",
    "INFY": "IT",
    "HDFCBANK": "BANKING",
    "ICICIBANK": "BANKING",
    "SBIN": "BANKING",
    "ITC": "FMCG",
    "HINDUNILVR": "FMCG",
    "MARUTI": "AUTO",
    "TATAMOTORS": "AUTO",
}

#: The default watchlist is all large caps.
_DEFAULT_TIERS: dict[str, str] = {symbol: "large" for symbol in _DEFAULT_SECTORS}

#: The recognised market-cap tiers, largest first.
CAP_TIERS: tuple[str, ...] = ("large", "mid", "small")

#: A bundled ~60-symbol NSE universe (large/mid/small) for wide validation.
WIDE_UNIVERSE_FILE = Path(__file__).with_name("universe_wide.json")


class WatchlistConfig(BaseModel):
    """The default universe: index benchmark, watchlist and sector map."""

    model_config = ConfigDict(frozen=True)

    index_symbol: str = Field(
        default=_DEFAULT_INDEX, min_length=1, description="Benchmark index symbol."
    )
    symbols: tuple[str, ...] = Field(
        default=tuple(_DEFAULT_SECTORS), description="Default watchlist symbols."
    )
    sectors: dict[str, str] = Field(
        default_factory=lambda: dict(_DEFAULT_SECTORS),
        description="Symbol -> sector metadata.",
    )
    tiers: dict[str, str] = Field(
        default_factory=lambda: dict(_DEFAULT_TIERS),
        description="Symbol -> market-cap tier (large/mid/small).",
    )

    @field_validator("index_symbol", mode="before")
    @classmethod
    def _normalise_index(cls, value: object) -> str:
        """Trim and upper-case the index symbol."""
        text = str(value).strip().upper()
        if not text:
            raise ValueError("index_symbol must not be blank.")
        return text

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalise_symbols(cls, value: object) -> tuple[str, ...]:
        """Trim, upper-case and de-duplicate symbols, preserving order."""
        if not isinstance(value, list | tuple):
            raise ValueError("symbols must be a list of strings.")
        seen: set[str] = set()
        result: list[str] = []
        for raw in value:
            symbol = str(raw).strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                result.append(symbol)
        return tuple(result)

    @field_validator("sectors", mode="before")
    @classmethod
    def _normalise_sectors(cls, value: object) -> dict[str, str]:
        """Upper-case symbol keys and trim sector names."""
        if not isinstance(value, dict):
            raise ValueError("sectors must be a mapping of symbol -> sector.")
        return {
            str(key).strip().upper(): str(sector).strip()
            for key, sector in value.items()
            if str(key).strip() and str(sector).strip()
        }

    @field_validator("tiers", mode="before")
    @classmethod
    def _normalise_tiers(cls, value: object) -> dict[str, str]:
        """Upper-case symbol keys and lower-case tier labels."""
        if not isinstance(value, dict):
            raise ValueError("tiers must be a mapping of symbol -> tier.")
        return {
            str(key).strip().upper(): str(tier).strip().lower()
            for key, tier in value.items()
            if str(key).strip() and str(tier).strip()
        }


def load_watchlist(path: Path | None = None) -> WatchlistConfig:
    """Load the watchlist, falling back to the seeded defaults.

    Args:
        path: Optional JSON file overriding the defaults. Missing or invalid
            files log a warning and yield the built-in defaults (never raise).

    Returns:
        The resolved :class:`WatchlistConfig`.
    """
    if path is None:
        return WatchlistConfig()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "Could not read watchlist file %s: %s; using defaults.", path, exc
        )
        return WatchlistConfig()
    if not isinstance(raw, dict):
        logger.warning("Watchlist file %s is not an object; using defaults.", path)
        return WatchlistConfig()
    try:
        return WatchlistConfig(**raw)
    except ValidationError as exc:
        logger.warning("Invalid watchlist file %s: %s; using defaults.", path, exc)
        return WatchlistConfig()


def load_wide_universe() -> WatchlistConfig:
    """Load the bundled wide NSE universe (large/mid/small caps) for validation.

    Returns:
        The :class:`WatchlistConfig` parsed from ``universe_wide.json``. If the
        bundled file is missing or invalid the built-in defaults are returned
        (mirroring :func:`load_watchlist`).
    """
    return load_watchlist(WIDE_UNIVERSE_FILE)


@lru_cache(maxsize=1)
def get_watchlist_config() -> WatchlistConfig:
    """Return the cached watchlist config, honouring ``WATCHLIST_FILE``."""
    from app.config.settings import get_settings

    configured = get_settings().watchlist_file
    return load_watchlist(Path(configured) if configured else None)
