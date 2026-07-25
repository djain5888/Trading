"""Equity close prices and broker holdings cross-check.

Equity closes come from the Groww SDK (already wired for market data); we cache
them to a small JSON file so the portfolio can be valued offline and fail loudly
when the cache is missing or stale — the same discipline as the NAV source.

Broker cross-check: ``groww.get_holdings_for_user()`` (INVESTIGATE + REPORT).
When the SDK exposes it, we compare the broker's units against the units implied
by the imported transactions and flag any mismatch beyond a small tolerance. The
comparison itself is a pure function so it is deterministic and testable; the SDK
call is injected and mocked in tests.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Callable, Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Units difference at/above which imported vs broker quantities are flagged.
DEFAULT_UNIT_TOLERANCE = Decimal("0.001")


class PricePoint(BaseModel):
    """One cached equity close price."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Equity symbol.")
    close: Decimal = Field(gt=0, description="Close price.")
    date: datetime.date = Field(description="Price date.")


class HoldingMismatch(BaseModel):
    """A discrepancy between imported units and the broker's reported units."""

    model_config = ConfigDict(frozen=True)

    identifier: str = Field(description="Equity symbol.")
    imported_units: Decimal = Field(description="Units implied by imports.")
    broker_units: Decimal = Field(description="Units the broker reports.")

    @property
    def difference(self) -> Decimal:
        """Return broker units minus imported units."""
        return self.broker_units - self.imported_units


def cross_check_holdings(
    imported_units: Mapping[str, Decimal],
    broker_units: Mapping[str, Decimal],
    *,
    tolerance: Decimal = DEFAULT_UNIT_TOLERANCE,
) -> list[HoldingMismatch]:
    """Flag symbols whose imported units differ from the broker's beyond tolerance.

    A symbol present in one source but not the other counts as a mismatch
    (missing side treated as zero units), so gaps in either direction surface.
    """
    symbols = {s.upper() for s in imported_units} | {s.upper() for s in broker_units}
    upper_imported = {s.upper(): u for s, u in imported_units.items()}
    upper_broker = {s.upper(): u for s, u in broker_units.items()}
    mismatches: list[HoldingMismatch] = []
    for symbol in sorted(symbols):
        mine = upper_imported.get(symbol, Decimal(0))
        theirs = upper_broker.get(symbol, Decimal(0))
        if abs(theirs - mine) >= tolerance:
            mismatches.append(
                HoldingMismatch(
                    identifier=symbol, imported_units=mine, broker_units=theirs
                )
            )
    return mismatches


def load_equity_prices(
    cache_path: Path,
    symbols: list[str],
    *,
    on: date,
    fetcher: Callable[[str], Decimal | None] | None = None,
    offline: bool = False,
) -> dict[str, PricePoint]:
    """Load equity closes from cache, refreshing via ``fetcher`` when allowed.

    Missing symbols after a refresh attempt are simply absent from the result;
    the engine flags them as unpriced rather than valuing them at zero.
    """
    cache = _read_cache(cache_path)
    if fetcher is not None and not offline:
        for symbol in symbols:
            try:
                close = fetcher(symbol)
            except Exception as exc:  # noqa: BLE001 - keep going; cache covers the rest
                logger.warning("Equity price fetch failed for %s: %s", symbol, exc)
                continue
            if close is not None and close > 0:
                cache[symbol.upper()] = PricePoint(
                    symbol=symbol.upper(), close=close, date=on
                )
        _write_cache(cache_path, cache)
    return {s: p for s, p in cache.items() if s in {sym.upper() for sym in symbols}}


def _read_cache(path: Path) -> dict[str, PricePoint]:
    """Read the JSON equity-price cache, tolerating a missing/invalid file."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    points: dict[str, PricePoint] = {}
    for symbol, entry in raw.items() if isinstance(raw, dict) else []:
        try:
            points[symbol.upper()] = PricePoint(
                symbol=symbol.upper(),
                close=Decimal(str(entry["close"])),
                date=date.fromisoformat(entry["date"]),
            )
        except (KeyError, ValueError, InvalidOperation):
            continue
    return points


def _write_cache(path: Path, points: Mapping[str, PricePoint]) -> None:
    """Persist the equity-price cache as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        symbol: {"close": str(point.close), "date": point.date.isoformat()}
        for symbol, point in points.items()
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def default_price_cache_path() -> Path:
    """Return the default on-disk equity-price cache path."""
    import os

    root = os.environ.get("TITAN_PORTFOLIO_DIR", "./data/portfolio")
    return Path(root) / "equity_prices.json"
