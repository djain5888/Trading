"""Pure candle invariant checks.

These functions are the single definition of what makes an individual candle
valid. They are shared by the :class:`~app.market.historical.models.Candle`
model (fail-fast at construction) and the validation engine (defensive
re-checking of candles that may have bypassed validation). Each returns an
error message, or ``None`` when the invariant holds.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal


def check_timestamp_aware(timestamp: datetime) -> str | None:
    """Return an error if the timestamp is not timezone-aware."""
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return "Candle timestamp must be timezone-aware."
    return None


def check_positive_prices(
    open_: Decimal, high: Decimal, low: Decimal, close: Decimal
) -> str | None:
    """Return an error if any price is not strictly positive."""
    if min(open_, high, low, close) <= 0:
        return "All OHLC prices must be strictly positive."
    return None


def check_ohlc_consistency(
    open_: Decimal, high: Decimal, low: Decimal, close: Decimal
) -> str | None:
    """Return an error if the OHLC relationships are inconsistent.

    ``high`` must be the maximum and ``low`` the minimum of the four prices.
    """
    if high < max(open_, close, low):
        return "High must be >= open, close and low."
    if low > min(open_, close, high):
        return "Low must be <= open, close and high."
    return None


def check_volume(volume: int) -> str | None:
    """Return an error if volume is negative."""
    if volume < 0:
        return "Volume must not be negative."
    return None


def check_open_interest(open_interest: int | None) -> str | None:
    """Return an error if open interest is present and negative."""
    if open_interest is not None and open_interest < 0:
        return "Open interest must not be negative."
    return None
