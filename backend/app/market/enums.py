"""Enumerations for the market-data domain.

These enums are provider-agnostic: they describe concepts of the Indian
market itself, not any particular broker's API vocabulary. Providers are
responsible for mapping their own representations to and from these values.
"""

from __future__ import annotations

from enum import StrEnum


class Exchange(StrEnum):
    """Supported trading exchanges / segments."""

    NSE = "NSE"
    """National Stock Exchange (equity)."""
    BSE = "BSE"
    """Bombay Stock Exchange (equity)."""
    NFO = "NFO"
    """NSE futures & options segment."""
    MCX = "MCX"
    """Multi Commodity Exchange."""


class Interval(StrEnum):
    """Candle / bar intervals for historical data."""

    ONE_MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    THIRTY_MINUTE = "30m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"

    @property
    def is_intraday(self) -> bool:
        """Return ``True`` for sub-daily intervals."""
        return self in {
            Interval.ONE_MINUTE,
            Interval.FIVE_MINUTE,
            Interval.FIFTEEN_MINUTE,
            Interval.THIRTY_MINUTE,
            Interval.ONE_HOUR,
        }


class MarketStatus(StrEnum):
    """Trading-session status of an exchange."""

    OPEN = "open"
    CLOSED = "closed"
    PRE_OPEN = "pre_open"
    POST_CLOSE = "post_close"
    HOLIDAY = "holiday"
