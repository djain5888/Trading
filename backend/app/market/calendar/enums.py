"""Enumerations for the market-calendar domain."""

from __future__ import annotations

from enum import StrEnum


class MarketState(StrEnum):
    """The overall state of an exchange at a point in time."""

    PRE_MARKET = "pre_market"
    OPEN = "open"
    POST_MARKET = "post_market"
    CLOSED = "closed"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    MAINTENANCE = "maintenance"


class SessionType(StrEnum):
    """A named trading session within a trading day."""

    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    POST_MARKET = "post_market"

    def to_market_state(self) -> MarketState:
        """Map this session type to the market state it implies."""
        return _SESSION_STATE[self]


_SESSION_STATE: dict[SessionType, MarketState] = {
    SessionType.PRE_MARKET: MarketState.PRE_MARKET,
    SessionType.REGULAR: MarketState.OPEN,
    SessionType.POST_MARKET: MarketState.POST_MARKET,
}
