"""Groww response parsing.

Implements the :class:`app.providers.base.ResponseParser` interface for Groww.
The method signatures and return types are fixed now so the rest of the
provider can be wired against them; the field-level mapping is deferred to API
integration and therefore raises ``NotImplementedError``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.market.enums import Exchange, Interval
from app.market.models import (
    HistoricalData,
    MarketDepth,
    MarketStatusInfo,
    Quote,
    SymbolSearchResponse,
)


class GrowwResponseParser:
    """Parses raw Groww payloads into Titan domain models."""

    def parse_market_status(
        self, payload: Mapping[str, Any], exchange: Exchange
    ) -> MarketStatusInfo:
        """Parse a market-status payload."""
        raise NotImplementedError("Groww market-status parsing is not yet integrated.")

    def parse_quote(self, payload: Mapping[str, Any], exchange: Exchange) -> Quote:
        """Parse a single-quote payload."""
        raise NotImplementedError("Groww quote parsing is not yet integrated.")

    def parse_quotes(
        self, payload: Mapping[str, Any], exchange: Exchange
    ) -> list[Quote]:
        """Parse a multi-quote payload."""
        raise NotImplementedError("Groww quotes parsing is not yet integrated.")

    def parse_historical(
        self,
        payload: Mapping[str, Any],
        symbol: str,
        exchange: Exchange,
        interval: Interval,
    ) -> HistoricalData:
        """Parse a historical-candles payload."""
        raise NotImplementedError("Groww historical parsing is not yet integrated.")

    def parse_search(
        self, payload: Mapping[str, Any], query: str
    ) -> SymbolSearchResponse:
        """Parse a symbol-search payload."""
        raise NotImplementedError("Groww search parsing is not yet integrated.")

    def parse_market_depth(
        self, payload: Mapping[str, Any], exchange: Exchange
    ) -> MarketDepth:
        """Parse a market-depth payload."""
        raise NotImplementedError("Groww depth parsing is not yet integrated.")
