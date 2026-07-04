"""The market-data provider interface.

Titan depends only on :class:`MarketDataProvider`; concrete providers (Groww,
and any future broker) implement it. The :class:`ResponseParser` protocol
describes the seam that turns raw provider payloads into domain models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from app.market.enums import Exchange, Interval, MarketStatus
from app.market.models import (
    HistoricalData,
    MarketDepth,
    MarketStatusInfo,
    Quote,
    SymbolSearchResponse,
)


class MarketDataProvider(ABC):
    """Abstract market-data source.

    Implementations translate a broker's API into Titan's provider-agnostic
    domain models and raise the shared provider exceptions on failure.
    """

    @abstractmethod
    async def get_market_status(
        self, exchange: Exchange = Exchange.NSE
    ) -> MarketStatusInfo:
        """Return the current trading-session status of an exchange.

        Args:
            exchange: Exchange to query.

        Returns:
            The exchange's current status.
        """

    @abstractmethod
    async def get_quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Quote:
        """Return a live quote for a single instrument.

        Args:
            symbol: Trading symbol.
            exchange: Listing exchange.

        Returns:
            The instrument's current quote.
        """

    @abstractmethod
    async def get_quotes(
        self, symbols: Sequence[str], exchange: Exchange = Exchange.NSE
    ) -> list[Quote]:
        """Return live quotes for several instruments.

        Args:
            symbols: Trading symbols.
            exchange: Listing exchange shared by all symbols.

        Returns:
            One quote per requested symbol.
        """

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        interval: Interval,
        start_date: datetime,
        end_date: datetime,
        exchange: Exchange = Exchange.NSE,
    ) -> HistoricalData:
        """Return historical OHLCV candles for an instrument.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.
            start_date: Inclusive range start (UTC).
            end_date: Inclusive range end (UTC).
            exchange: Listing exchange.

        Returns:
            The requested candle series.
        """

    @abstractmethod
    async def search_symbol(
        self, query: str, exchange: Exchange | None = None
    ) -> SymbolSearchResponse:
        """Search the instrument master by free text.

        Args:
            query: Search query.
            exchange: Optional exchange filter.

        Returns:
            Matching instruments.
        """

    @abstractmethod
    async def get_market_depth(
        self, symbol: str, exchange: Exchange = Exchange.NSE
    ) -> MarketDepth:
        """Return the order-book depth for an instrument.

        Args:
            symbol: Trading symbol.
            exchange: Listing exchange.

        Returns:
            The current bid/ask depth snapshot.
        """

    async def is_market_open(self, exchange: Exchange = Exchange.NSE) -> bool:
        """Return whether the exchange is currently open for trading.

        The default implementation derives the answer from
        :meth:`get_market_status`; providers may override with a cheaper check.

        Args:
            exchange: Exchange to query.

        Returns:
            ``True`` if the exchange session is open.
        """
        status = await self.get_market_status(exchange)
        return status.status is MarketStatus.OPEN


@runtime_checkable
class ResponseParser(Protocol):
    """Interface for turning raw provider payloads into domain models.

    A parser isolates payload-shape knowledge from transport and orchestration,
    so response formats can evolve without touching the provider flow.
    """

    def parse_market_status(
        self, payload: Mapping[str, Any], exchange: Exchange
    ) -> MarketStatusInfo:
        """Parse a market-status payload."""
        ...

    def parse_quote(self, payload: Mapping[str, Any], exchange: Exchange) -> Quote:
        """Parse a single-quote payload."""
        ...

    def parse_quotes(
        self, payload: Mapping[str, Any], exchange: Exchange
    ) -> list[Quote]:
        """Parse a multi-quote payload."""
        ...

    def parse_historical(
        self,
        payload: Mapping[str, Any],
        symbol: str,
        exchange: Exchange,
        interval: Interval,
    ) -> HistoricalData:
        """Parse a historical-candles payload."""
        ...

    def parse_search(
        self, payload: Mapping[str, Any], query: str
    ) -> SymbolSearchResponse:
        """Parse a symbol-search payload."""
        ...

    def parse_market_depth(
        self, payload: Mapping[str, Any], exchange: Exchange
    ) -> MarketDepth:
        """Parse a market-depth payload."""
        ...
