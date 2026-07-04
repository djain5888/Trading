"""Market-data service.

A thin, provider-agnostic gateway between the application and any
:class:`app.providers.base.MarketDataProvider`. It normalises inputs, validates
provider responses, logs every provider request, translates/propagates provider
errors, and caches market status for a short window. It deliberately contains
no trading, indicator, scanner or AI logic.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

from app.core.logging import get_logger
from app.market.enums import Exchange, Interval, MarketStatus
from app.market.models import HistoricalData, MarketStatusInfo, Quote
from app.providers.base import MarketDataProvider
from app.providers.exceptions import InvalidSymbolError, ProviderError

logger = get_logger(__name__)

T = TypeVar("T")

#: Default lifetime of a cached market-status entry, in seconds.
DEFAULT_MARKET_STATUS_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class _StatusCacheEntry:
    """A cached market-status value with its monotonic expiry time."""

    value: MarketStatusInfo
    expires_at: float


class MarketDataService:
    """Application-facing gateway to a market-data provider.

    The service depends only on the :class:`MarketDataProvider` interface; it
    never references a concrete provider. All methods are asynchronous.
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        *,
        market_status_ttl_seconds: float = DEFAULT_MARKET_STATUS_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialise the service.

        Args:
            provider: The market-data provider to delegate to.
            market_status_ttl_seconds: How long to cache market status.
            clock: Monotonic time source (injectable for testing).
        """
        self._provider = provider
        self._status_ttl = market_status_ttl_seconds
        self._clock = clock
        self._status_cache: dict[Exchange, _StatusCacheEntry] = {}

    # -- Public API --------------------------------------------------------

    async def get_quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Quote:
        """Return a validated, normalised quote for a single instrument.

        Args:
            symbol: Trading symbol (case-insensitive, whitespace-trimmed).
            exchange: Listing exchange.

        Returns:
            The instrument's quote.

        Raises:
            InvalidSymbolError: If ``symbol`` is blank.
            ProviderError: If the provider fails or returns inconsistent data.
        """
        normalized = self._normalize_symbol(symbol)
        quote = await self._call(
            "get_quote", self._provider.get_quote(normalized, exchange)
        )
        self._validate_quote(quote, normalized, exchange)
        return quote

    async def get_quotes(
        self, symbols: Sequence[str], exchange: Exchange = Exchange.NSE
    ) -> list[Quote]:
        """Return validated quotes for several instruments, in request order.

        Duplicate and blank symbols are removed before the provider is called.

        Args:
            symbols: Trading symbols.
            exchange: Listing exchange shared by all symbols.

        Returns:
            One quote per unique requested symbol, ordered as requested.

        Raises:
            InvalidSymbolError: If no valid symbols remain after normalising.
            ProviderError: If the provider fails or omits a requested symbol.
        """
        normalized = self._normalize_symbols(symbols)
        quotes = await self._call(
            "get_quotes", self._provider.get_quotes(normalized, exchange)
        )
        return self._normalize_quotes(quotes, normalized, exchange)

    async def get_historical_data(
        self,
        symbol: str,
        interval: Interval,
        start_date: datetime,
        end_date: datetime,
        exchange: Exchange = Exchange.NSE,
    ) -> HistoricalData:
        """Return validated historical candles for an instrument.

        Args:
            symbol: Trading symbol.
            interval: Candle interval.
            start_date: Inclusive range start (UTC).
            end_date: Inclusive range end (UTC).
            exchange: Listing exchange.

        Returns:
            The requested candle series.

        Raises:
            InvalidSymbolError: If ``symbol`` is blank.
            ValueError: If ``end_date`` is not strictly after ``start_date``.
            ProviderError: If the provider fails or returns inconsistent data.
        """
        normalized = self._normalize_symbol(symbol)
        if end_date <= start_date:
            raise ValueError("'end_date' must be strictly after 'start_date'.")
        data = await self._call(
            "get_historical_data",
            self._provider.get_historical_data(
                normalized, interval, start_date, end_date, exchange
            ),
        )
        self._validate_historical(data, normalized, interval)
        return data

    async def get_market_status(
        self, exchange: Exchange = Exchange.NSE
    ) -> MarketStatusInfo:
        """Return the exchange's market status, cached for a short window.

        Args:
            exchange: Exchange to query.

        Returns:
            The (possibly cached) market status.

        Raises:
            ProviderError: If the provider fails.
        """
        now = self._clock()
        cached = self._status_cache.get(exchange)
        if cached is not None and cached.expires_at > now:
            logger.debug("Market status cache hit for %s", exchange)
            return cached.value

        status = await self._call(
            "get_market_status", self._provider.get_market_status(exchange)
        )
        self._status_cache[exchange] = _StatusCacheEntry(
            value=status, expires_at=now + self._status_ttl
        )
        return status

    async def is_market_open(self, exchange: Exchange = Exchange.NSE) -> bool:
        """Return whether the exchange is currently open.

        Derived from :meth:`get_market_status`, so it benefits from the cache.

        Args:
            exchange: Exchange to query.

        Returns:
            ``True`` if the exchange session is open.
        """
        status = await self.get_market_status(exchange)
        return status.status is MarketStatus.OPEN

    # -- Internals ---------------------------------------------------------

    async def _call(self, operation: str, awaitable: Awaitable[T]) -> T:
        """Log, await and error-handle a single provider operation.

        Args:
            operation: Name of the provider operation (for logging).
            awaitable: The provider coroutine to await.

        Returns:
            The provider result.

        Raises:
            ProviderError: Re-raised after logging when the provider fails.
        """
        logger.info("Market-data provider request: %s", operation)
        try:
            result = await awaitable
        except ProviderError as exc:
            logger.error("Provider error during %s: %s", operation, exc)
            raise
        logger.debug("Market-data provider request succeeded: %s", operation)
        return result

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Trim and upper-case a symbol.

        Args:
            symbol: Raw symbol.

        Returns:
            The normalised symbol.

        Raises:
            InvalidSymbolError: If the symbol is blank.
        """
        normalized = symbol.strip().upper()
        if not normalized:
            raise InvalidSymbolError("Symbol must not be blank.", symbol=symbol)
        return normalized

    def _normalize_symbols(self, symbols: Sequence[str]) -> list[str]:
        """Normalise, drop blanks and de-duplicate while preserving order.

        Args:
            symbols: Raw symbols.

        Returns:
            Ordered, unique, normalised symbols.

        Raises:
            InvalidSymbolError: If no valid symbols remain.
        """
        seen: set[str] = set()
        result: list[str] = []
        for raw in symbols:
            candidate = raw.strip().upper()
            if candidate and candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
        if not result:
            raise InvalidSymbolError("At least one valid symbol is required.")
        return result

    @staticmethod
    def _validate_quote(quote: Quote, symbol: str, exchange: Exchange) -> None:
        """Validate that a quote matches the request.

        Args:
            quote: The provider-returned quote.
            symbol: The normalised requested symbol.
            exchange: The requested exchange.

        Raises:
            ProviderError: If the quote does not match the request.
        """
        if quote.symbol.strip().upper() != symbol:
            raise ProviderError(
                f"Provider returned quote for '{quote.symbol}', expected '{symbol}'."
            )
        if quote.exchange is not exchange:
            raise ProviderError(
                f"Provider returned quote on '{quote.exchange}', "
                f"expected '{exchange}'."
            )

    def _normalize_quotes(
        self, quotes: Sequence[Quote], symbols: Sequence[str], exchange: Exchange
    ) -> list[Quote]:
        """Validate and reorder quotes to match the requested symbols.

        Args:
            quotes: The provider-returned quotes.
            symbols: The normalised requested symbols, in order.
            exchange: The requested exchange.

        Returns:
            Quotes ordered to match ``symbols``.

        Raises:
            ProviderError: If a requested symbol is missing from the response.
        """
        by_symbol: dict[str, Quote] = {q.symbol.strip().upper(): q for q in quotes}
        ordered: list[Quote] = []
        for symbol in symbols:
            quote = by_symbol.get(symbol)
            if quote is None:
                raise ProviderError(
                    f"Provider response is missing requested symbol '{symbol}'."
                )
            self._validate_quote(quote, symbol, exchange)
            ordered.append(quote)
        return ordered

    @staticmethod
    def _validate_historical(
        data: HistoricalData, symbol: str, interval: Interval
    ) -> None:
        """Validate that a historical series matches the request.

        Args:
            data: The provider-returned series.
            symbol: The normalised requested symbol.
            interval: The requested interval.

        Raises:
            ProviderError: If the series does not match the request.
        """
        if data.symbol.strip().upper() != symbol:
            raise ProviderError(
                f"Provider returned history for '{data.symbol}', expected '{symbol}'."
            )
        if data.interval is not interval:
            raise ProviderError(
                f"Provider returned '{data.interval}' candles, expected '{interval}'."
            )
