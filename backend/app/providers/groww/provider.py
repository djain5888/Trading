"""Groww market-data provider.

Implements :class:`app.providers.base.MarketDataProvider` for Groww by wiring
together the authenticator, request builder and response parser. Every public
method follows the same shape — authenticate, build a request, send it, parse
the response — so integrating the real API means implementing a single
transport seam (:meth:`_send`) plus the parser.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from app.config.broker import GrowwSettings
from app.market.enums import Exchange, Interval
from app.market.models import (
    HistoricalData,
    HistoricalDataRequest,
    MarketDepth,
    MarketDepthRequest,
    MarketStatusInfo,
    MarketStatusRequest,
    Quote,
    QuoteRequest,
    QuotesRequest,
    SymbolSearchRequest,
    SymbolSearchResponse,
)
from app.providers.base import MarketDataProvider
from app.providers.groww.auth import GrowwAuthenticator
from app.providers.groww.parser import GrowwResponseParser
from app.providers.groww.request_builder import GrowwRequestBuilder
from app.providers.http import RequestSpec


class GrowwProvider(MarketDataProvider):
    """Groww implementation of the market-data provider interface."""

    def __init__(
        self,
        settings: GrowwSettings,
        *,
        authenticator: GrowwAuthenticator | None = None,
        request_builder: GrowwRequestBuilder | None = None,
        parser: GrowwResponseParser | None = None,
    ) -> None:
        """Initialise the provider.

        Collaborators are injectable to keep the provider testable; sensible
        defaults are constructed from ``settings`` when omitted.

        Args:
            settings: Groww credentials and endpoint configuration.
            authenticator: Session manager. Defaults to a new authenticator.
            request_builder: Request factory. Defaults to a new builder.
            parser: Response parser. Defaults to a new parser.
        """
        self._settings = settings
        self._auth = authenticator or GrowwAuthenticator(settings)
        self._builder = request_builder or GrowwRequestBuilder(settings)
        self._parser = parser or GrowwResponseParser()

    async def _send(self, spec: RequestSpec) -> dict[str, Any]:
        """Execute a request against Groww and return the raw JSON payload.

        This is the single transport seam. Implementing it (with the
        configured ``base_url`` and timeout, session refresh, and error
        mapping) is all that stands between this skeleton and live data.

        Args:
            spec: The request to execute.

        Raises:
            NotImplementedError: The HTTP transport is not yet integrated.
        """
        raise NotImplementedError(
            f"Groww HTTP transport is not yet integrated "
            f"(would call {spec.method} {spec.full_url(self._settings.base_url)})."
        )

    async def get_market_status(
        self, exchange: Exchange = Exchange.NSE
    ) -> MarketStatusInfo:
        """Return the current trading-session status of an exchange."""
        spec = self._builder.build_market_status(MarketStatusRequest(exchange=exchange))
        payload = await self._send(spec)
        return self._parser.parse_market_status(payload, exchange)

    async def get_quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Quote:
        """Return a live quote for a single instrument."""
        spec = self._builder.build_quote(QuoteRequest(symbol=symbol, exchange=exchange))
        payload = await self._send(spec)
        return self._parser.parse_quote(payload, exchange)

    async def get_quotes(
        self, symbols: Sequence[str], exchange: Exchange = Exchange.NSE
    ) -> list[Quote]:
        """Return live quotes for several instruments."""
        spec = self._builder.build_quotes(
            QuotesRequest(symbols=tuple(symbols), exchange=exchange)
        )
        payload = await self._send(spec)
        return self._parser.parse_quotes(payload, exchange)

    async def get_historical_data(
        self,
        symbol: str,
        interval: Interval,
        start_date: datetime,
        end_date: datetime,
        exchange: Exchange = Exchange.NSE,
    ) -> HistoricalData:
        """Return historical OHLCV candles for an instrument."""
        spec = self._builder.build_historical(
            HistoricalDataRequest(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                start=start_date,
                end=end_date,
            )
        )
        payload = await self._send(spec)
        return self._parser.parse_historical(payload, symbol, exchange, interval)

    async def search_symbol(
        self, query: str, exchange: Exchange | None = None
    ) -> SymbolSearchResponse:
        """Search the instrument master by free text."""
        spec = self._builder.build_search(
            SymbolSearchRequest(query=query, exchange=exchange)
        )
        payload = await self._send(spec)
        return self._parser.parse_search(payload, query)

    async def get_market_depth(
        self, symbol: str, exchange: Exchange = Exchange.NSE
    ) -> MarketDepth:
        """Return the order-book depth for an instrument."""
        spec = self._builder.build_market_depth(
            MarketDepthRequest(symbol=symbol, exchange=exchange)
        )
        payload = await self._send(spec)
        return self._parser.parse_market_depth(payload, exchange)
