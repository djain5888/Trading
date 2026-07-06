"""The real Groww market-data provider.

Implements :class:`MarketDataProvider` over the Groww HTTP API by composing a
session manager (auth/refresh), a pooled retrying HTTP client, and a response
mapper. Each call authenticates, builds a request, sends it, and maps the JSON
into domain models. Collaborators are injectable for testing.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from app.config.broker import GrowwSettings
from app.core.clock import Clock
from app.market.enums import Exchange, Interval
from app.market.models import (
    HistoricalData,
    MarketDepth,
    MarketStatusInfo,
    Quote,
    SymbolSearchResponse,
)
from app.providers.base import MarketDataProvider
from app.providers.groww import endpoints
from app.providers.groww.http_client import GrowwHTTPClient, Sleeper
from app.providers.groww.response_mapper import GrowwResponseMapper
from app.providers.groww.session import GrowwSessionManager
from app.providers.http import RequestSpec


class GrowwMarketDataProvider(MarketDataProvider):
    """Groww implementation of the market-data provider interface."""

    def __init__(
        self,
        settings: GrowwSettings,
        *,
        http_client: GrowwHTTPClient | None = None,
        session: GrowwSessionManager | None = None,
        mapper: GrowwResponseMapper | None = None,
        clock: Clock | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        """Initialise the provider.

        Args:
            settings: Groww credentials and endpoint/HTTP configuration.
            http_client: Shared HTTP client; a pooled one is built if omitted.
            session: Session manager; a default is built if omitted.
            mapper: Response mapper; a default is built if omitted.
            clock: Time source for token expiry.
            sleeper: Async sleep used for backoff (test injection).
        """
        self._settings = settings
        self._http = http_client or GrowwHTTPClient(settings, sleeper=sleeper)
        self._session = session or GrowwSessionManager(
            settings, self._http, clock=clock
        )
        self._mapper = mapper or GrowwResponseMapper()

    async def _send(self, spec: RequestSpec) -> dict[str, Any]:
        """Authenticate and send a request, returning the JSON body."""
        token = await self._session.get_access_token()
        return await self._http.request(
            spec, extra_headers={"Authorization": f"Bearer {token}"}
        )

    async def get_market_status(
        self, exchange: Exchange = Exchange.NSE
    ) -> MarketStatusInfo:
        """Return the current trading-session status of an exchange."""
        spec = endpoints.market_status(self._settings.api_version, exchange)
        return self._mapper.parse_market_status(await self._send(spec), exchange)

    async def get_quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Quote:
        """Return a live quote for a single instrument."""
        spec = endpoints.quote(self._settings.api_version, symbol, exchange)
        return self._mapper.parse_quote(await self._send(spec), exchange)

    async def get_quotes(
        self, symbols: Sequence[str], exchange: Exchange = Exchange.NSE
    ) -> list[Quote]:
        """Return live quotes for several instruments."""
        spec = endpoints.quotes(self._settings.api_version, symbols, exchange)
        return self._mapper.parse_quotes(await self._send(spec), exchange)

    async def get_historical_data(
        self,
        symbol: str,
        interval: Interval,
        start_date: datetime,
        end_date: datetime,
        exchange: Exchange = Exchange.NSE,
    ) -> HistoricalData:
        """Return historical OHLCV candles for an instrument."""
        spec = endpoints.historical(
            self._settings.api_version, symbol, exchange, interval, start_date, end_date
        )
        return self._mapper.parse_historical(
            await self._send(spec), symbol, exchange, interval
        )

    async def search_symbol(
        self, query: str, exchange: Exchange | None = None
    ) -> SymbolSearchResponse:
        """Search the instrument master by free text."""
        spec = endpoints.search(self._settings.api_version, query, exchange)
        return self._mapper.parse_search(await self._send(spec), query)

    async def get_market_depth(
        self, symbol: str, exchange: Exchange = Exchange.NSE
    ) -> MarketDepth:
        """Return the order-book depth for an instrument."""
        spec = endpoints.market_depth(self._settings.api_version, symbol, exchange)
        return self._mapper.parse_market_depth(await self._send(spec), exchange)

    async def aclose(self) -> None:
        """Release the underlying HTTP client."""
        await self._http.aclose()
