"""Groww request construction.

Pure functions that translate domain requests into transport-agnostic
:class:`RequestSpec` objects. No I/O happens here and no base URL is embedded;
paths are composed from the configured API-version segment so they remain
overridable. The exact Groww route names are placeholders until integration.
"""

from __future__ import annotations

from app.config.broker import GrowwSettings
from app.market.models import (
    HistoricalDataRequest,
    MarketDepthRequest,
    MarketStatusRequest,
    QuoteRequest,
    QuotesRequest,
    SymbolSearchRequest,
)
from app.providers.http import HttpMethod, RequestSpec


class GrowwRequestBuilder:
    """Builds :class:`RequestSpec` objects for Groww endpoints."""

    def __init__(self, settings: GrowwSettings) -> None:
        """Initialise the builder.

        Args:
            settings: Groww credentials and endpoint configuration.
        """
        self._settings = settings

    def _path(self, *segments: str) -> str:
        """Compose a versioned path from the configured API version.

        Args:
            *segments: Path segments following the version prefix.

        Returns:
            A leading-slash path such as ``/v1/quote``.
        """
        parts = [self._settings.api_version, *segments]
        return "/" + "/".join(part.strip("/") for part in parts)

    def _auth_headers(self) -> dict[str, str]:
        """Return authorization headers derived from the API key."""
        return {"Authorization": f"Bearer {self._settings.api_key}"}

    def build_market_status(self, request: MarketStatusRequest) -> RequestSpec:
        """Build the market-status request."""
        return RequestSpec(
            method=HttpMethod.GET,
            path=self._path("market", "status"),
            params={"exchange": request.exchange.value},
            headers=self._auth_headers(),
        )

    def build_quote(self, request: QuoteRequest) -> RequestSpec:
        """Build the single-quote request."""
        return RequestSpec(
            method=HttpMethod.GET,
            path=self._path("quote"),
            params={"symbol": request.symbol, "exchange": request.exchange.value},
            headers=self._auth_headers(),
        )

    def build_quotes(self, request: QuotesRequest) -> RequestSpec:
        """Build the multi-quote request."""
        return RequestSpec(
            method=HttpMethod.GET,
            path=self._path("quotes"),
            params={
                "symbols": ",".join(request.symbols),
                "exchange": request.exchange.value,
            },
            headers=self._auth_headers(),
        )

    def build_historical(self, request: HistoricalDataRequest) -> RequestSpec:
        """Build the historical-data request."""
        return RequestSpec(
            method=HttpMethod.GET,
            path=self._path("historical"),
            params={
                "symbol": request.symbol,
                "exchange": request.exchange.value,
                "interval": request.interval.value,
                "start": request.start.isoformat(),
                "end": request.end.isoformat(),
            },
            headers=self._auth_headers(),
        )

    def build_search(self, request: SymbolSearchRequest) -> RequestSpec:
        """Build the symbol-search request."""
        params = {"query": request.query}
        if request.exchange is not None:
            params["exchange"] = request.exchange.value
        return RequestSpec(
            method=HttpMethod.GET,
            path=self._path("search"),
            params=params,
            headers=self._auth_headers(),
        )

    def build_market_depth(self, request: MarketDepthRequest) -> RequestSpec:
        """Build the market-depth request."""
        return RequestSpec(
            method=HttpMethod.GET,
            path=self._path("depth"),
            params={"symbol": request.symbol, "exchange": request.exchange.value},
            headers=self._auth_headers(),
        )
