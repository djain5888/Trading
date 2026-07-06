"""Groww endpoint request factory.

Pure builders that turn domain arguments into transport-agnostic
:class:`RequestSpec` objects. Only route *paths* live here (composed from the
configured API-version segment); the host comes from configuration, so no URL
is hardcoded. Route names are the integration's assumed contract and are the
single place to adjust if Groww's paths differ.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from app.market.enums import Exchange, Interval, MarketStatus
from app.providers.http import HttpMethod, RequestSpec

# Mapping between Titan intervals and Groww's interval tokens. Centralised so
# both request building and any future validation share one definition.
INTERVAL_TOKENS: dict[Interval, str] = {
    Interval.ONE_MINUTE: "1m",
    Interval.FIVE_MINUTE: "5m",
    Interval.FIFTEEN_MINUTE: "15m",
    Interval.THIRTY_MINUTE: "30m",
    Interval.ONE_HOUR: "1h",
    Interval.ONE_DAY: "1d",
    Interval.ONE_WEEK: "1w",
    Interval.ONE_MONTH: "1M",
}

# Mapping from Groww status tokens to domain market statuses.
STATUS_TOKENS: dict[str, MarketStatus] = {
    "open": MarketStatus.OPEN,
    "closed": MarketStatus.CLOSED,
    "pre_open": MarketStatus.PRE_OPEN,
    "preopen": MarketStatus.PRE_OPEN,
    "post_close": MarketStatus.POST_CLOSE,
    "postclose": MarketStatus.POST_CLOSE,
    "holiday": MarketStatus.HOLIDAY,
}


def _path(api_version: str, *segments: str) -> str:
    """Compose a versioned, leading-slash path."""
    parts = [api_version, *segments]
    return "/" + "/".join(part.strip("/") for part in parts)


def auth(api_version: str) -> str:
    """Return the authentication path (used by the session manager)."""
    return _path(api_version, "auth", "token")


def market_status(api_version: str, exchange: Exchange) -> RequestSpec:
    """Build the market-status request."""
    return RequestSpec(
        method=HttpMethod.GET,
        path=_path(api_version, "market", "status"),
        params={"exchange": exchange.value},
    )


def quote(api_version: str, symbol: str, exchange: Exchange) -> RequestSpec:
    """Build the single-quote request."""
    return RequestSpec(
        method=HttpMethod.GET,
        path=_path(api_version, "quote"),
        params={"symbol": symbol, "exchange": exchange.value},
    )


def quotes(api_version: str, symbols: Sequence[str], exchange: Exchange) -> RequestSpec:
    """Build the multi-quote request."""
    return RequestSpec(
        method=HttpMethod.GET,
        path=_path(api_version, "quotes"),
        params={"symbols": ",".join(symbols), "exchange": exchange.value},
    )


def historical(
    api_version: str,
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    start: datetime,
    end: datetime,
) -> RequestSpec:
    """Build the historical-data request."""
    return RequestSpec(
        method=HttpMethod.GET,
        path=_path(api_version, "historical"),
        params={
            "symbol": symbol,
            "exchange": exchange.value,
            "interval": INTERVAL_TOKENS[interval],
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
    )


def search(api_version: str, query: str, exchange: Exchange | None) -> RequestSpec:
    """Build the symbol-search request."""
    params = {"query": query}
    if exchange is not None:
        params["exchange"] = exchange.value
    return RequestSpec(
        method=HttpMethod.GET, path=_path(api_version, "search"), params=params
    )


def market_depth(api_version: str, symbol: str, exchange: Exchange) -> RequestSpec:
    """Build the market-depth request."""
    return RequestSpec(
        method=HttpMethod.GET,
        path=_path(api_version, "depth"),
        params={"symbol": symbol, "exchange": exchange.value},
    )
