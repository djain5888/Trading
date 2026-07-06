"""Unit tests for the real Groww market-data provider (mocked HTTP)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest
from app.config.broker import GrowwSettings
from app.core.clock import FakeClock
from app.market.enums import Exchange, Interval, MarketStatus
from app.providers.exceptions import (
    AuthenticationError,
    InvalidSymbolError,
    NetworkError,
    ProviderError,
    RateLimitError,
)
from app.providers.groww.http_client import GrowwHTTPClient
from app.providers.groww.market_data_provider import GrowwMarketDataProvider
from app.providers.groww.session import GrowwSessionManager

Handler = Callable[[httpx.Request], httpx.Response]
_NOW = datetime(2025, 1, 6, 10, 0, tzinfo=UTC)


class RecordingSleeper:
    """Async sleeper that records delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _settings(**overrides: object) -> GrowwSettings:
    """Return Groww settings with a token-less (direct-key) default."""
    base: dict[str, object] = {
        "api_key": "test-key",
        "base_url": "https://groww.test",
        "max_retries": 3,
    }
    base.update(overrides)
    return GrowwSettings(**base)  # type: ignore[arg-type]


def _provider(
    handler: Handler,
    *,
    settings: GrowwSettings | None = None,
    sleeper: RecordingSleeper | None = None,
) -> GrowwMarketDataProvider:
    """Build a provider whose HTTP client uses a mock transport."""
    settings = settings or _settings()
    client = httpx.AsyncClient(
        base_url=settings.base_url, transport=httpx.MockTransport(handler)
    )
    http = GrowwHTTPClient(
        settings, client=client, sleeper=sleeper or RecordingSleeper()
    )
    return GrowwMarketDataProvider(settings, http_client=http, clock=FakeClock(_NOW))


def _json(payload: object, status: int = 200) -> httpx.Response:
    """Build a JSON response."""
    return httpx.Response(status, json=payload)


# -- Authentication --------------------------------------------------------


async def test_authentication_exchanges_and_caches_token() -> None:
    """With a secret configured, the provider exchanges and caches a token."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/auth/token"):
            return _json({"access_token": "tok-123", "expires_in": 3600})
        assert request.headers["Authorization"] == "Bearer tok-123"
        return _json(
            {
                "symbol": "RELIANCE",
                "last_price": "2500.5",
                "timestamp": _NOW.isoformat(),
            }
        )

    provider = _provider(handler, settings=_settings(api_secret="shh"))

    await provider.get_quote("RELIANCE")
    await provider.get_quote("RELIANCE")

    assert calls.count("/v1/auth/token") == 1  # token cached across calls


async def test_direct_key_auth_without_secret() -> None:
    """Without a secret, the API key is used directly as the bearer token."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        return _json(
            {
                "symbol": "RELIANCE",
                "last_price": "2500.5",
                "timestamp": _NOW.isoformat(),
            }
        )

    provider = _provider(handler)
    quote = await provider.get_quote("RELIANCE")
    assert quote.last_price == Decimal("2500.5")


async def test_token_refresh_after_expiry() -> None:
    """An expired token triggers a second auth exchange."""
    clock = FakeClock(_NOW)
    auth_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal auth_calls
        if request.url.path.endswith("/auth/token"):
            auth_calls += 1
            return _json({"access_token": f"tok-{auth_calls}", "expires_in": 100})
        return _json({"symbol": "R", "last_price": "1", "timestamp": _NOW.isoformat()})

    settings = _settings(api_secret="shh", token_refresh_skew_seconds=0)
    client = httpx.AsyncClient(
        base_url=settings.base_url, transport=httpx.MockTransport(handler)
    )
    http = GrowwHTTPClient(settings, client=client, sleeper=RecordingSleeper())
    session = GrowwSessionManager(settings, http, clock=clock)
    provider = GrowwMarketDataProvider(settings, http_client=http, session=session)

    await provider.get_quote("R")
    clock.advance(timedelta(seconds=200))  # token TTL was 100s -> expired
    await provider.get_quote("R")

    assert auth_calls == 2


# -- Quotes / historical / search / depth ----------------------------------


async def test_get_quote() -> None:
    """A quote payload maps to a Quote."""
    provider = _provider(
        lambda r: _json(
            {
                "symbol": "RELIANCE",
                "last_price": "2500.5",
                "open": "2490",
                "volume": 1000,
                "timestamp": _NOW.isoformat(),
            }
        )
    )
    quote = await provider.get_quote("RELIANCE")
    assert quote.symbol == "RELIANCE"
    assert quote.last_price == Decimal("2500.5")
    assert quote.exchange is Exchange.NSE


async def test_get_quotes() -> None:
    """A multi-quote payload maps to a list of quotes."""
    provider = _provider(
        lambda r: _json(
            {
                "data": [
                    {"symbol": "A", "last_price": "1", "timestamp": _NOW.isoformat()},
                    {"symbol": "B", "last_price": "2", "timestamp": _NOW.isoformat()},
                ]
            }
        )
    )
    quotes = await provider.get_quotes(["A", "B"])
    assert [q.symbol for q in quotes] == ["A", "B"]


async def test_get_market_status() -> None:
    """A status payload maps to MarketStatusInfo."""
    provider = _provider(
        lambda r: _json({"status": "open", "timestamp": _NOW.isoformat()})
    )
    status = await provider.get_market_status()
    assert status.status is MarketStatus.OPEN


async def test_get_historical_data() -> None:
    """A historical payload maps to candles."""
    provider = _provider(
        lambda r: _json(
            {
                "candles": [
                    {
                        "timestamp": _NOW.isoformat(),
                        "open": "100",
                        "high": "105",
                        "low": "99",
                        "close": "101",
                        "volume": 500,
                    }
                ]
            }
        )
    )
    data = await provider.get_historical_data("RELIANCE", Interval.ONE_DAY, _NOW, _NOW)
    assert len(data.candles) == 1
    assert data.candles[0].close == Decimal("101")


async def test_search_symbol() -> None:
    """A search payload maps to results."""
    provider = _provider(
        lambda r: _json(
            {"results": [{"symbol": "REL", "name": "Reliance", "exchange": "NSE"}]}
        )
    )
    response = await provider.search_symbol("rel")
    assert response.results[0].symbol == "REL"


async def test_get_market_depth() -> None:
    """A depth payload maps to MarketDepth."""
    provider = _provider(
        lambda r: _json(
            {
                "symbol": "RELIANCE",
                "timestamp": _NOW.isoformat(),
                "bids": [{"price": "100", "quantity": 10, "orders": 2}],
                "asks": [{"price": "101", "quantity": 5}],
            }
        )
    )
    depth = await provider.get_market_depth("RELIANCE")
    assert depth.bids[0].price == Decimal("100")
    assert depth.asks[0].quantity == 5


# -- Error handling --------------------------------------------------------


async def test_invalid_symbol_maps_404() -> None:
    """A 404 becomes InvalidSymbolError."""
    provider = _provider(lambda r: _json({"error": "not found"}, status=404))
    with pytest.raises(InvalidSymbolError):
        await provider.get_quote("NOPE")


async def test_authentication_error_maps_401() -> None:
    """A 401 becomes AuthenticationError."""
    provider = _provider(lambda r: _json({"error": "unauthorized"}, status=401))
    with pytest.raises(AuthenticationError):
        await provider.get_quote("RELIANCE")


async def test_malformed_response() -> None:
    """A non-JSON body becomes a ProviderError."""
    provider = _provider(lambda r: httpx.Response(200, text="not json"))
    with pytest.raises(ProviderError):
        await provider.get_quote("RELIANCE")


async def test_malformed_schema() -> None:
    """A JSON body missing required fields becomes a ProviderError."""
    provider = _provider(lambda r: _json({"unexpected": True}))
    with pytest.raises(ProviderError):
        await provider.get_quote("RELIANCE")


async def test_timeout_maps_to_network_error_after_retries() -> None:
    """Timeouts are retried and finally raised as NetworkError."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectTimeout("slow", request=request)

    sleeper = RecordingSleeper()
    provider = _provider(handler, settings=_settings(max_retries=2), sleeper=sleeper)

    with pytest.raises(NetworkError):
        await provider.get_quote("RELIANCE")
    assert attempts == 3  # initial + 2 retries
    assert len(sleeper.delays) == 2


# -- Rate limit / retry ----------------------------------------------------


async def test_rate_limit_then_success_honours_retry_after() -> None:
    """A 429 with Retry-After is retried after the hinted delay."""
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "2"}, json={}),
            _json({"symbol": "R", "last_price": "1", "timestamp": _NOW.isoformat()}),
        ]
    )
    sleeper = RecordingSleeper()
    provider = _provider(lambda r: next(responses), sleeper=sleeper)

    quote = await provider.get_quote("R")
    assert quote.symbol == "R"
    assert sleeper.delays == [2.0]


async def test_rate_limit_exhausted_raises() -> None:
    """Persistent 429s exhaust retries and raise RateLimitError."""
    provider = _provider(
        lambda r: httpx.Response(429, json={}),
        settings=_settings(max_retries=1),
    )
    with pytest.raises(RateLimitError):
        await provider.get_quote("R")


async def test_server_error_retried_then_success() -> None:
    """A 500 is retried and can succeed."""
    responses = iter(
        [
            httpx.Response(500, json={"error": "boom"}),
            _json({"symbol": "R", "last_price": "1", "timestamp": _NOW.isoformat()}),
        ]
    )
    sleeper = RecordingSleeper()
    provider = _provider(lambda r: next(responses), sleeper=sleeper)

    quote = await provider.get_quote("R")
    assert quote.symbol == "R"
    assert len(sleeper.delays) == 1


async def test_provider_satisfies_interface() -> None:
    """The real provider is a MarketDataProvider."""
    from app.providers.base import MarketDataProvider

    provider = _provider(lambda r: _json({}))
    assert isinstance(provider, MarketDataProvider)
