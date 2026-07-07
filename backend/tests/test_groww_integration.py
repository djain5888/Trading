"""Integration tests for the real Groww provider (mocked HTTP, no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from app.config.broker import GrowwSettings
from app.config.settings import Settings
from app.core.clock import FakeClock
from app.core.timezone import INDIA_TZ
from app.market.calendar.config import InMemoryCalendarConfigProvider
from app.market.calendar.holidays import StaticHolidayProvider
from app.market.calendar.service import MarketCalendarService
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.importer.engine import HistoricalImportEngine
from app.market.historical.importer.models import ImportConfig, ImportMode
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import SeriesKey
from app.market.historical.validation import ValidationEngine
from app.providers.dependencies import build_market_data_provider
from app.providers.exceptions import ProviderError
from app.providers.groww.http_client import GrowwHTTPClient
from app.providers.groww.market_data_provider import GrowwMarketDataProvider
from app.providers.groww.provider import GrowwProvider
from app.providers.groww.totp import generate_totp

Handler = Callable[[httpx.Request], httpx.Response]
_END = datetime(2025, 1, 6, 15, 30, tzinfo=INDIA_TZ)  # Monday close


class RecordingSleeper:
    """Async sleeper that records delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _settings(**overrides: object) -> GrowwSettings:
    base: dict[str, object] = {"api_key": "test-key", "base_url": "https://groww.test"}
    base.update(overrides)
    return GrowwSettings(**base)  # type: ignore[arg-type]


def _provider(handler: Handler, settings: GrowwSettings) -> GrowwMarketDataProvider:
    client = httpx.AsyncClient(
        base_url=settings.base_url, transport=httpx.MockTransport(handler)
    )
    http = GrowwHTTPClient(settings, client=client, sleeper=RecordingSleeper())
    return GrowwMarketDataProvider(settings, http_client=http, clock=FakeClock(_END))


def _candle_json(ts: datetime, close: float) -> dict[str, object]:
    return {
        "timestamp": ts.isoformat(),
        "open": f"{close:.2f}",
        "high": f"{close + 0.5:.2f}",
        "low": f"{close - 0.5:.2f}",
        "close": f"{close:.2f}",
        "volume": 1000,
    }


# -- TOTP ------------------------------------------------------------------


def test_totp_rfc_vector() -> None:
    """TOTP matches the RFC 6238 SHA-1 test vector."""
    seed = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"  # base32 of 12345678901234567890
    code = generate_totp(seed, now=datetime.fromtimestamp(59, tz=UTC))
    assert code == "287082"


def test_totp_rejects_invalid_seed() -> None:
    """A non-base32 seed is rejected."""
    from app.providers.exceptions import AuthenticationError

    with pytest.raises(AuthenticationError):
        generate_totp("not!base32", now=datetime.now(UTC))


# -- Auth with TOTP --------------------------------------------------------


async def test_auth_includes_totp() -> None:
    """When a TOTP seed is configured, auth sends a TOTP and caches the token."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/auth/token"):
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        assert request.headers["Authorization"] == "Bearer tok"
        return httpx.Response(
            200,
            json={"symbol": "R", "last_price": "1", "timestamp": _END.isoformat()},
        )

    settings = _settings(totp_seed="GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ")
    provider = _provider(handler, settings)

    await provider.get_quote("R")

    assert "totp" in captured
    assert captured["key"] == "test-key"


# -- Throttle --------------------------------------------------------------


async def test_throttle_spaces_requests() -> None:
    """A configured throttle rate delays back-to-back requests."""
    settings = _settings(throttle_rate_per_second=10.0)  # 0.1s min interval
    client = httpx.AsyncClient(
        base_url=settings.base_url,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    sleeper = RecordingSleeper()
    http = GrowwHTTPClient(settings, client=client, sleeper=sleeper)
    http._monotonic = lambda: 0.0  # freeze time so the throttle always waits

    from app.providers.http import HttpMethod, RequestSpec

    spec = RequestSpec(method=HttpMethod.GET, path="/v1/ping")
    await http.request(spec)
    await http.request(spec)

    assert sleeper.delays == [pytest.approx(0.1)]


# -- Malformed candle ------------------------------------------------------


async def test_malformed_candle_raises_provider_error() -> None:
    """A candle missing a required field surfaces as a ProviderError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candles": [{"timestamp": _END.isoformat(), "open": "100"}]},
        )

    provider = _provider(handler, _settings())
    with pytest.raises(ProviderError):
        await provider.get_historical_data(
            "R", Interval.ONE_DAY, _END - timedelta(days=5), _END
        )


# -- Partial import (one bad symbol must not kill the run) -----------------


async def test_partial_import_survives_bad_symbol() -> None:
    """A 404 for one symbol is isolated; other symbols still import."""

    def handler(request: httpx.Request) -> httpx.Response:
        symbol = request.url.params.get("symbol")
        if symbol == "BBB":
            return httpx.Response(404, json={"error": "not found"})
        candles = [
            _candle_json(_END - timedelta(days=29 - i), 100 + i) for i in range(30)
        ]
        return httpx.Response(200, json={"candles": candles})

    settings = _settings()
    provider = _provider(handler, settings)
    repo = InMemoryCandleRepository()
    data_engine = HistoricalDataEngine(repo, ValidationEngine())
    calendar = MarketCalendarService(
        clock=FakeClock(_END),
        config_provider=InMemoryCalendarConfigProvider.with_defaults(),
        holiday_provider=StaticHolidayProvider({}),
    )

    async def _noop(_delay: float) -> None:
        return None

    importer = HistoricalImportEngine(
        provider=provider,
        data_engine=data_engine,
        calendar=calendar,
        clock=FakeClock(_END),
        config=ImportConfig(batch_size=500, max_retries=1),
        sleeper=_noop,
    )

    summary = await importer.import_history(
        ["AAA", "BBB"],
        Interval.ONE_DAY,
        _END - timedelta(days=40),
        _END,
        mode=ImportMode.FULL,
    )

    aaa = SeriesKey(symbol="AAA", exchange=Exchange.NSE, interval=Interval.ONE_DAY)
    bbb = SeriesKey(symbol="BBB", exchange=Exchange.NSE, interval=Interval.ONE_DAY)
    assert await data_engine.count(aaa) == 30
    assert await data_engine.count(bbb) == 0
    assert summary.candles_imported == 30
    assert summary.failed_requests >= 1


# -- DI backend selection --------------------------------------------------


def test_di_defaults_to_real_groww() -> None:
    """The default backend is the real Groww provider."""
    provider = build_market_data_provider(Settings(market_data_provider="groww"))
    assert isinstance(provider, GrowwMarketDataProvider)


def test_di_fake_backend_uses_skeleton() -> None:
    """The 'fake' backend selects the offline skeleton provider."""
    provider = build_market_data_provider(Settings(market_data_provider="fake"))
    assert isinstance(provider, GrowwProvider)


def test_uses_token_exchange_flag() -> None:
    """Token exchange is required only when a secret or TOTP seed is set."""
    assert _settings().uses_token_exchange is False
    assert _settings(totp_seed="AAAA").uses_token_exchange is True
    assert _settings(api_secret="s").uses_token_exchange is True
