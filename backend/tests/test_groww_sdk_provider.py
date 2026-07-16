"""Unit tests for the growwapi-SDK provider (mocked GrowwAPI + pyotp, no network)."""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest
from app.config.broker import GrowwSettings
from app.config.settings import Settings
from app.core.clock import FakeClock
from app.core.timezone import INDIA_TZ
from app.market.enums import Interval
from app.providers.dependencies import build_market_data_provider
from app.providers.exceptions import AuthenticationError, ProviderError
from app.providers.groww.sdk_provider import GrowwSDKProvider, _build_growwapi
from app.workflow.diagnostics import run_health_checks

_NOW = datetime(2025, 1, 6, 15, 30, tzinfo=INDIA_TZ)


class FakeGrowwClient:
    """A stand-in for a built ``growwapi.GrowwAPI`` instance."""

    SEGMENT_CASH = "CASH"
    EXCHANGE_NSE = "NSE"
    EXCHANGE_BSE = "BSE"

    def __init__(
        self,
        *,
        candles: dict[str, list[list[float]]] | None = None,
        ltp: dict[str, float] | None = None,
        forbidden_symbols: frozenset[str] = frozenset(),
    ) -> None:
        self._candles = candles or {}
        self._ltp = ltp or {}
        self._forbidden = forbidden_symbols
        self.historical_calls: list[dict[str, Any]] = []
        self.ltp_calls: list[str] = []

    def get_historical_candle_data(
        self,
        *,
        trading_symbol: str,
        exchange: str,
        segment: str,
        start_time: str,
        end_time: str,
        interval_in_minutes: int,
    ) -> Mapping[str, Any]:
        self.historical_calls.append(
            {
                "trading_symbol": trading_symbol,
                "exchange": exchange,
                "segment": segment,
                "interval_in_minutes": interval_in_minutes,
            }
        )
        if trading_symbol in self._forbidden:
            raise RuntimeError("Access forbidden")
        return {"candles": self._candles.get(trading_symbol, [])}

    def get_ltp(
        self, *, segment: str, exchange_trading_symbols: str
    ) -> Mapping[str, Any]:
        self.ltp_calls.append(exchange_trading_symbols)
        symbol = exchange_trading_symbols.split("_", 1)[1]
        if symbol in self._forbidden:
            raise RuntimeError("Access forbidden")
        if exchange_trading_symbols in self._ltp:
            return {exchange_trading_symbols: self._ltp[exchange_trading_symbols]}
        return {}


def _settings(**overrides: object) -> GrowwSettings:
    base: dict[str, object] = {"access_token": "test-token"}
    base.update(overrides)
    return GrowwSettings(**base)  # type: ignore[arg-type]


def _provider(client: FakeGrowwClient, **settings: object) -> GrowwSDKProvider:
    return GrowwSDKProvider(_settings(**settings), client=client, clock=FakeClock(_NOW))


def _row(epoch: int, close: float) -> list[float]:
    return [epoch, close, close + 1.0, close - 1.0, close, 1000]


# -- Historical daily candles ----------------------------------------------


async def test_historical_daily_candles_are_mapped() -> None:
    """SDK candle rows map into the canonical Candle model, daily interval."""
    rows = [_row(1_700_000_000 + i * 86_400, 100 + i) for i in range(3)]
    client = FakeGrowwClient(candles={"RELIANCE": rows})
    provider = _provider(client)

    data = await provider.get_historical_data(
        "reliance", Interval.ONE_DAY, datetime(2025, 1, 1, tzinfo=INDIA_TZ), _NOW
    )

    assert data.symbol == "RELIANCE"
    assert len(data.candles) == 3
    assert data.candles[0].close == Decimal("100")
    assert data.candles[0].timestamp.tzinfo is not None
    assert client.historical_calls[0]["interval_in_minutes"] == 1440
    assert client.historical_calls[0]["exchange"] == "NSE"  # from EXCHANGE_NSE
    assert client.historical_calls[0]["segment"] == "CASH"  # from SEGMENT_CASH


async def test_index_candles_tolerate_zero_and_missing_volume() -> None:
    """Index rows with zero or absent volume parse cleanly (BUG 1)."""
    rows = [
        [1_700_000_000, 100.0, 101.0, 99.0, 100.5, 0],  # explicit zero volume
        [1_700_086_400, 100.5, 102.0, 100.0, 101.5],  # no volume column at all
    ]
    client = FakeGrowwClient(candles={"NIFTY": rows})
    provider = _provider(client)

    data = await provider.get_historical_data(
        "NIFTY", Interval.ONE_DAY, datetime(2025, 1, 1, tzinfo=INDIA_TZ), _NOW
    )

    assert len(data.candles) == 2
    assert all(candle.volume == 0 for candle in data.candles)
    assert data.candles[1].close == Decimal("101.5")


async def test_malformed_row_is_skipped_not_failed() -> None:
    """A genuinely malformed row is skipped; the series is not failed (BUG 1)."""
    rows = [
        [1_700_000_000, 100.0, 101.0, 99.0, 100.5, 1000],  # valid
        [1_700_086_400, 100.5],  # too short -> skipped
    ]
    client = FakeGrowwClient(candles={"NIFTY": rows})
    provider = _provider(client)

    data = await provider.get_historical_data(
        "NIFTY", Interval.ONE_DAY, datetime(2025, 1, 1, tzinfo=INDIA_TZ), _NOW
    )

    assert len(data.candles) == 1  # bad row dropped, no ProviderError raised


# -- Latest price (LTP) ----------------------------------------------------


async def test_get_quote_returns_ltp() -> None:
    """get_quote maps the SDK LTP into a Quote with the clock timestamp."""
    client = FakeGrowwClient(ltp={"NSE_RELIANCE": 1307.8})
    provider = _provider(client)

    quote = await provider.get_quote("reliance")

    assert quote.symbol == "RELIANCE"
    assert quote.last_price == Decimal("1307.8")
    assert quote.timestamp == _NOW
    assert client.ltp_calls == ["NSE_RELIANCE"]  # single-symbol string, not a tuple


async def test_get_quotes_isolates_per_symbol_failure() -> None:
    """A forbidden symbol is skipped; the rest of the watchlist still resolves."""
    client = FakeGrowwClient(
        ltp={"NSE_AAA": 10.0, "NSE_CCC": 30.0},
        forbidden_symbols=frozenset({"BBB"}),
    )
    provider = _provider(client)

    quotes = await provider.get_quotes(["AAA", "BBB", "CCC"])

    assert {q.symbol for q in quotes} == {"AAA", "CCC"}


async def test_forbidden_maps_to_authentication_error() -> None:
    """An 'Access forbidden' SDK error becomes AuthenticationError."""
    client = FakeGrowwClient(forbidden_symbols=frozenset({"RELIANCE"}))
    provider = _provider(client)

    with pytest.raises(AuthenticationError) as excinfo:
        await provider.get_quote("RELIANCE")
    assert "forbidden" in str(excinfo.value).lower()


async def test_unsupported_operations_raise_provider_error() -> None:
    """Operations outside the SDK path fail clearly rather than silently."""
    provider = _provider(FakeGrowwClient())

    with pytest.raises(ProviderError):
        await provider.get_market_status()
    with pytest.raises(ProviderError):
        await provider.search_symbol("rel")
    with pytest.raises(ProviderError):
        await provider.get_market_depth("RELIANCE")


# -- Client construction / auth --------------------------------------------


class _FakeGrowwAPI:
    """A fake ``growwapi.GrowwAPI`` class for auth-flow tests."""

    SEGMENT_CASH = "CASH"
    EXCHANGE_NSE = "NSE"
    EXCHANGE_BSE = "BSE"
    built_with: str | None = None
    exchange_calls: list[dict[str, str]] = []

    def __init__(self, access_token: str, /) -> None:
        # Positional-only: mirrors the real SDK, which rejects access_token=...
        type(self).built_with = access_token

    @staticmethod
    def get_access_token(*, api_key: str, totp: str) -> str:
        _FakeGrowwAPI.exchange_calls.append({"api_key": api_key, "totp": totp})
        return f"live-{api_key}-{totp}"


class _FakeTOTP:
    """A fake ``pyotp.TOTP`` yielding a fixed code."""

    def __init__(self, secret: str) -> None:
        self.secret = secret

    def now(self) -> str:
        return "654321"


def _install_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeGrowwAPI.built_with = None
    _FakeGrowwAPI.exchange_calls = []
    growwapi_mod = types.ModuleType("growwapi")
    growwapi_mod.GrowwAPI = _FakeGrowwAPI  # type: ignore[attr-defined]
    pyotp_mod = types.ModuleType("pyotp")
    pyotp_mod.TOTP = _FakeTOTP  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "growwapi", growwapi_mod)
    monkeypatch.setitem(sys.modules, "pyotp", pyotp_mod)


def test_build_client_prefers_totp(monkeypatch: pytest.MonkeyPatch) -> None:
    """When TOTP creds are set, the token is regenerated via pyotp + GrowwAPI."""
    _install_sdk(monkeypatch)
    settings = GrowwSettings(totp_token="apikey", totp_secret="SEED")

    _build_growwapi(settings)

    assert _FakeGrowwAPI.exchange_calls == [{"api_key": "apikey", "totp": "654321"}]
    assert _FakeGrowwAPI.built_with == "live-apikey-654321"


def test_build_client_uses_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without TOTP creds, GROWW_ACCESS_TOKEN is used directly (no exchange)."""
    _install_sdk(monkeypatch)
    settings = GrowwSettings(access_token="direct-token")

    _build_growwapi(settings)

    assert _FakeGrowwAPI.built_with == "direct-token"
    assert _FakeGrowwAPI.exchange_calls == []


def test_build_client_without_credentials_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No SDK credentials configured is a clear ProviderError."""
    _install_sdk(monkeypatch)
    with pytest.raises(ProviderError):
        _build_growwapi(GrowwSettings())


# -- DI backend selection --------------------------------------------------


def test_di_defaults_to_sdk_backend() -> None:
    """groww_sdk is the default market-data backend."""
    assert isinstance(build_market_data_provider(Settings()), GrowwSDKProvider)


# -- Health surfacing ------------------------------------------------------


def test_health_sdk_forbidden_is_critical() -> None:
    """An expired/unapproved key surfaces as a critical provider check."""

    async def _forbidden(_groww: GrowwSettings) -> None:
        raise AuthenticationError(
            "Groww access forbidden — API key expired or not approved "
            "(Groww keys reset daily at 6 AM).",
            details="Access forbidden",
        )

    settings = Settings(
        market_data_provider="groww_sdk",
        groww=GrowwSettings(access_token="tok"),
    )
    checks = run_health_checks(settings, auth_probe=_forbidden)
    provider = next(check for check in checks if check.name == "Provider")

    assert provider.healthy is False
    assert provider.critical is True
    assert "forbidden" in provider.detail.lower()


def test_health_sdk_not_configured_is_non_critical() -> None:
    """Missing SDK token is a non-critical 'not configured' state."""
    settings = Settings(market_data_provider="groww_sdk", groww=GrowwSettings())
    checks = run_health_checks(settings)
    provider = next(check for check in checks if check.name == "Provider")

    assert provider.critical is False
    assert "token missing" in provider.detail.lower()


def test_health_sdk_authenticated() -> None:
    """A successful SDK probe yields a healthy provider check."""

    async def _ok(_groww: GrowwSettings) -> None:
        return None

    settings = Settings(
        market_data_provider="groww_sdk", groww=GrowwSettings(access_token="tok")
    )
    checks = run_health_checks(settings, auth_probe=_ok)
    provider = next(check for check in checks if check.name == "Provider")

    assert provider.healthy is True
    assert "sdk authenticated" in provider.detail.lower()
