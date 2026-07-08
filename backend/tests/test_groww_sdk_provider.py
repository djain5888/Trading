"""Unit tests for the growwapi-SDK-backed provider (mocked client, no network)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest
from app.config.broker import GrowwSettings
from app.config.settings import Settings
from app.core.clock import FakeClock
from app.core.timezone import INDIA_TZ
from app.market.enums import Exchange, Interval
from app.providers.dependencies import build_market_data_provider
from app.providers.exceptions import ProviderError
from app.providers.groww.sdk_provider import GrowwSDKProvider

_NOW = datetime(2025, 1, 6, 15, 30, tzinfo=INDIA_TZ)


class FakeGrowwClient:
    """A stand-in for ``growwapi.GrowwAPI`` recording calls and returning fixtures."""

    def __init__(
        self,
        *,
        candles: dict[str, list[list[float]]] | None = None,
        ltp: dict[str, float] | None = None,
        fail_symbols: frozenset[str] = frozenset(),
    ) -> None:
        self._candles = candles or {}
        self._ltp = ltp or {}
        self._fail_symbols = fail_symbols
        self.historical_calls: list[dict[str, Any]] = []
        self.ltp_calls: list[tuple[str, ...]] = []

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
                "start_time": start_time,
                "end_time": end_time,
                "interval_in_minutes": interval_in_minutes,
            }
        )
        if trading_symbol in self._fail_symbols:
            raise RuntimeError("SDK exploded")
        return {"candles": self._candles.get(trading_symbol, [])}

    def get_ltp(
        self, *, segment: str, exchange_trading_symbols: tuple[str, ...]
    ) -> Mapping[str, Any]:
        self.ltp_calls.append(exchange_trading_symbols)
        result: dict[str, float] = {}
        for key in exchange_trading_symbols:
            symbol = key.split("_", 1)[1]
            if symbol in self._fail_symbols:
                raise RuntimeError("SDK exploded")
            if key in self._ltp:
                result[key] = self._ltp[key]
        return result


def _settings(**overrides: object) -> GrowwSettings:
    base: dict[str, object] = {"api_key": "test-key"}
    base.update(overrides)
    return GrowwSettings(**base)  # type: ignore[arg-type]


def _provider(client: FakeGrowwClient) -> GrowwSDKProvider:
    return GrowwSDKProvider(_settings(), client=client, clock=FakeClock(_NOW))


def _row(epoch: int, close: float) -> list[float]:
    return [epoch, close, close + 1.0, close - 1.0, close, 1000]


# -- Historical daily candles ----------------------------------------------


async def test_historical_daily_candles_are_mapped() -> None:
    """SDK candle rows map into the canonical Candle model, daily interval."""
    rows = [_row(1_700_000_000 + i * 86_400, 100 + i) for i in range(3)]
    client = FakeGrowwClient(candles={"RELIANCE": rows})
    provider = _provider(client)

    data = await provider.get_historical_data(
        "reliance",
        Interval.ONE_DAY,
        datetime(2025, 1, 1, tzinfo=INDIA_TZ),
        _NOW,
    )

    assert data.symbol == "RELIANCE"
    assert data.exchange is Exchange.NSE
    assert data.interval is Interval.ONE_DAY
    assert len(data.candles) == 3
    assert data.candles[0].close == Decimal("100")
    assert data.candles[0].high == Decimal("101")
    assert data.candles[0].timestamp.tzinfo is not None
    # Daily maps to a 1440-minute request.
    assert client.historical_calls[0]["interval_in_minutes"] == 1440
    assert client.historical_calls[0]["trading_symbol"] == "RELIANCE"


async def test_historical_sdk_failure_becomes_provider_error() -> None:
    """A raised SDK error is isolated as a ProviderError (per-symbol failure)."""
    client = FakeGrowwClient(fail_symbols=frozenset({"BADSYM"}))
    provider = _provider(client)

    with pytest.raises(ProviderError):
        await provider.get_historical_data(
            "badsym", Interval.ONE_DAY, datetime(2025, 1, 1, tzinfo=INDIA_TZ), _NOW
        )


async def test_malformed_candle_row_raises_provider_error() -> None:
    """A short/garbled candle row surfaces as a ProviderError."""
    client = FakeGrowwClient(candles={"RELIANCE": [[1_700_000_000, 100.0]]})
    provider = _provider(client)

    with pytest.raises(ProviderError):
        await provider.get_historical_data(
            "RELIANCE", Interval.ONE_DAY, datetime(2025, 1, 1, tzinfo=INDIA_TZ), _NOW
        )


# -- Latest price (LTP) ----------------------------------------------------


async def test_get_quote_returns_ltp() -> None:
    """get_quote maps the SDK LTP into a Quote with the clock timestamp."""
    client = FakeGrowwClient(ltp={"NSE_RELIANCE": 2500.5})
    provider = _provider(client)

    quote = await provider.get_quote("reliance")

    assert quote.symbol == "RELIANCE"
    assert quote.last_price == Decimal("2500.5")
    assert quote.timestamp == _NOW
    assert client.ltp_calls == [("NSE_RELIANCE",)]


async def test_get_quotes_isolates_per_symbol_failure() -> None:
    """A failing symbol is skipped; the rest of the watchlist still resolves."""
    client = FakeGrowwClient(
        ltp={"NSE_AAA": 10.0, "NSE_CCC": 30.0},
        fail_symbols=frozenset({"BBB"}),
    )
    provider = _provider(client)

    quotes = await provider.get_quotes(["AAA", "BBB", "CCC"])

    symbols = {quote.symbol for quote in quotes}
    assert symbols == {"AAA", "CCC"}  # BBB failed and was dropped, not raised


async def test_unsupported_operations_raise_provider_error() -> None:
    """Operations outside the SDK path fail clearly rather than silently."""
    provider = _provider(FakeGrowwClient())

    with pytest.raises(ProviderError):
        await provider.get_market_status()
    with pytest.raises(ProviderError):
        await provider.search_symbol("rel")
    with pytest.raises(ProviderError):
        await provider.get_market_depth("RELIANCE")


# -- DI backend selection --------------------------------------------------


def test_di_selects_sdk_backend() -> None:
    """MARKET_DATA_PROVIDER=groww_sdk builds the SDK provider."""
    provider = build_market_data_provider(Settings(market_data_provider="groww_sdk"))
    assert isinstance(provider, GrowwSDKProvider)
