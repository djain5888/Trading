"""Unit tests for the indicator engine."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from app.core.clock import SystemClock
from app.core.timezone import INDIA_TZ
from app.indicators.calculators import EmaCalculator, SmaCalculator
from app.indicators.engine import IndicatorEngine
from app.indicators.exceptions import (
    IndicatorError,
    InsufficientHistoryError,
    InvalidPeriodError,
    UnknownIndicatorError,
)
from app.indicators.models import PriceSeries
from app.indicators.registry import default_registry
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import Candle, SeriesKey
from app.market.historical.validation import ValidationEngine

_BASE = datetime(2025, 1, 6, 9, 15, tzinfo=INDIA_TZ)


def _candle(i: int, *, o: str, h: str, low: str, c: str, v: int = 1000) -> Candle:
    """Build a candle at minute offset ``i``."""
    return Candle(
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        interval=Interval.ONE_MINUTE,
        timestamp=_BASE + timedelta(minutes=i),
        open=Decimal(o),
        high=Decimal(h),
        low=Decimal(low),
        close=Decimal(c),
        volume=v,
    )


def _closes(values: Sequence[float]) -> list[Candle]:
    """Build candles from a close series (OHLC flat around close)."""
    out: list[Candle] = []
    for i, close in enumerate(values):
        price = f"{close:.4f}"
        high = f"{close + 0.5:.4f}"
        low = f"{close - 0.5:.4f}"
        out.append(_candle(i, o=price, h=high, low=low, c=price))
    return out


def _series(candles: Sequence[Candle]) -> PriceSeries:
    return PriceSeries.from_candles(candles)


def _engine() -> IndicatorEngine:
    return IndicatorEngine(
        HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine()),
        SystemClock(),
        default_registry(),
    )


# -- Registry --------------------------------------------------------------


def test_registry_lists_all_indicators() -> None:
    """All 17 built-in indicators are registered."""
    names = default_registry().available()
    assert len(names) == 17
    assert {"sma", "ema", "rsi", "macd", "adx", "atr", "vwap", "obv"} <= set(names)


def test_unknown_indicator_raises() -> None:
    """An unregistered indicator name is rejected."""
    with pytest.raises(UnknownIndicatorError):
        default_registry().create("does_not_exist")


# -- Moving averages -------------------------------------------------------


def test_sma() -> None:
    """SMA matches the manual rolling mean."""
    candles = _closes([1, 2, 3, 4, 5])
    results = SmaCalculator(period=3).calculate(_series(candles))
    assert [r.value for r in results] == pytest.approx([2.0, 3.0, 4.0])
    assert results[0].lookback == 3
    assert results[0].metadata["period"] == 3.0


def test_ema() -> None:
    """EMA seeds with SMA then applies the recursive formula."""
    candles = _closes([1, 2, 3, 4, 5])
    results = EmaCalculator(period=3).calculate(_series(candles))
    # seed = mean(1,2,3)=2; alpha=0.5; next = 4*0.5+2*0.5=3; then 5*0.5+3*0.5=4
    assert [r.value for r in results] == pytest.approx([2.0, 3.0, 4.0])


# -- Momentum --------------------------------------------------------------


def test_rsi_all_gains_is_100() -> None:
    """A monotonically rising series has RSI 100."""
    candles = _closes([float(x) for x in range(1, 20)])
    results = default_registry().create("rsi", period=14).calculate(_series(candles))
    assert results[-1].value == pytest.approx(100.0)


def test_roc() -> None:
    """ROC is the percentage change over the period."""
    candles = _closes([10, 11, 12, 13])
    results = default_registry().create("roc", period=2).calculate(_series(candles))
    # index2: (12/10-1)*100=20 ; index3: (13/11-1)*100≈18.18
    assert results[0].value == pytest.approx(20.0)


def test_momentum() -> None:
    """Momentum is the absolute difference over the period."""
    candles = _closes([10, 11, 13, 16])
    results = (
        default_registry().create("momentum", period=1).calculate(_series(candles))
    )
    assert [r.value for r in results] == pytest.approx([1.0, 2.0, 3.0])


# -- Trend -----------------------------------------------------------------


def test_macd_components() -> None:
    """MACD exposes signal and histogram in metadata."""
    candles = _closes([float(x) for x in range(1, 60)])
    results = (
        default_registry()
        .create("macd", fast=12, slow=26, signal=9)
        .calculate(_series(candles))
    )
    last = results[-1]
    assert "signal" in last.metadata
    assert "histogram" in last.metadata
    assert last.metadata["histogram"] == pytest.approx(
        last.value - last.metadata["signal"]
    )


def test_adx_and_dmi() -> None:
    """ADX/DMI produce finite directional values on a trending series."""
    candles = _closes([float(x) for x in range(1, 60)])
    adx = default_registry().create("adx", period=14).calculate(_series(candles))
    dmi = default_registry().create("dmi", period=14).calculate(_series(candles))
    assert adx and math.isfinite(adx[-1].value)
    assert "plus_di" in adx[-1].metadata
    assert dmi and "minus_di" in dmi[-1].metadata


# -- Volatility ------------------------------------------------------------


def test_atr() -> None:
    """ATR is positive on a series with range."""
    candles = _closes([float(x) for x in range(1, 30)])
    results = default_registry().create("atr", period=14).calculate(_series(candles))
    assert results[-1].value > 0


def test_bollinger_bands() -> None:
    """Bollinger upper/lower straddle the middle band."""
    candles = _closes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    results = (
        default_registry()
        .create("bollinger", period=5, multiplier=2.0)
        .calculate(_series(candles))
    )
    last = results[-1]
    assert last.metadata["lower"] < last.value < last.metadata["upper"]


def test_stddev() -> None:
    """Standard deviation of a constant series is zero."""
    candles = _closes([5, 5, 5, 5, 5])
    results = default_registry().create("stddev", period=3).calculate(_series(candles))
    assert results[-1].value == pytest.approx(0.0)


# -- Volume ----------------------------------------------------------------


def test_vwap() -> None:
    """VWAP is defined from the first candle."""
    candles = _closes([10, 20, 30])
    results = default_registry().create("vwap").calculate(_series(candles))
    assert len(results) == 3
    assert all(math.isfinite(r.value) for r in results)


def test_obv() -> None:
    """OBV accumulates volume by close direction."""
    candles = [
        _candle(0, o="10", h="11", low="9", c="10", v=100),
        _candle(1, o="10", h="12", low="9", c="11", v=200),  # up -> +200
        _candle(2, o="11", h="12", low="9", c="10", v=150),  # down -> -150
    ]
    results = default_registry().create("obv").calculate(_series(candles))
    assert [r.value for r in results] == pytest.approx([0.0, 200.0, 50.0])


def test_volume_sma() -> None:
    """Volume SMA averages the volume column."""
    candles = [
        _candle(i, o="10", h="11", low="9", c="10", v=v)
        for i, v in enumerate([100, 200, 300])
    ]
    results = (
        default_registry().create("volume_sma", period=3).calculate(_series(candles))
    )
    assert results[-1].value == pytest.approx(200.0)


# -- Price transforms ------------------------------------------------------


def test_price_transforms() -> None:
    """Typical, median and weighted-close use the H/L/C formulas."""
    candles = [_candle(0, o="10", h="12", low="8", c="11")]
    reg = default_registry()
    typical = reg.create("typical_price").calculate(_series(candles))[0]
    median = reg.create("median_price").calculate(_series(candles))[0]
    weighted = reg.create("weighted_close").calculate(_series(candles))[0]
    assert typical.value == pytest.approx((12 + 8 + 11) / 3)
    assert median.value == pytest.approx((12 + 8) / 2)
    assert weighted.value == pytest.approx((12 + 8 + 2 * 11) / 4)


# -- Validation & edge cases -----------------------------------------------


def test_insufficient_history() -> None:
    """Too few candles raises InsufficientHistoryError."""
    candles = _closes([1, 2])
    with pytest.raises(InsufficientHistoryError):
        SmaCalculator(period=5).calculate(_series(candles))


def test_invalid_period() -> None:
    """A non-positive period is rejected at construction."""
    with pytest.raises(InvalidPeriodError):
        SmaCalculator(period=0)


def test_unordered_candles_rejected() -> None:
    """Out-of-order candles are rejected when building the series."""
    candles = [
        _candle(2, o="1", h="2", low="1", c="1"),
        _candle(1, o="1", h="2", low="1", c="1"),
    ]
    with pytest.raises(IndicatorError):
        PriceSeries.from_candles(candles)


# -- Caching & incremental -------------------------------------------------


def test_cache_returns_same_object() -> None:
    """Identical computations are served from cache."""
    engine = _engine()
    candles = _closes([float(x) for x in range(1, 30)])
    first = engine.compute_from_candles(candles, "sma", period=5)
    second = engine.compute_from_candles(candles, "sma", period=5)
    assert first is second


def test_incremental_matches_full() -> None:
    """Incremental extension equals a full recompute for a window indicator."""
    engine = _engine()
    candles = _closes([float(x) for x in range(1, 40)])
    prefix = engine.compute_from_candles(candles[:-5], "sma", period=5)
    incremental = engine.compute_incremental(candles, "sma", prefix, period=5)
    full = engine.compute_from_candles(candles, "sma", period=5)
    assert [r.value for r in incremental] == pytest.approx([r.value for r in full])


async def test_compute_reads_from_data_engine() -> None:
    """The async compute path reads candles from the data engine."""
    repo = InMemoryCandleRepository()
    data_engine = HistoricalDataEngine(repo, ValidationEngine())
    candles = _closes([float(x) for x in range(1, 30)])
    await repo.save_many(candles)
    engine = IndicatorEngine(data_engine, SystemClock(), default_registry())

    key = SeriesKey(
        symbol="RELIANCE", exchange=Exchange.NSE, interval=Interval.ONE_MINUTE
    )
    results = await engine.compute(
        key, "sma", candles[0].timestamp, candles[-1].timestamp, period=5
    )
    assert len(results) == len(candles) - 4


# -- Performance -----------------------------------------------------------


def test_large_dataset() -> None:
    """Indicators compute over a large series without error."""
    engine = _engine()
    candles = _closes([float((x % 100) + 1) for x in range(20_000)])
    results = engine.compute_from_candles(candles, "sma", period=50)
    assert len(results) == 20_000 - 49


def test_di_singleton() -> None:
    """The DI accessor returns a cached singleton engine."""
    from app.indicators.dependencies import (
        _engine_singleton,
        get_indicator_engine,
    )

    _engine_singleton.cache_clear()
    try:
        assert get_indicator_engine() is get_indicator_engine()
    finally:
        _engine_singleton.cache_clear()
