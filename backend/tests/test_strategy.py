"""Unit tests for the strategy engine and its three strategies."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from app.core.clock import FakeClock
from app.core.timezone import INDIA_TZ
from app.indicators.dependencies import get_indicator_registry
from app.indicators.engine import IndicatorEngine
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import Candle
from app.market.historical.validation import ValidationEngine
from app.market.regime.models import Regime, RegimeReport
from app.strategy.base import StrategyContext
from app.strategy.engine import StrategyEngine
from app.strategy.models import StrategyConfig
from app.strategy.strategies import (
    BreakoutStrategy,
    MomentumStrategy,
    PullbackStrategy,
)

_END = datetime(2025, 1, 6, 15, 30, tzinfo=INDIA_TZ)

_CONFIG = StrategyConfig(
    ema_fast=3,
    ema_mid=5,
    ema_slow=8,
    macd_fast=3,
    macd_slow=6,
    macd_signal=3,
    rsi_period=5,
    rsi_momentum_low=55.0,
    rsi_momentum_high=100.0,
    pullback_fast=5,
    pullback_slow=8,
    pullback_touch_pct=1.5,
    breakout_lookback=5,
    volume_period=5,
    breakout_volume_factor=1.5,
    atr_period=3,
    stop_atr_mult=2.0,
    reward_risk=2.0,
    history_days=100,
    sector_map={"AAA": "IT"},
)

Bar = tuple[float, float, float, float, float]  # open, high, low, close, volume


def _candles(rows: Sequence[Bar], symbol: str = "AAA") -> list[Candle]:
    """Build daily candles from explicit OHLCV rows (oldest first)."""
    count = len(rows)
    return [
        Candle(
            symbol=symbol,
            exchange=Exchange.NSE,
            interval=Interval.ONE_DAY,
            timestamp=_END - timedelta(days=count - 1 - i),
            open=Decimal(f"{o:.4f}"),
            high=Decimal(f"{h:.4f}"),
            low=Decimal(f"{low:.4f}"),
            close=Decimal(f"{c:.4f}"),
            volume=int(v),
        )
        for i, (o, h, low, c, v) in enumerate(rows)
    ]


def _indicators() -> IndicatorEngine:
    repo = InMemoryCandleRepository()
    data = HistoricalDataEngine(repo, ValidationEngine())
    return IndicatorEngine(data, FakeClock(_END), get_indicator_registry())


def _context(
    rows: Sequence[Bar],
    *,
    config: StrategyConfig = _CONFIG,
    regime: RegimeReport | None = None,
    sector_strength: float | None = None,
) -> StrategyContext:
    return StrategyContext(
        symbol="AAA",
        candles=tuple(_candles(rows)),
        indicator_engine=_indicators(),
        scanner_hits=frozenset(),
        regime=regime,
        sector_strength=sector_strength,
        rs=None,
        config=config,
    )


# Scenario builders --------------------------------------------------------


def _uptrend() -> list[Bar]:
    """A strong, accelerating uptrend with flat volume (momentum setup)."""
    rows: list[Bar] = []
    price = 100.0
    step = 2.0
    for _ in range(16):
        step += 0.2  # accelerate, so MACD line stays above its signal
        price += step
        rows.append((price - 1.5, price + 0.5, price - 2.0, price, 1000))
    return rows


def _breakout() -> list[Bar]:
    """A flat consolidation, then a gap-up breakout bar on a volume spike."""
    rows: list[Bar] = [(100.0, 100.5, 99.5, 100.0, 1000) for _ in range(9)]
    rows.append((108.0, 112.0, 107.5, 111.0, 4000))  # gap break + 4x volume
    return rows


def _pullback() -> list[Bar]:
    """An uptrend that dips to the fast EMA and closes back above it."""
    rows: list[Bar] = []
    price = 100.0
    for _ in range(9):
        price += 1.5
        rows.append((price - 1.0, price + 0.5, price - 1.0, price, 1000))
    # Two-bar dip, then a bullish recovery bar that reclaims the fast EMA.
    rows.append((113.0, 113.5, 109.5, 110.0, 1000))
    rows.append((109.5, 110.5, 108.5, 109.0, 1000))
    rows.append((109.5, 114.0, 109.0, 113.5, 1000))
    return rows


# -- Trigger isolation -----------------------------------------------------


def test_breakout_triggers_only_breakout() -> None:
    """The breakout setup fires on a break + volume, and the others do not."""
    ctx = _context(_breakout())
    assert BreakoutStrategy().evaluate(ctx) is not None
    assert PullbackStrategy().evaluate(ctx) is None
    assert MomentumStrategy().evaluate(ctx) is None


def test_momentum_triggers_only_momentum() -> None:
    """The momentum setup fires on a strong aligned trend, and the others do not."""
    ctx = _context(_uptrend())
    assert MomentumStrategy().evaluate(ctx) is not None
    assert BreakoutStrategy().evaluate(ctx) is None  # flat volume, no expansion
    assert PullbackStrategy().evaluate(ctx) is None  # price far above the MA


def test_pullback_triggers_only_pullback() -> None:
    """The pullback setup fires on a retrace-and-reclaim, and the others do not."""
    ctx = _context(_pullback())
    assert PullbackStrategy().evaluate(ctx) is not None
    assert BreakoutStrategy().evaluate(ctx) is None
    assert MomentumStrategy().evaluate(ctx) is None


# -- Engine: RR / stop / target maths --------------------------------------


async def _engine(
    config: StrategyConfig = _CONFIG,
) -> tuple[StrategyEngine, HistoricalDataEngine]:
    repo = InMemoryCandleRepository()
    data = HistoricalDataEngine(repo, ValidationEngine())
    indicators = IndicatorEngine(data, FakeClock(_END), get_indicator_registry())
    engine = StrategyEngine(data, indicators, FakeClock(_END), config)
    return engine, data


async def test_rr_stop_target_from_atr() -> None:
    """Stop = entry - N*ATR, target = entry + RR*risk, RR ratio equals config."""
    engine, data = await _engine()
    await data.import_candles(_candles(_breakout()))

    report = await engine.analyze(["AAA"], interval=Interval.ONE_DAY, end=_END)

    assert report.available
    setup = next(s for s in report.setups if s.strategy == "breakout")
    risk = setup.entry - setup.stop
    assert risk > 0
    assert setup.stop < setup.entry < setup.target
    # target - entry is RR times the risk (allowing for 2dp rounding).
    assert setup.target - setup.entry == pytest.approx(
        _CONFIG.reward_risk * risk, abs=0.02
    )
    assert setup.reward_risk == _CONFIG.reward_risk


# -- Engine: confidence blending -------------------------------------------


async def test_confidence_bull_beats_bear() -> None:
    """A bullish setup scores higher in a bull regime than in a bear regime."""
    engine, data = await _engine()
    await data.import_candles(_candles(_breakout()))
    bull = _regime(Regime.BULL)
    bear = _regime(Regime.BEAR)

    hi = await engine.analyze(["AAA"], interval=Interval.ONE_DAY, end=_END, regime=bull)
    lo = await engine.analyze(["AAA"], interval=Interval.ONE_DAY, end=_END, regime=bear)

    assert hi.setups[0].confidence > lo.setups[0].confidence


async def test_confidence_neutral_without_context() -> None:
    """Missing regime/sector/RS blends with neutral weights, never failing."""
    engine, data = await _engine()
    await data.import_candles(_candles(_breakout()))

    report = await engine.analyze(["AAA"], interval=Interval.ONE_DAY, end=_END)

    assert report.available
    assert report.setups
    assert all(0 <= s.confidence <= 100 for s in report.setups)


# -- Engine: degradation ---------------------------------------------------


async def test_short_history_symbol_excluded() -> None:
    """A symbol with too little history is excluded, not failed."""
    engine, data = await _engine()
    await data.import_candles(_candles(_breakout()[:4], symbol="AAA"))

    report = await engine.analyze(["AAA"], interval=Interval.ONE_DAY, end=_END)

    assert report.available is False
    assert report.setups == ()
    assert "unavailable" in report.detail.lower()


def _regime(kind: Regime) -> RegimeReport:
    from app.market.regime.models import Volatility

    return RegimeReport(
        available=True,
        regime=kind,
        confidence=100.0,
        volatility=Volatility.NORMAL,
        breadth=50.0,
        index_symbol="NIFTY",
        detail="test",
        generated_at=_END,
    )
