"""Tests for the baseline edge search (entry/exit logic and comparability)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from app.backtest.baselines.base import BaselineContext, atr_expanding, is_new_high
from app.backtest.baselines.dependencies import create_baseline
from app.backtest.baselines.engine import BaselineEngine
from app.backtest.baselines.models import BaselineConfig, BaselineName
from app.backtest.baselines.strategies import (
    DualMomentum,
    MeanRevert,
    Momentum121,
    RSMomentumCash,
    TopRSWeekly,
    TrendFollow,
    VolatilityBreak,
)
from app.backtest.guard import LookaheadGuard
from app.backtest.models import BacktestConfig, BacktestReport
from app.core.clock import FakeClock
from app.core.timezone import INDIA_TZ
from app.indicators.dependencies import get_indicator_registry
from app.indicators.engine import IndicatorEngine
from app.market.calendar.config import InMemoryCalendarConfigProvider
from app.market.calendar.holidays import StaticHolidayProvider
from app.market.calendar.service import MarketCalendarService
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import Candle
from app.market.historical.validation import ValidationEngine
from app.paper.models import ExitReason

_TZ = INDIA_TZ

#: Small lookbacks so seeded series stay short; the *logic* is what we test.
_TUNING = BaselineConfig(
    top_n=1,
    hold_days=5,
    history_days=200,
    rs_lookback=5,
    momentum_lookback=20,
    momentum_skip=2,
    atr_period=5,
    rsi_period=5,
    trend_ema=20,
    mean_revert_hold=5,
)


def _calendar() -> MarketCalendarService:
    return MarketCalendarService(
        clock=FakeClock(datetime(2026, 7, 16, tzinfo=_TZ)),
        config_provider=InMemoryCalendarConfigProvider.with_defaults(),
        holiday_provider=StaticHolidayProvider({}),
    )


def _data() -> HistoricalDataEngine:
    return HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())


def _clock() -> FakeClock:
    return FakeClock(datetime(2026, 7, 16, tzinfo=_TZ))


def _trading_days(cal: MarketCalendarService, start: date, count: int) -> list[date]:
    days: list[date] = []
    day = start
    while len(days) < count:
        if cal.is_trading_day(day, Exchange.NSE):
            days.append(day)
        day += timedelta(days=1)
    return days


async def _seed(
    data: HistoricalDataEngine,
    symbol: str,
    days: Sequence[date],
    prices: Sequence[float],
) -> None:
    """Seed a daily series whose open and close both equal ``prices``."""
    candles = [
        Candle(
            symbol=symbol,
            exchange=Exchange.NSE,
            interval=Interval.ONE_DAY,
            timestamp=datetime(day.year, day.month, day.day, 9, 15, tzinfo=_TZ),
            open=Decimal(f"{price:.4f}"),
            high=Decimal(f"{price * 1.01:.4f}"),
            low=Decimal(f"{price * 0.99:.4f}"),
            close=Decimal(f"{price:.4f}"),
            volume=100_000,
        )
        for day, price in zip(days, prices, strict=True)
    ]
    await data.import_candles(candles)


def _ctx(data: HistoricalDataEngine, day: date) -> BaselineContext:
    guard = LookaheadGuard(data)
    guard.freeze_at(day)
    indicators = IndicatorEngine(data, _clock(), get_indicator_registry())
    return BaselineContext(guard, indicators, BacktestConfig(), _TUNING.history_days)


async def _run(
    data: HistoricalDataEngine,
    name: BaselineName,
    symbols: Sequence[str],
    start: date,
    end: date,
) -> BacktestReport:
    cal = _calendar()
    guard = LookaheadGuard(data)
    indicators = IndicatorEngine(data, _clock(), get_indicator_registry())
    engine = BaselineEngine(
        data=guard,
        indicators=indicators,
        calendar=cal,
        baseline=create_baseline(name, _TUNING),
        config=BacktestConfig(),
        history_days=_TUNING.history_days,
    )
    return await engine.run(symbols, start, end)


# -- Registry --------------------------------------------------------------


def test_create_baseline_covers_every_name() -> None:
    """Every enumerated baseline is constructible and self-identifies."""
    for name in BaselineName:
        baseline = create_baseline(name)
        assert baseline.name == name.value


# -- BUY_AND_HOLD ----------------------------------------------------------


async def test_buy_and_hold_buys_each_symbol_once_and_holds() -> None:
    """Buy-and-hold enters each symbol on day two and holds to the window end."""
    cal = _calendar()
    days = _trading_days(cal, date(2024, 1, 1), 120)
    data = _data()
    for i, symbol in enumerate(("AAA", "BBB", "CCC")):
        await _seed(data, symbol, days, [100.0 + i * 10 + j * 0.5 for j in range(120)])

    start, end = days[40], days[80]
    report = await _run(
        data, BaselineName.BUY_AND_HOLD, ["AAA", "BBB", "CCC"], start, end
    )

    assert report.trades == 3
    assert {t.symbol for t in report.closed_trades} == {"AAA", "BBB", "CCC"}
    assert all(t.entry_date == days[41] for t in report.closed_trades)  # next-day open
    assert all(t.exit_date == end for t in report.closed_trades)
    assert all(t.exit_kind.value == "end_of_backtest" for t in report.closed_trades)


# -- TOP_RS_WEEKLY ---------------------------------------------------------


async def test_top_rs_weekly_picks_the_strongest_symbol() -> None:
    """The weekly plan selects the single strongest symbol by trailing return."""
    cal = _calendar()
    days = _trading_days(cal, date(2024, 1, 1), 60)
    data = _data()
    # CCC has the steepest recent return, so it must rank first.
    await _seed(data, "AAA", days, [100.0 + j * 0.1 for j in range(60)])
    await _seed(data, "BBB", days, [100.0 + j * 0.5 for j in range(60)])
    await _seed(data, "CCC", days, [100.0 + j * 2.0 for j in range(60)])

    plan = await TopRSWeekly(_TUNING).plan(
        _ctx(data, days[50]), days[50], ["AAA", "BBB", "CCC"], frozenset()
    )

    assert [e.symbol for e in plan.entries] == ["CCC"]
    assert plan.entries[0].max_hold == _TUNING.hold_days
    assert plan.entries[0].uses_stop is False


async def test_top_rs_weekly_times_out_after_the_hold_period() -> None:
    """Weekly entries close on the holding-period timeout, not a stop."""
    cal = _calendar()
    days = _trading_days(cal, date(2024, 1, 1), 90)
    data = _data()
    await _seed(data, "CCC", days, [100.0 + j * 1.5 for j in range(90)])

    report = await _run(data, BaselineName.TOP_RS_WEEKLY, ["CCC"], days[30], days[80])

    assert report.trades > 0
    assert any(t.exit_kind.value == "timeout" for t in report.closed_trades)
    assert all(t.exit_reason is not ExitReason.STOP for t in report.closed_trades)


# -- TOP_RS_TRAILING -------------------------------------------------------


async def test_top_rs_trailing_exits_on_the_trailing_stop() -> None:
    """A run-up then a crash trips the ATR trailing stop (a STOP exit)."""
    cal = _calendar()
    days = _trading_days(cal, date(2024, 1, 1), 120)
    data = _data()
    prices = [100.0 + j * 2.0 for j in range(80)]  # strong uptrend...
    prices += [prices[-1] * (0.90**k) for k in range(1, 41)]  # ...then a crash
    await _seed(data, "CCC", days, prices)

    report = await _run(
        data, BaselineName.TOP_RS_TRAILING, ["CCC"], days[40], days[115]
    )

    assert report.trades > 0
    assert any(t.exit_reason is ExitReason.STOP for t in report.closed_trades)


# -- MOMENTUM_12_1 ---------------------------------------------------------


async def test_momentum_selects_leader_and_drops_the_rest() -> None:
    """Monthly momentum holds the top-N and rebalances losers out."""
    cal = _calendar()
    days = _trading_days(cal, date(2024, 1, 1), 60)
    data = _data()
    await _seed(data, "AAA", days, [100.0 + j * 2.0 for j in range(60)])  # leader
    await _seed(data, "BBB", days, [100.0 + j * 0.2 for j in range(60)])  # laggard

    plan = await Momentum121(_TUNING).plan(
        _ctx(data, days[50]), days[50], ["AAA", "BBB"], frozenset({"BBB"})
    )

    assert [e.symbol for e in plan.entries] == ["AAA"]  # rotate into the leader
    assert "BBB" in plan.exits  # drop the laggard it no longer holds


# -- MEAN_REVERT -----------------------------------------------------------


async def test_mean_revert_enters_oversold_uptrend() -> None:
    """A pullback (RSI < 30) in an uptrend (close > 200-EMA proxy) is bought."""
    cal = _calendar()
    days = _trading_days(cal, date(2024, 1, 1), 40)
    data = _data()
    prices = [100.0 + j * 2.0 for j in range(35)]  # long uptrend
    prices += [prices[-1] - k * 3.0 for k in range(1, 6)]  # 5 sharp down days
    await _seed(data, "CCC", days, prices)

    plan = await MeanRevert(_TUNING).plan(
        _ctx(data, days[39]), days[39], ["CCC"], frozenset()
    )

    assert [e.symbol for e in plan.entries] == ["CCC"]
    assert plan.entries[0].max_hold == _TUNING.mean_revert_hold


async def test_mean_revert_exits_when_rsi_recovers() -> None:
    """A held name whose RSI has climbed back above 50 is flagged to exit."""
    cal = _calendar()
    days = _trading_days(cal, date(2024, 1, 1), 40)
    data = _data()
    prices = [100.0 + j * 2.0 for j in range(40)]  # rising into the day (high RSI)
    await _seed(data, "CCC", days, prices)

    plan = await MeanRevert(_TUNING).plan(
        _ctx(data, days[39]), days[39], ["CCC"], frozenset({"CCC"})
    )

    assert "CCC" in plan.exits
    assert not plan.entries  # nothing new bought while overbought


# -- Comparability & guards ------------------------------------------------


async def test_baseline_report_is_comparable_to_the_backtest() -> None:
    """A baseline yields a full, comparable report tagged with its name."""
    cal = _calendar()
    days = _trading_days(cal, date(2024, 1, 1), 120)
    data = _data()
    for i, symbol in enumerate(("AAA", "BBB")):
        await _seed(data, symbol, days, [100.0 + i * 20 + j for j in range(120)])

    report = await _run(
        data, BaselineName.BUY_AND_HOLD, ["AAA", "BBB"], days[40], days[80]
    )

    # Same metrics surface as the main backtest, tagged with the baseline name.
    assert isinstance(report.execution_leakage.perfect_expectancy_r, float)
    assert report.exit_analysis is not None
    assert {row.strategy for row in report.per_strategy} == {"buy_and_hold"}
    assert report.trades == 2


async def test_baseline_rejects_inverted_range() -> None:
    """An end date before the start is rejected."""
    with pytest.raises(ValueError):
        await _run(
            _data(),
            BaselineName.BUY_AND_HOLD,
            ["AAA"],
            date(2025, 2, 1),
            date(2025, 1, 1),
        )


# -- TASK-024 candidates: entry/exit logic ---------------------------------


def _bar(on: date, close: float) -> Candle:
    """A coherent daily candle (open=close, +/-1% high/low) for helper tests."""
    return Candle(
        symbol="X",
        exchange=Exchange.NSE,
        interval=Interval.ONE_DAY,
        timestamp=datetime(on.year, on.month, on.day, 9, 15, tzinfo=_TZ),
        open=Decimal(f"{close:.2f}"),
        high=Decimal(f"{close * 1.01:.2f}"),
        low=Decimal(f"{close * 0.99:.2f}"),
        close=Decimal(f"{close:.2f}"),
        volume=100_000,
    )


def test_new_high_and_atr_expanding_helpers() -> None:
    """The breakout and volatility-expansion helpers are correct."""
    days = _trading_days(_calendar(), date(2024, 1, 1), 10)
    closes = [100.0] * 9 + [125.0]  # a flat base then a breakout on the last bar
    candles = [_bar(day, close) for day, close in zip(days, closes, strict=True)]

    assert is_new_high(candles, 5) is True  # 125 clears the prior 5-bar high
    assert is_new_high(candles[:-1], 5) is False  # a flat bar makes no new high
    assert atr_expanding([1.0, 1.0, 1.0, 3.0], 2) is True  # 3.0 > 1.0
    assert atr_expanding([3.0, 2.0, 1.0], 2) is False


async def test_trend_follow_enters_above_ema_and_exits_below() -> None:
    """Trend-follow buys above the trend EMA and sells when price drops below it."""
    cal = _calendar()
    days = _trading_days(cal, date(2024, 1, 1), 45)
    data = _data()
    await _seed(data, "CCC", days, [100.0 + j * 2.0 for j in range(45)])  # rising

    plan = await TrendFollow(_TUNING).plan(
        _ctx(data, days[44]), days[44], ["CCC"], frozenset()
    )
    assert [e.symbol for e in plan.entries] == ["CCC"]  # above EMA -> enter

    # Now a series that has fallen well below its EMA at the end.
    data2 = _data()
    prices = [100.0 + j * 2.0 for j in range(35)] + [
        168.0 - k * 10.0 for k in range(1, 11)
    ]
    await _seed(data2, "CCC", days, prices)
    plan2 = await TrendFollow(_TUNING).plan(
        _ctx(data2, days[44]), days[44], ["CCC"], frozenset({"CCC"})
    )
    assert "CCC" in plan2.exits  # dropped below EMA -> exit


async def test_dual_momentum_is_gated_by_the_index() -> None:
    """Dual momentum buys the top-RS name only when the index is above its EMA."""
    cal = _calendar()
    days = _trading_days(cal, date(2024, 1, 1), 45)
    # Bullish index -> entries fire.
    data = _data()
    await _seed(data, "CCC", days, [100.0 + j * 2.0 for j in range(45)])
    await _seed(data, "NIFTY", days, [100.0 + j * 1.0 for j in range(45)])  # above EMA
    plan = await DualMomentum(_TUNING).plan(
        _ctx(data, days[44]), days[44], ["CCC"], frozenset()
    )
    assert [e.symbol for e in plan.entries] == ["CCC"]

    # Bearish index (below its EMA) -> gate closed, no entries.
    data2 = _data()
    await _seed(data2, "CCC", days, [100.0 + j * 2.0 for j in range(45)])
    bear = [100.0 + j * 1.0 for j in range(30)] + [
        130.0 - k * 6.0 for k in range(1, 16)
    ]
    await _seed(data2, "NIFTY", days, bear)
    plan2 = await DualMomentum(_TUNING).plan(
        _ctx(data2, days[44]), days[44], ["CCC"], frozenset()
    )
    assert plan2.entries == ()


async def test_rs_momentum_cash_exits_to_cash_in_a_bear_index() -> None:
    """The cash overlay closes every position when the index turns bearish."""
    cal = _calendar()
    days = _trading_days(cal, date(2024, 1, 1), 45)
    data = _data()
    await _seed(data, "CCC", days, [100.0 + j * 2.0 for j in range(45)])
    bear = [100.0 + j * 1.0 for j in range(30)] + [
        130.0 - k * 6.0 for k in range(1, 16)
    ]
    await _seed(data, "NIFTY", days, bear)

    plan = await RSMomentumCash(_TUNING).plan(
        _ctx(data, days[44]), days[44], ["CCC"], frozenset({"CCC"})
    )

    assert plan.exits == frozenset({"CCC"})  # risk-off: hold cash
    assert plan.entries == ()


async def test_volatility_break_enters_on_expanding_breakout() -> None:
    """A new high confirmed by ATR expansion opens a trailing-stop position."""
    cal = _calendar()
    days = _trading_days(cal, date(2024, 1, 1), 30)
    data = _data()
    # Flat, low-volatility base, then an accelerating breakout on the last bars.
    prices = [100.0] * 24 + [102.0, 105.0, 109.0, 114.0, 120.0, 127.0]
    await _seed(data, "CCC", days, prices)

    tuning = BaselineConfig(
        top_n=1,
        history_days=200,
        atr_period=5,
        vol_breakout_lookback=5,
        vol_expansion_lookback=3,
    )
    plan = await VolatilityBreak(tuning).plan(
        _ctx(data, days[29]), days[29], ["CCC"], frozenset()
    )

    assert [e.symbol for e in plan.entries] == ["CCC"]
    assert plan.entries[0].uses_stop is True  # ridden by a trailing stop
    assert plan.entries[0].trailing_distance is not None
