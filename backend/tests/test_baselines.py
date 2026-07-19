"""Tests for the baseline edge search (entry/exit logic and comparability)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from app.backtest.baselines.base import BaselineContext
from app.backtest.baselines.dependencies import create_baseline
from app.backtest.baselines.engine import BaselineEngine
from app.backtest.baselines.models import BaselineConfig, BaselineName
from app.backtest.baselines.strategies import MeanRevert, Momentum121, TopRSWeekly
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
