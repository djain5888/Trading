"""Tests for walk-forward validation: windows, the <30-trade guard, verdicts."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from app.backtest.baselines.models import BaselineConfig, BaselineName
from app.backtest.models import BacktestConfig
from app.backtest.validation import (
    ValidationCell,
    ValidationRunner,
    build_verdict,
    check_history,
    limit_by_tier,
    partition_by_tier,
    split_windows,
)
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

_TZ = INDIA_TZ


# -- Window splitting ------------------------------------------------------


def test_split_windows_are_contiguous_and_non_overlapping() -> None:
    """Three windows tile the range end-to-end with no gaps or overlaps."""
    windows = split_windows(date(2022, 1, 1), date(2024, 12, 31), 3)

    assert [w.index for w in windows] == [0, 1, 2]
    assert windows[0].start == date(2022, 1, 1)
    assert windows[-1].end == date(2024, 12, 31)
    for earlier, later in zip(windows, windows[1:], strict=False):
        assert later.start == earlier.end + timedelta(days=1)  # contiguous, no overlap


def test_split_windows_rejects_bad_inputs() -> None:
    """A non-positive count or a too-short range is rejected."""
    with pytest.raises(ValueError):
        split_windows(date(2022, 1, 1), date(2022, 12, 31), 0)
    with pytest.raises(ValueError):
        split_windows(date(2022, 1, 1), date(2022, 1, 2), 5)


# -- Confirmed-history sufficiency check (TASK-025) ------------------------


def test_check_history_refuses_short_history() -> None:
    """~16 months of data cannot support 3 x 12-month windows — refuse."""
    check = check_history(
        date(2024, 3, 1),
        date(2025, 7, 1),
        lookback_days=430,
        windows=3,
        window_months=12,
    )

    assert check.ok is False
    assert "Refusing to validate" in check.message
    assert "2024-03-01..2025-07-01" in check.message  # states the real window
    assert check.required_days > check.usable_days


def test_check_history_accepts_sufficient_history() -> None:
    """Seven years easily covers 3 x 12-month windows after the lookback."""
    check = check_history(
        date(2018, 1, 1),
        date(2025, 1, 1),
        lookback_days=430,
        windows=3,
        window_months=12,
    )

    assert check.ok is True
    assert check.usable_days >= check.required_days


# -- The <30-trade guard and the majority verdict --------------------------


def _cell(
    strategy: str,
    window: int,
    *,
    trades: int,
    ret: float,
    bench: float,
    min_trades: int = 30,
) -> ValidationCell:
    """Build a cell, applying the evaluable/beats rules exactly as the runner."""
    evaluable = trades >= min_trades
    return ValidationCell(
        strategy=strategy,
        window=window,
        trades=trades,
        expectancy_r=0.0,
        total_return_pct=ret,
        benchmark_return_pct=bench,
        max_drawdown_pct=0.0,
        evaluable=evaluable,
        beats_benchmark=evaluable and ret > bench,
    )


def test_verdict_passes_only_on_a_majority() -> None:
    """Beating the benchmark in 2 of 3 evaluable windows passes."""
    cells = [
        _cell("s", 0, trades=40, ret=12.0, bench=5.0),  # beat
        _cell("s", 1, trades=40, ret=1.0, bench=5.0),  # miss
        _cell("s", 2, trades=40, ret=9.0, bench=5.0),  # beat
    ]

    verdict = build_verdict("s", cells, 3, 30)

    assert verdict.beats_count == 2
    assert verdict.passed is True
    assert verdict.note.startswith("PASS")


def test_verdict_fails_on_a_minority() -> None:
    """Beating in only 1 of 3 windows fails the majority test."""
    cells = [
        _cell("s", 0, trades=40, ret=12.0, bench=5.0),  # beat
        _cell("s", 1, trades=40, ret=1.0, bench=5.0),  # miss
        _cell("s", 2, trades=40, ret=2.0, bench=5.0),  # miss
    ]

    verdict = build_verdict("s", cells, 3, 30)

    assert verdict.beats_count == 1
    assert verdict.passed is False
    assert verdict.note.startswith("FAIL")


def test_verdict_thin_window_cannot_beat() -> None:
    """A window with < min trades is NOT EVALUABLE and never counts as a beat."""
    cells = [
        _cell("s", 0, trades=12, ret=99.0, bench=5.0),  # huge return but too few trades
        _cell("s", 1, trades=40, ret=9.0, bench=5.0),  # beat
        _cell("s", 2, trades=40, ret=1.0, bench=5.0),  # miss
    ]

    verdict = build_verdict("s", cells, 3, 30)

    assert cells[0].evaluable is False
    assert cells[0].beats_benchmark is False  # a lucky thin window cannot promote noise
    assert verdict.beats_count == 1
    assert verdict.passed is False


def test_verdict_all_thin_is_not_evaluable() -> None:
    """A strategy with too few trades everywhere is flagged NOT EVALUABLE."""
    cells = [_cell("s", i, trades=5, ret=50.0, bench=5.0) for i in range(3)]

    verdict = build_verdict("s", cells, 3, 30)

    assert verdict.evaluable_windows == 0
    assert verdict.passed is False
    assert "NOT EVALUABLE" in verdict.note


# -- Integration: the guard flows through the runner -----------------------


def _calendar() -> MarketCalendarService:
    return MarketCalendarService(
        clock=FakeClock(datetime(2026, 7, 16, tzinfo=_TZ)),
        config_provider=InMemoryCalendarConfigProvider.with_defaults(),
        holiday_provider=StaticHolidayProvider({}),
    )


async def _seed(
    data: HistoricalDataEngine,
    symbol: str,
    days: Sequence[date],
    prices: Sequence[float],
) -> None:
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


async def test_runner_flags_thin_windows_not_evaluable() -> None:
    """A small universe yields < 30 trades/window, so the runner flags it."""
    cal = _calendar()
    days: list[date] = []
    day = date(2022, 1, 1)
    while len(days) < 400:
        if cal.is_trading_day(day, Exchange.NSE):
            days.append(day)
        day += timedelta(days=1)
    data = HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())
    for i, symbol in enumerate(("AAA", "BBB", "NIFTY")):
        await _seed(data, symbol, days, [100.0 + i * 5 + j * 0.5 for j in range(400)])

    tuning = BaselineConfig(top_n=1, history_days=120, trend_ema=20, atr_period=5)
    runner = ValidationRunner(
        data_engine=data,
        indicators=IndicatorEngine(
            data, FakeClock(datetime(2026, 7, 16, tzinfo=_TZ)), get_indicator_registry()
        ),
        calendar=cal,
        config=BacktestConfig(),
        baseline_config=tuning,
    )
    span = await runner.stored_range(["AAA", "BBB"])
    assert span is not None
    start = span[0] + timedelta(days=tuning.history_days)

    report = await runner.validate(
        ["AAA", "BBB"],
        [BaselineName.TREND_FOLLOW],
        start,
        span[1],
        window_count=3,
    )

    assert len(report.windows) == 3
    trend_cells = [c for c in report.cells if c.strategy == "trend_follow"]
    assert all(cell.trades < 30 and not cell.evaluable for cell in trend_cells)
    verdict = report.verdicts[0]
    assert "NOT EVALUABLE" in verdict.note
    assert report.passed == ()  # nothing passes on this thin data


# -- Tier segmentation and the tiered matrix -------------------------------


def test_partition_and_limit_are_tier_balanced() -> None:
    """The tier helpers group by cap tier and truncate in a balanced way."""
    symbols = ["A", "B", "C", "D"]
    tiers = {"A": "large", "B": "small", "C": "large", "D": "mid"}

    groups = partition_by_tier(symbols, tiers)
    assert groups == {"large": ["A", "C"], "mid": ["D"], "small": ["B"]}

    limited = limit_by_tier(symbols, tiers, 2)
    assert len(limited) == 2
    # Round-robin picks one large then one mid before a second large.
    assert set(limited) == {"A", "D"}


async def test_validate_tiers_splits_the_matrix_by_tier() -> None:
    """Tiered validation yields an overall report plus one report per cap tier."""
    cal = _calendar()
    days: list[date] = []
    day = date(2022, 1, 1)
    while len(days) < 400:
        if cal.is_trading_day(day, Exchange.NSE):
            days.append(day)
        day += timedelta(days=1)
    data = HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())
    universe = ["AAA", "BBB", "CCC", "DDD"]
    tiers = {"AAA": "large", "BBB": "large", "CCC": "mid", "DDD": "small"}
    for i, symbol in enumerate((*universe, "NIFTY")):
        await _seed(data, symbol, days, [100.0 + i * 5 + j * 0.5 for j in range(400)])

    tuning = BaselineConfig(top_n=1, history_days=120, trend_ema=20, atr_period=5)
    runner = ValidationRunner(
        data_engine=data,
        indicators=IndicatorEngine(
            data, FakeClock(datetime(2026, 7, 16, tzinfo=_TZ)), get_indicator_registry()
        ),
        calendar=cal,
        config=BacktestConfig(),
        baseline_config=tuning,
    )
    span = await runner.stored_range(universe)
    assert span is not None
    start = span[0] + timedelta(days=tuning.history_days)

    tiered = await runner.validate_tiers(
        universe,
        [BaselineName.TREND_FOLLOW],
        start,
        span[1],
        window_count=3,
        tiers=tiers,
    )

    # Overall spans the whole universe; three tiers appear, largest first.
    assert tiered.universe_size == 4
    assert tiered.symbols == tuple(universe)
    assert [t.tier for t in tiered.tiers] == ["large", "mid", "small"]
    assert tiered.tiers[0].symbols == ("AAA", "BBB")
    assert tiered.tiers[1].symbols == ("CCC",)
    assert tiered.tiers[2].symbols == ("DDD",)

    # Matrix correctness: the large-tier report reproduces a stand-alone run
    # over exactly the large-tier symbols and the same windows.
    large = tiered.tiers[0]
    standalone = await runner.validate(
        ["AAA", "BBB"],
        [BaselineName.TREND_FOLLOW],
        start,
        span[1],
        window_count=3,
    )
    assert [c.model_dump() for c in large.report.cells] == [
        c.model_dump() for c in standalone.cells
    ]
    # Every tier shares the identical window layout with the overall run.
    for tier in tiered.tiers:
        assert tier.report.windows == tiered.overall.windows
