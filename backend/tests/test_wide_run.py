"""Tests for TASK-030: aggregated leverage-cap logging and tier selection."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from app.backtest.baselines.base import BaselineContext, BaselinePlan, EntryIntent
from app.backtest.baselines.engine import BaselineEngine
from app.backtest.baselines.models import BaselineConfig
from app.backtest.guard import LookaheadGuard
from app.backtest.models import BacktestConfig
from app.backtest.validation import partition_by_tier
from app.cli.main import TierChoice, _quiet_import_logs
from app.config.watchlist import CAP_TIERS, load_wide_universe
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
_ENGINE_LOGGER = "app.backtest.baselines.engine"


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
    await data.import_candles(
        [
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
    )


def _trading_days(count: int) -> list[date]:
    cal = _calendar()
    days: list[date] = []
    day = date(2022, 1, 1)
    while len(days) < count:
        if cal.is_trading_day(day, Exchange.NSE):
            days.append(day)
        day += timedelta(days=1)
    return days


class _BatchEntryBaseline:
    """A baseline that requests a whole batch of oversized entries on day one.

    Each intent's stop distance is tiny, so 1%-risk sizing makes a single
    position more than fill the book; the first fills it and every later intent
    in the same batch is turned away by the 1x cap — the flood we aggregate.
    """

    name = "batch_entry"

    def __init__(self, symbols: Sequence[str]) -> None:
        """Store the symbols to request once, on the first plan call."""
        self._symbols = tuple(symbols)
        self._emitted = False

    async def plan(
        self,
        ctx: BaselineContext,
        day: date,
        universe: Sequence[str],
        held: frozenset[str],
    ) -> BaselinePlan:
        """Emit the full batch once; nothing thereafter."""
        if self._emitted:
            return BaselinePlan()
        self._emitted = True
        return BaselinePlan(
            tuple(
                EntryIntent(
                    symbol=symbol,
                    signal_price=100.0,
                    stop_distance=0.5,  # tiny risk -> enormous 1%-risk size
                    uses_stop=False,
                    trailing_distance=None,
                    max_hold=None,
                )
                for symbol in self._symbols
            )
        )


async def _seed_rising_universe(data: HistoricalDataEngine) -> list[str]:
    """Seed a few gently-rising symbols; return their names."""
    days = _trading_days(60)
    symbols = ["T0", "T1", "T2", "T3"]
    for i, symbol in enumerate(symbols):
        await _seed(
            data, symbol, days, [100.0 + i * 5 + j * 0.05 for j in range(len(days))]
        )
    return symbols


async def _cap_hitting_report(data: HistoricalDataEngine, symbols: list[str]) -> None:
    """Run the batch baseline whose oversized entries repeatedly hit the cap."""
    tuning = BaselineConfig(history_days=30, atr_period=5, trend_ema=20)
    engine = BaselineEngine(
        data=LookaheadGuard(data),
        indicators=IndicatorEngine(
            data, FakeClock(datetime(2026, 7, 16, tzinfo=_TZ)), get_indicator_registry()
        ),
        calendar=_calendar(),
        baseline=_BatchEntryBaseline(symbols),
        config=BacktestConfig(),
        history_days=tuning.history_days,
    )
    days = _trading_days(60)
    await engine.run(symbols, days[35], days[-1])


# -- Aggregated leverage-cap logging ---------------------------------------


async def test_leverage_cap_logs_one_summary_not_per_day(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The cap emits a single INFO summary, never a flood of per-day WARNINGs."""
    data = HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())
    symbols = await _seed_rising_universe(data)

    with caplog.at_level(logging.INFO, logger=_ENGINE_LOGGER):
        await _cap_hitting_report(data, symbols)

    summaries = [r for r in caplog.records if "leverage cap skipped" in r.message]
    assert len(summaries) == 1  # exactly one aggregated line
    assert summaries[0].levelno == logging.INFO
    assert "entries across" in summaries[0].getMessage()
    # The old per-symbol-per-day WARNING flood must be gone at INFO level.
    per_day = [r for r in caplog.records if "Leverage cap hit" in r.message]
    assert per_day == []


async def test_leverage_cap_detail_is_debug_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The per-skip detail still exists, but only at DEBUG (opt-in)."""
    data = HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())
    symbols = await _seed_rising_universe(data)

    with caplog.at_level(logging.DEBUG, logger=_ENGINE_LOGGER):
        await _cap_hitting_report(data, symbols)

    detail = [r for r in caplog.records if "Leverage cap hit" in r.message]
    assert detail  # present at DEBUG
    assert all(r.levelno == logging.DEBUG for r in detail)


# -- Tier selection --------------------------------------------------------


def test_tier_choice_values() -> None:
    """The --tier option offers all three caps plus 'all'."""
    assert {t.value for t in TierChoice} == {"all", "large", "mid", "small"}


def test_single_tier_selection_is_a_strict_subset() -> None:
    """Selecting one tier yields only that tier's symbols — a smaller universe."""
    watchlist = load_wide_universe()
    tiers = dict(watchlist.tiers)
    groups = partition_by_tier(list(watchlist.symbols), tiers)

    total = len(watchlist.symbols)
    seen: set[str] = set()
    for tier in CAP_TIERS:
        members = groups[tier]
        assert members  # every tier is represented
        assert 0 < len(members) < total  # strictly smaller than the full universe
        assert all(tiers[s] == tier for s in members)  # only that tier
        seen.update(members)
    assert seen == set(watchlist.symbols)  # the tiers partition the universe


# -- Quiet logging context manager -----------------------------------------


def test_quiet_import_logs_raises_and_restores_levels() -> None:
    """--quiet raises noisy loggers to WARNING, then restores prior levels."""
    noisy = logging.getLogger("app.providers.groww.sdk_provider")
    noisy.setLevel(logging.INFO)

    with _quiet_import_logs(True):
        assert noisy.level == logging.WARNING
    assert noisy.level == logging.INFO  # restored

    # Disabled: levels are untouched.
    with _quiet_import_logs(False):
        assert noisy.level == logging.INFO
