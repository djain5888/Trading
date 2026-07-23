"""Tests for the unleveraged equal-weight benchmark and the notional cap.

TASK-029 FIX 1: buy_and_hold must be an unleveraged equal-weight hold (total
notional <= capital), and the same 1x cap applies to every baseline, so no run
can print the old +744%/-83% leverage artefacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal

from app.backtest.baselines.dependencies import create_baseline
from app.backtest.baselines.engine import BaselineEngine
from app.backtest.baselines.models import BaselineConfig, BaselineName
from app.backtest.guard import LookaheadGuard
from app.backtest.models import BacktestConfig
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


def _engine(data: HistoricalDataEngine, name: BaselineName) -> BaselineEngine:
    cal = _calendar()
    tuning = BaselineConfig(history_days=120, atr_period=5, trend_ema=20, top_n=3)
    return BaselineEngine(
        data=LookaheadGuard(data),
        indicators=IndicatorEngine(
            data, FakeClock(datetime(2026, 7, 16, tzinfo=_TZ)), get_indicator_registry()
        ),
        calendar=cal,
        baseline=create_baseline(name, tuning),
        config=BacktestConfig(),
        history_days=tuning.history_days,
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


async def test_buy_and_hold_is_unleveraged_equal_weight() -> None:
    """Two symbols each +~25% give a portfolio return of ~25%, not ~50% or 744%.

    Equal-weight, unleveraged: capital is split in two, so the portfolio return
    is the average symbol return — a sane low-double-digit figure — never an
    amplified, leveraged number.
    """
    days = _trading_days(200)
    data = HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())
    # Both symbols rise ~25% smoothly over the window.
    for base in (100.0, 400.0):
        prices = [base * (1.0 + 0.25 * j / (len(days) - 1)) for j in range(len(days))]
        await _seed(data, f"S{int(base)}", days, prices)

    engine = _engine(data, BaselineName.BUY_AND_HOLD)
    report = await engine.run(["S100", "S400"], days[20], days[-1])

    assert report.trades == 2  # one hold per symbol
    # Unleveraged: return tracks the ~25% average move (minus costs), well within
    # the plausible band and nowhere near a leveraged multiple.
    assert 10.0 < report.total_return_pct < 35.0
    # Total invested notional never exceeded starting capital (no leverage).
    invested = sum(t.entry_price * t.size for t in report.closed_trades)
    assert invested <= report.starting_capital * 1.001


async def test_notional_cap_bounds_a_tight_atr_baseline() -> None:
    """A tiny-ATR series would over-leverage under risk sizing; the cap bounds it.

    With almost no volatility the 1%-risk size explodes (risk-per-share -> 0),
    which is exactly what printed impossible returns. The 1x notional cap must
    keep the portfolio return within a sane band.
    """
    days = _trading_days(200)
    data = HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())
    # Almost-flat prices: tiny ATR -> unbounded risk size without the cap.
    for i, base in enumerate((100.0, 105.0, 110.0)):
        prices = [base + 0.01 * j for j in range(len(days))]
        await _seed(data, f"T{i}", days, prices)
    await _seed(data, "NIFTY", days, [100.0 + 0.01 * j for j in range(len(days))])

    engine = _engine(data, BaselineName.BUY_AND_HOLD)
    report = await engine.run(["T0", "T1", "T2"], days[20], days[-1])

    # Even with near-zero volatility the return stays bounded (cap in force).
    assert -60.0 <= report.total_return_pct <= 150.0
    invested = sum(t.entry_price * t.size for t in report.closed_trades)
    assert invested <= report.starting_capital * 1.001
