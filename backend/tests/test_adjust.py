"""Tests for split/bonus back-adjustment (TASK-029 FIX 2)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from app.backtest.adjust import adjust_for_splits
from app.core.timezone import INDIA_TZ
from app.market.enums import Exchange, Interval
from app.market.historical.models import Candle

_TZ = INDIA_TZ


def _bar(day: date, close: float) -> Candle:
    """Build a coherent daily candle closing at ``close``."""
    return Candle(
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        interval=Interval.ONE_DAY,
        timestamp=datetime(day.year, day.month, day.day, 9, 15, tzinfo=_TZ),
        open=Decimal(f"{close:.4f}"),
        high=Decimal(f"{close * 1.01:.4f}"),
        low=Decimal(f"{close * 0.99:.4f}"),
        close=Decimal(f"{close:.4f}"),
        volume=1000,
    )


def test_five_to_one_split_is_back_adjusted() -> None:
    """A 5:1 split (2500 -> 500) back-adjusts pre-split prices by 1/5.

    Hand check: the two pre-split closes 2500 and 2510 become 500.0 and 502.0,
    so the whole series is continuous around 500 with no ~80% crash.
    """
    candles = [
        _bar(date(2024, 1, 1), 2500.0),
        _bar(date(2024, 1, 2), 2510.0),
        _bar(date(2024, 1, 3), 500.0),  # 5:1 split
        _bar(date(2024, 1, 4), 505.0),
    ]

    result = adjust_for_splits("RELIANCE", candles)

    assert result.usable is True
    assert result.events_applied == 1
    assert result.events[0].factor == 5.0
    closes = [float(c.close) for c in result.adjusted]
    assert closes == [500.0, 502.0, 500.0, 505.0]  # continuous, no 80% gap
    # Pre-split volume scales up by the split factor (more shares).
    assert result.adjusted[0].volume == 5000


def test_one_to_one_bonus_is_back_adjusted() -> None:
    """A 1:1 bonus halves the price (factor 2.0); pre-event prices scale by 1/2."""
    candles = [
        _bar(date(2024, 6, 3), 1200.0),
        _bar(date(2024, 6, 4), 600.0),  # 1:1 bonus
        _bar(date(2024, 6, 5), 610.0),
    ]

    result = adjust_for_splits("RELIANCE", candles)

    assert result.usable is True
    assert result.events[0].factor == 2.0
    assert float(result.adjusted[0].close) == 600.0


def test_unexplained_crash_marks_symbol_unusable() -> None:
    """A large gap that fits no clean split/bonus factor fails loudly."""
    candles = [
        _bar(date(2024, 1, 1), 1000.0),
        _bar(date(2024, 1, 2), 533.0),  # -46.7% (factor ~1.88): no clean ratio
        _bar(date(2024, 1, 3), 540.0),
    ]

    result = adjust_for_splits("RELIANCE", candles)

    assert result.usable is False
    assert result.events_applied == 0
    assert result.adjusted == ()
    assert "not explainable" in result.reason


def test_clean_series_is_unchanged() -> None:
    """A series with no large gaps is usable with zero events and no rewrite."""
    candles = [_bar(date(2024, 1, d), 100.0 + d) for d in range(1, 6)]

    result = adjust_for_splits("RELIANCE", candles)

    assert result.usable is True
    assert result.events_applied == 0
    assert [float(c.close) for c in result.adjusted] == [
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
    ]
