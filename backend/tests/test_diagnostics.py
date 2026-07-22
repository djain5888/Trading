"""Tests for return-integrity diagnostics (TASK-028).

Return-calc correctness on hand-verified prices, split/bonus detection on
unadjusted data, and the candle date audit (duplicates, out-of-order,
wrong-year leakage, missing sessions).
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from app.backtest.diagnostics import (
    audit_candles,
    hold_return_pct,
    suspected_adjustments,
)
from app.core.timezone import INDIA_TZ
from app.market.enums import Exchange, Interval
from app.market.historical.models import Candle

_TZ = INDIA_TZ


def _bar(day: date, close: float, *, symbol: str = "RELIANCE") -> Candle:
    """Build a coherent daily candle closing at ``close``."""
    return Candle(
        symbol=symbol,
        exchange=Exchange.NSE,
        interval=Interval.ONE_DAY,
        timestamp=datetime(day.year, day.month, day.day, 9, 15, tzinfo=_TZ),
        open=Decimal(f"{close:.4f}"),
        high=Decimal(f"{close * 1.01:.4f}"),
        low=Decimal(f"{close * 0.99:.4f}"),
        close=Decimal(f"{close:.4f}"),
        volume=1000,
    )


# -- Hold-return math (hand-verified) --------------------------------------


def test_hold_return_matches_hand_calc() -> None:
    """(last/first - 1) on hand-picked closes: 100 -> 150 is +50%."""
    candles = [
        _bar(date(2024, 1, 1), 100.0),
        _bar(date(2024, 6, 1), 120.0),
        _bar(date(2024, 12, 2), 150.0),
    ]

    assert hold_return_pct(candles) == 50.0  # 150/100 - 1


def test_hold_return_uses_chronological_ends_not_input_order() -> None:
    """Return is first/last by DATE even if candles arrive out of order."""
    out_of_order = [
        _bar(date(2024, 12, 2), 150.0),
        _bar(date(2024, 1, 1), 100.0),
        _bar(date(2024, 6, 1), 120.0),
    ]

    assert hold_return_pct(out_of_order) == 50.0


def test_hold_return_none_when_too_few_candles() -> None:
    """A single candle has no measurable hold return."""
    assert hold_return_pct([_bar(date(2024, 1, 1), 100.0)]) is None
    assert hold_return_pct([]) is None


# -- Split / bonus (unadjusted price) detection ----------------------------


def test_suspected_split_is_flagged() -> None:
    """A 5:1 split on unadjusted data reads as an ~80% overnight crash."""
    candles = [
        _bar(date(2024, 1, 1), 2500.0),
        _bar(date(2024, 1, 2), 2510.0),
        _bar(date(2024, 1, 3), 500.0),  # 5:1 split: -80.1% overnight
        _bar(date(2024, 1, 4), 505.0),
    ]

    jumps = suspected_adjustments(candles, threshold_pct=25.0)

    assert len(jumps) == 1
    assert jumps[0].on == date(2024, 1, 3)
    assert jumps[0].change_pct < -75.0  # ~ -80%


def test_no_false_split_flag_on_normal_moves() -> None:
    """Ordinary large-cap sessions (a few %) are not flagged as adjustments."""
    candles = [
        _bar(date(2024, 1, 1), 2500.0),
        _bar(date(2024, 1, 2), 2560.0),  # +2.4%
        _bar(date(2024, 1, 3), 2500.0),  # -2.3%
    ]

    assert suspected_adjustments(candles, threshold_pct=25.0) == []


# -- Candle date audit -----------------------------------------------------


def test_audit_clean_window_reports_sane_return() -> None:
    """A clean Mon-Fri week has no defects and a hand-checkable hold return."""
    days = [date(2024, 1, d) for d in (1, 2, 3, 4, 5)]  # Mon..Fri
    candles = [_bar(day, 100.0 + i) for i, day in enumerate(days)]

    audit = audit_candles("RELIANCE", candles, days[0], days[-1])

    assert audit.clean is True
    assert audit.candles == 5
    assert audit.missing_sessions == 0  # contiguous weekdays
    assert audit.first_close == 100.0 and audit.last_close == 104.0
    assert audit.hold_return_pct == 4.0  # 104/100 - 1
    assert audit.suspected_adjustments == ()


def test_audit_detects_duplicates_and_out_of_order() -> None:
    """Duplicate dates and a descending pair are both reported."""
    candles = [
        _bar(date(2024, 1, 3), 102.0),  # out of order (after this comes an earlier day)
        _bar(date(2024, 1, 1), 100.0),
        _bar(date(2024, 1, 1), 100.5),  # duplicate date
    ]

    audit = audit_candles("RELIANCE", candles, date(2024, 1, 1), date(2024, 1, 31))

    assert audit.duplicate_dates == (date(2024, 1, 1),)
    assert audit.out_of_order == 1  # 2024-01-03 followed by 2024-01-01
    assert audit.clean is False


def test_audit_detects_wrong_year_leakage() -> None:
    """Candles dated outside the window (wrong-year rows) are flagged."""
    candles = [
        _bar(date(2019, 5, 2), 100.0),  # wrong year — outside window
        _bar(date(2024, 1, 2), 110.0),
        _bar(date(2024, 1, 3), 120.0),
    ]

    audit = audit_candles("RELIANCE", candles, date(2024, 1, 1), date(2024, 1, 31))

    assert date(2019, 5, 2) in audit.out_of_window_dates
    assert audit.clean is False
