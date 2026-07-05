"""Unit tests for the Clock abstraction."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.core.clock import FakeClock, SystemClock
from app.core.timezone import INDIA_TZ


def test_system_clock_is_timezone_aware() -> None:
    """SystemClock returns an IST-aware datetime with a +5:30 offset."""
    now = SystemClock().now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(hours=5, minutes=30)


def test_fake_clock_naive_input_is_assumed_ist() -> None:
    """A naive instant is treated as wall-clock IST."""
    clock = FakeClock(datetime(2025, 1, 6, 10, 0))
    now = clock.now()
    assert now.tzinfo == INDIA_TZ
    assert (now.hour, now.minute) == (10, 0)


def test_fake_clock_aware_input_is_converted_to_ist() -> None:
    """An aware instant in another zone is converted to IST."""
    clock = FakeClock(datetime(2025, 1, 6, 4, 30, tzinfo=UTC))
    now = clock.now()
    assert now.utcoffset() == timedelta(hours=5, minutes=30)
    assert (now.hour, now.minute) == (10, 0)


def test_fake_clock_set_and_advance() -> None:
    """The fake clock can be repositioned and advanced."""
    clock = FakeClock(datetime(2025, 1, 6, 10, 0))
    clock.advance(timedelta(hours=2))
    assert clock.now().hour == 12
    clock.set(datetime(2025, 1, 7, 9, 0))
    assert clock.today() == datetime(2025, 1, 7).date()
