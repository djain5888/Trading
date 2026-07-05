"""Interval-to-duration mapping.

Gap and overlap detection operate on the fixed grid implied by an interval.
Calendar-variable intervals (months) have no fixed duration and therefore
disable grid-based detection.
"""

from __future__ import annotations

from datetime import timedelta

from app.market.enums import Interval

_INTERVAL_DELTAS: dict[Interval, timedelta] = {
    Interval.ONE_MINUTE: timedelta(minutes=1),
    Interval.FIVE_MINUTE: timedelta(minutes=5),
    Interval.FIFTEEN_MINUTE: timedelta(minutes=15),
    Interval.THIRTY_MINUTE: timedelta(minutes=30),
    Interval.ONE_HOUR: timedelta(hours=1),
    Interval.ONE_DAY: timedelta(days=1),
    Interval.ONE_WEEK: timedelta(weeks=1),
}


def interval_delta(interval: Interval) -> timedelta | None:
    """Return the fixed duration of an interval, or ``None`` if variable.

    Args:
        interval: The candle interval.

    Returns:
        The fixed :class:`~datetime.timedelta`, or ``None`` for calendar-
        variable intervals such as one month.
    """
    return _INTERVAL_DELTAS.get(interval)
