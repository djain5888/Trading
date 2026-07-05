"""Clock abstraction.

Production code must never call :func:`datetime.now` or ``datetime.utcnow``
directly. Instead it depends on a :class:`Clock`, which is the single sanctioned
place time enters the system. :class:`SystemClock` wraps the real wall clock;
:class:`FakeClock` is a deterministic clock for tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta

from app.core.timezone import INDIA_TZ


class Clock(ABC):
    """A source of the current time, always timezone-aware in IST."""

    @abstractmethod
    def now(self) -> datetime:
        """Return the current instant as an IST-aware :class:`datetime`."""

    def today(self) -> date:
        """Return the current calendar date in IST."""
        return self.now().date()


class SystemClock(Clock):
    """A clock backed by the operating-system wall clock.

    This is the *only* production component permitted to read real time.
    """

    def now(self) -> datetime:
        """Return the current instant in IST."""
        return datetime.now(tz=INDIA_TZ)


class FakeClock(Clock):
    """A deterministic, manually controlled clock for tests.

    The provided instant is normalised to IST. If a naive datetime is given it
    is assumed to already be wall-clock IST.
    """

    def __init__(self, moment: datetime) -> None:
        """Initialise the clock at a fixed instant.

        Args:
            moment: The instant the clock should report.
        """
        self._moment = self._to_ist(moment)

    @staticmethod
    def _to_ist(moment: datetime) -> datetime:
        """Return ``moment`` as an IST-aware datetime."""
        if moment.tzinfo is None:
            return moment.replace(tzinfo=INDIA_TZ)
        return moment.astimezone(INDIA_TZ)

    def now(self) -> datetime:
        """Return the currently set instant in IST."""
        return self._moment

    def set(self, moment: datetime) -> None:
        """Set the clock to a new instant.

        Args:
            moment: The new instant.
        """
        self._moment = self._to_ist(moment)

    def advance(self, delta: timedelta) -> None:
        """Advance the clock by a duration.

        Args:
            delta: The amount of time to move forward (may be negative).
        """
        self._moment += delta
