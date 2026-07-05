"""Dependency-injection wiring for the market-calendar service.

Collaborators are exposed as cached, side-effect-free singletons; there is no
global mutable state. Any of them can be overridden by constructing
:class:`MarketCalendarService` directly (as the tests do).
"""

from __future__ import annotations

from functools import lru_cache

from app.core.clock import Clock, SystemClock
from app.market.calendar.config import (
    CalendarConfigProvider,
    InMemoryCalendarConfigProvider,
)
from app.market.calendar.holidays import HolidayProvider, StaticHolidayProvider
from app.market.calendar.service import MarketCalendarService


@lru_cache(maxsize=1)
def get_clock() -> Clock:
    """Return the process-wide system clock."""
    return SystemClock()


@lru_cache(maxsize=1)
def get_calendar_config_provider() -> CalendarConfigProvider:
    """Return the default calendar configuration provider."""
    return InMemoryCalendarConfigProvider.with_defaults()


@lru_cache(maxsize=1)
def get_holiday_provider() -> HolidayProvider:
    """Return the default holiday provider.

    Defaults to an empty provider; deployments substitute a file- or
    service-backed provider via this dependency.
    """
    return StaticHolidayProvider({})


@lru_cache(maxsize=1)
def _service_singleton() -> MarketCalendarService:
    """Return the cached market-calendar service."""
    return MarketCalendarService(
        clock=get_clock(),
        config_provider=get_calendar_config_provider(),
        holiday_provider=get_holiday_provider(),
    )


def get_market_calendar_service() -> MarketCalendarService:
    """FastAPI dependency returning the shared market-calendar service."""
    return _service_singleton()
