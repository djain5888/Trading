"""Market-calendar package: the single source of truth for market timing."""

from app.market.calendar.config import (
    CalendarConfigProvider,
    InMemoryCalendarConfigProvider,
    UnknownExchangeError,
    nse_calendar_config,
)
from app.market.calendar.dependencies import (
    get_calendar_config_provider,
    get_clock,
    get_holiday_provider,
    get_market_calendar_service,
)
from app.market.calendar.enums import MarketState, SessionType
from app.market.calendar.holidays import HolidayProvider, StaticHolidayProvider
from app.market.calendar.models import (
    ExchangeCalendarConfig,
    MaintenanceWindow,
    SessionWindow,
)
from app.market.calendar.service import MarketCalendarService

__all__ = [
    "CalendarConfigProvider",
    "ExchangeCalendarConfig",
    "HolidayProvider",
    "InMemoryCalendarConfigProvider",
    "MaintenanceWindow",
    "MarketCalendarService",
    "MarketState",
    "SessionType",
    "SessionWindow",
    "StaticHolidayProvider",
    "UnknownExchangeError",
    "get_calendar_config_provider",
    "get_clock",
    "get_holiday_provider",
    "get_market_calendar_service",
    "nse_calendar_config",
]
