"""Exhaustive unit tests for the MarketCalendarService."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from app.core.clock import FakeClock
from app.core.timezone import INDIA_TZ
from app.market.calendar.config import (
    InMemoryCalendarConfigProvider,
    UnknownExchangeError,
    nse_calendar_config,
)
from app.market.calendar.enums import MarketState, SessionType
from app.market.calendar.holidays import StaticHolidayProvider
from app.market.calendar.models import MaintenanceWindow, SessionWindow
from app.market.calendar.service import MarketCalendarService
from app.market.enums import Exchange

# Reference dates (NSE). Mon 2025-01-06 is a normal trading day; 2025-01-11 is a
# Saturday; 2025-08-15 (Independence Day, a Friday) is used as a holiday.
_MONDAY = date(2025, 1, 6)
_FRIDAY = date(2025, 1, 10)
_SATURDAY = date(2025, 1, 11)
_HOLIDAY = date(2025, 8, 15)


def _ist(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    """Build an IST-aware datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=INDIA_TZ)


def _service(
    moment: datetime,
    *,
    holidays: list[date] | None = None,
    maintenance: list[MaintenanceWindow] | None = None,
) -> MarketCalendarService:
    """Build a calendar service anchored at ``moment``."""
    config = nse_calendar_config()
    if maintenance is not None:
        config = config.model_copy(update={"maintenance_windows": tuple(maintenance)})
    config_provider = InMemoryCalendarConfigProvider({Exchange.NSE: config})
    holiday_provider = StaticHolidayProvider({Exchange.NSE: holidays or []})
    return MarketCalendarService(
        clock=FakeClock(moment),
        config_provider=config_provider,
        holiday_provider=holiday_provider,
    )


# -- Market states ---------------------------------------------------------


def test_pre_market() -> None:
    """Before the regular open the market is in pre-market."""
    service = _service(_ist(2025, 1, 6, 9, 5))
    assert service.get_market_state() is MarketState.PRE_MARKET
    session = service.current_session()
    assert session is not None
    assert session.session_type is SessionType.PRE_MARKET


def test_market_open() -> None:
    """During the regular session the market is open."""
    service = _service(_ist(2025, 1, 6, 10, 0))
    assert service.get_market_state() is MarketState.OPEN
    assert service.is_market_open() is True


def test_post_market() -> None:
    """During the closing session the market is in post-market."""
    service = _service(_ist(2025, 1, 6, 15, 45))
    assert service.get_market_state() is MarketState.POST_MARKET


def test_closed_before_pre_market() -> None:
    """Early morning on a trading day is closed."""
    service = _service(_ist(2025, 1, 6, 8, 0))
    assert service.get_market_state() is MarketState.CLOSED
    assert service.current_session() is None


def test_closed_between_regular_and_post() -> None:
    """The gap between the regular close and the closing session is closed."""
    service = _service(_ist(2025, 1, 6, 15, 35))
    assert service.get_market_state() is MarketState.CLOSED


def test_closed_after_all_sessions() -> None:
    """After the closing session the market is closed."""
    service = _service(_ist(2025, 1, 6, 16, 30))
    assert service.get_market_state() is MarketState.CLOSED


def test_weekend() -> None:
    """Saturdays are weekends."""
    service = _service(_ist(2025, 1, 11, 10, 0))
    assert service.get_market_state() is MarketState.WEEKEND
    assert service.is_market_open() is False
    assert service.is_trading_day() is False


def test_holiday() -> None:
    """A configured holiday reports the holiday state."""
    service = _service(_ist(2025, 8, 15, 10, 0), holidays=[_HOLIDAY])
    assert service.get_market_state() is MarketState.HOLIDAY
    assert service.is_trading_day() is False


def test_maintenance_overrides_open() -> None:
    """A maintenance window takes precedence over an otherwise open market."""
    window = MaintenanceWindow(
        start=_ist(2025, 1, 6, 9, 30), end=_ist(2025, 1, 6, 11, 0)
    )
    service = _service(_ist(2025, 1, 6, 10, 0), maintenance=[window])
    assert service.get_market_state() is MarketState.MAINTENANCE


# -- Boundary times --------------------------------------------------------


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (9, 0, MarketState.PRE_MARKET),  # pre-market start (inclusive)
        (9, 15, MarketState.OPEN),  # regular open (inclusive)
        (15, 30, MarketState.CLOSED),  # regular close (exclusive)
        (15, 40, MarketState.POST_MARKET),  # closing session start
        (16, 0, MarketState.CLOSED),  # closing session end (exclusive)
    ],
)
def test_session_boundaries(hour: int, minute: int, expected: MarketState) -> None:
    """Session edges are half-open: start inclusive, end exclusive."""
    service = _service(_ist(2025, 1, 6, hour, minute))
    assert service.get_market_state() is expected


# -- Trading day -----------------------------------------------------------


def test_is_trading_day_explicit_dates() -> None:
    """is_trading_day evaluates the supplied date."""
    service = _service(_ist(2025, 1, 6, 10, 0), holidays=[_HOLIDAY])
    assert service.is_trading_day(_MONDAY) is True
    assert service.is_trading_day(_SATURDAY) is False
    assert service.is_trading_day(_HOLIDAY) is False


def test_current_trading_date_on_trading_day() -> None:
    """On a trading day the current trading date is today."""
    service = _service(_ist(2025, 1, 6, 10, 0))
    assert service.current_trading_date() == _MONDAY


def test_current_trading_date_on_weekend_rolls_forward() -> None:
    """On a weekend the current trading date is the next trading day."""
    service = _service(_ist(2025, 1, 11, 10, 0))
    assert service.current_trading_date() == date(2025, 1, 13)


# -- Next open / close -----------------------------------------------------


def test_next_market_open_same_day() -> None:
    """Before the open, the next open is today's regular start."""
    service = _service(_ist(2025, 1, 6, 8, 0))
    assert service.next_market_open() == _ist(2025, 1, 6, 9, 15)


def test_next_market_open_when_open_is_next_day() -> None:
    """While open, the next open is the following trading day."""
    service = _service(_ist(2025, 1, 6, 10, 0))
    assert service.next_market_open() == _ist(2025, 1, 7, 9, 15)


def test_next_market_open_skips_weekend() -> None:
    """After Friday's close the next open skips the weekend to Monday."""
    service = _service(_ist(2025, 1, 10, 16, 30))
    assert service.next_market_open() == _ist(2025, 1, 13, 9, 15)


def test_next_market_close_same_day() -> None:
    """Before the close, the next close is today's regular end."""
    service = _service(_ist(2025, 1, 6, 10, 0))
    assert service.next_market_close() == _ist(2025, 1, 6, 15, 30)


def test_next_market_close_after_close_is_next_day() -> None:
    """After the close, the next close is the following trading day."""
    service = _service(_ist(2025, 1, 6, 16, 0))
    assert service.next_market_close() == _ist(2025, 1, 7, 15, 30)


def test_next_market_close_skips_weekend() -> None:
    """On a weekend the next close is Monday's regular end."""
    service = _service(_ist(2025, 1, 11, 10, 0))
    assert service.next_market_close() == _ist(2025, 1, 13, 15, 30)


# -- Timezone --------------------------------------------------------------


def test_utc_input_is_interpreted_in_ist() -> None:
    """A UTC instant is converted to IST before state resolution."""
    # 04:30 UTC on Monday == 10:00 IST -> market open.
    service = _service(datetime(2025, 1, 6, 4, 30, tzinfo=UTC))
    assert service.get_market_state() is MarketState.OPEN


# -- Errors & config -------------------------------------------------------


def test_unknown_exchange_raises() -> None:
    """Querying an unconfigured exchange raises a clear error."""
    service = _service(_ist(2025, 1, 6, 10, 0))
    with pytest.raises(UnknownExchangeError):
        service.get_market_state(Exchange.BSE)


def test_session_window_rejects_inverted_range() -> None:
    """A session with end <= start is rejected."""
    with pytest.raises(ValueError):
        SessionWindow(
            session_type=SessionType.REGULAR,
            start=datetime(2025, 1, 1, 15, 30).time(),
            end=datetime(2025, 1, 1, 9, 15).time(),
        )


# -- Holiday loaders -------------------------------------------------------


def test_holiday_provider_from_json(tmp_path: Path) -> None:
    """Holidays load correctly from a JSON file."""
    path = tmp_path / "holidays.json"
    path.write_text('{"NSE": ["2025-08-15", "2025-10-02"]}', encoding="utf-8")

    provider = StaticHolidayProvider.from_json(path)
    holidays = provider.holidays(Exchange.NSE)
    assert date(2025, 8, 15) in holidays
    assert date(2025, 10, 2) in holidays
    assert provider.holidays(Exchange.BSE) == frozenset()


def test_holiday_provider_from_csv(tmp_path: Path) -> None:
    """Holidays load correctly from a CSV file."""
    path = tmp_path / "holidays.csv"
    path.write_text("date,exchange\n2025-08-15,NSE\n2025-10-02,NSE\n", encoding="utf-8")

    provider = StaticHolidayProvider.from_csv(path)
    assert date(2025, 8, 15) in provider.holidays(Exchange.NSE)


def test_holiday_provider_from_csv_missing_columns(tmp_path: Path) -> None:
    """A CSV without required columns is rejected."""
    path = tmp_path / "bad.csv"
    path.write_text("foo,bar\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        StaticHolidayProvider.from_csv(path)


# -- Dependency injection --------------------------------------------------


def test_di_returns_singleton() -> None:
    """The DI accessor returns a cached singleton service."""
    from app.market.calendar.dependencies import (
        _service_singleton,
        get_market_calendar_service,
    )

    _service_singleton.cache_clear()
    try:
        assert get_market_calendar_service() is get_market_calendar_service()
    finally:
        _service_singleton.cache_clear()
