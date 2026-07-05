"""Exchange calendar configuration and its provider.

Trading hours are declared here as configuration data. Adding a future
exchange (e.g. BSE) means registering another :class:`ExchangeCalendarConfig`;
no calendar logic changes. The :class:`CalendarConfigProvider` protocol is the
seam a future config source (database, remote service) can implement instead.
"""

from __future__ import annotations

from calendar import SATURDAY, SUNDAY
from datetime import time
from typing import Protocol, runtime_checkable

from app.core.timezone import MARKET_TIMEZONE_NAME
from app.market.calendar.enums import SessionType
from app.market.calendar.models import ExchangeCalendarConfig, SessionWindow
from app.market.enums import Exchange

#: Weekdays on which Indian equity exchanges do not trade.
_WEEKEND_DAYS: frozenset[int] = frozenset({SATURDAY, SUNDAY})

# NSE regular equity timings (IST). Declared as named data, not inline magic.
_NSE_PRE_MARKET = SessionWindow(
    session_type=SessionType.PRE_MARKET, start=time(9, 0), end=time(9, 15)
)
_NSE_REGULAR = SessionWindow(
    session_type=SessionType.REGULAR, start=time(9, 15), end=time(15, 30)
)
_NSE_POST_MARKET = SessionWindow(
    session_type=SessionType.POST_MARKET, start=time(15, 40), end=time(16, 0)
)


def nse_calendar_config() -> ExchangeCalendarConfig:
    """Return the default NSE calendar configuration."""
    return ExchangeCalendarConfig(
        exchange=Exchange.NSE,
        timezone=MARKET_TIMEZONE_NAME,
        weekend_days=_WEEKEND_DAYS,
        sessions=(_NSE_PRE_MARKET, _NSE_REGULAR, _NSE_POST_MARKET),
    )


class UnknownExchangeError(LookupError):
    """Raised when no calendar configuration exists for an exchange."""


@runtime_checkable
class CalendarConfigProvider(Protocol):
    """Supplies calendar configuration for an exchange."""

    def get(self, exchange: Exchange) -> ExchangeCalendarConfig:
        """Return the configuration for ``exchange``.

        Raises:
            UnknownExchangeError: If the exchange is not configured.
        """
        ...


class InMemoryCalendarConfigProvider:
    """A config provider backed by an in-memory registry."""

    def __init__(self, configs: dict[Exchange, ExchangeCalendarConfig]) -> None:
        """Initialise the provider.

        Args:
            configs: Mapping of exchange to its calendar configuration.
        """
        self._configs = dict(configs)

    def get(self, exchange: Exchange) -> ExchangeCalendarConfig:
        """Return the configuration for ``exchange``.

        Args:
            exchange: The exchange to look up.

        Returns:
            The exchange's calendar configuration.

        Raises:
            UnknownExchangeError: If the exchange is not configured.
        """
        try:
            return self._configs[exchange]
        except KeyError as exc:
            raise UnknownExchangeError(
                f"No calendar configuration registered for {exchange}."
            ) from exc

    @classmethod
    def with_defaults(cls) -> InMemoryCalendarConfigProvider:
        """Return a provider preloaded with the shipped default exchanges."""
        return cls({Exchange.NSE: nse_calendar_config()})
