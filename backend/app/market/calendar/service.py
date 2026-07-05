"""The market-calendar service: the single source of truth for market timing.

Every module needing to know whether the market is open, what the trading date
is, or when the next open/close occurs must go through this service rather than
reading the clock itself. All computations are pure functions of the injected
:class:`~app.core.clock.Clock`, calendar configuration and holiday data, which
makes them deterministic and exhaustively testable.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.core.clock import Clock
from app.core.logging import get_logger
from app.market.calendar.config import CalendarConfigProvider
from app.market.calendar.enums import MarketState
from app.market.calendar.holidays import HolidayProvider
from app.market.calendar.models import ExchangeCalendarConfig, SessionWindow
from app.market.enums import Exchange

logger = get_logger(__name__)

#: Upper bound on the forward search for a trading day, guarding against an
#: exchange configured (incorrectly) with no reachable trading days.
_MAX_TRADING_DAY_LOOKAHEAD = 3660

#: Default exchange used when a caller does not specify one.
_DEFAULT_EXCHANGE = Exchange.NSE


class MarketCalendarService:
    """Authoritative provider of market-timing information.

    The service is exchange-parametric: it works for any exchange whose
    configuration and holidays are registered, so supporting BSE is purely a
    matter of configuration.
    """

    def __init__(
        self,
        clock: Clock,
        config_provider: CalendarConfigProvider,
        holiday_provider: HolidayProvider,
    ) -> None:
        """Initialise the service.

        Args:
            clock: The injected time source.
            config_provider: Supplies per-exchange calendar configuration.
            holiday_provider: Supplies per-exchange holiday dates.
        """
        self._clock = clock
        self._config_provider = config_provider
        self._holiday_provider = holiday_provider

    # -- Public API --------------------------------------------------------

    def get_market_state(self, exchange: Exchange = _DEFAULT_EXCHANGE) -> MarketState:
        """Return the current market state for ``exchange``."""
        config = self._config_provider.get(exchange)
        now = self._now(config)
        state, _ = self._resolve(now, config, exchange)
        return state

    def is_market_open(self, exchange: Exchange = _DEFAULT_EXCHANGE) -> bool:
        """Return whether ``exchange`` is currently in its regular session."""
        return self.get_market_state(exchange) is MarketState.OPEN

    def is_trading_day(
        self, on: date | None = None, exchange: Exchange = _DEFAULT_EXCHANGE
    ) -> bool:
        """Return whether a date is a trading day for ``exchange``.

        Args:
            on: The date to test; defaults to the current trading-timezone date.
            exchange: The exchange to test.

        Returns:
            ``True`` if the date is neither a weekend nor a holiday.
        """
        config = self._config_provider.get(exchange)
        target = on if on is not None else self._now(config).date()
        return self._is_trading_day(target, config, exchange)

    def current_session(
        self, exchange: Exchange = _DEFAULT_EXCHANGE
    ) -> SessionWindow | None:
        """Return the active session window, or ``None`` if not in session."""
        config = self._config_provider.get(exchange)
        now = self._now(config)
        _, session = self._resolve(now, config, exchange)
        return session

    def next_market_open(self, exchange: Exchange = _DEFAULT_EXCHANGE) -> datetime:
        """Return the next instant the regular session opens.

        If the market is currently open, the *following* trading day's open is
        returned.
        """
        config = self._config_provider.get(exchange)
        now = self._now(config)
        return self._next_boundary(now, config, exchange, config.regular_session.start)

    def next_market_close(self, exchange: Exchange = _DEFAULT_EXCHANGE) -> datetime:
        """Return the next instant the regular session closes."""
        config = self._config_provider.get(exchange)
        now = self._now(config)
        return self._next_boundary(now, config, exchange, config.regular_session.end)

    def current_trading_date(self, exchange: Exchange = _DEFAULT_EXCHANGE) -> date:
        """Return the effective trading date.

        Returns today's date when today is a trading day; otherwise the next
        upcoming trading day.
        """
        config = self._config_provider.get(exchange)
        today = self._now(config).date()
        if self._is_trading_day(today, config, exchange):
            return today
        return self._next_trading_day(today, config, exchange, inclusive=False)

    # -- Internals ---------------------------------------------------------

    def _now(self, config: ExchangeCalendarConfig) -> datetime:
        """Return the current instant in the exchange's timezone."""
        return self._clock.now().astimezone(ZoneInfo(config.timezone))

    def _resolve(
        self, now: datetime, config: ExchangeCalendarConfig, exchange: Exchange
    ) -> tuple[MarketState, SessionWindow | None]:
        """Resolve the market state and active session for an instant.

        Args:
            now: The instant, in the exchange timezone.
            config: The exchange configuration.
            exchange: The exchange being resolved.

        Returns:
            A tuple of the market state and the active session (or ``None``).
        """
        if any(window.contains(now) for window in config.maintenance_windows):
            return MarketState.MAINTENANCE, None
        target = now.date()
        if now.weekday() in config.weekend_days:
            return MarketState.WEEKEND, None
        if target in self._holiday_provider.holidays(exchange):
            return MarketState.HOLIDAY, None
        local = now.time()
        for session in config.sessions:
            if session.contains(local):
                return session.session_type.to_market_state(), session
        return MarketState.CLOSED, None

    def _is_trading_day(
        self, target: date, config: ExchangeCalendarConfig, exchange: Exchange
    ) -> bool:
        """Return whether ``target`` is a trading day."""
        if target.weekday() in config.weekend_days:
            return False
        return target not in self._holiday_provider.holidays(exchange)

    def _next_trading_day(
        self,
        origin: date,
        config: ExchangeCalendarConfig,
        exchange: Exchange,
        *,
        inclusive: bool,
    ) -> date:
        """Return the first trading day at or after ``origin``.

        Args:
            origin: The date to search from.
            config: The exchange configuration.
            exchange: The exchange being searched.
            inclusive: Whether ``origin`` itself may be returned.

        Returns:
            The next trading date.

        Raises:
            RuntimeError: If no trading day is found within the lookahead bound.
        """
        candidate = origin if inclusive else origin + timedelta(days=1)
        for _ in range(_MAX_TRADING_DAY_LOOKAHEAD):
            if self._is_trading_day(candidate, config, exchange):
                return candidate
            candidate += timedelta(days=1)
        raise RuntimeError(
            f"No trading day found within {_MAX_TRADING_DAY_LOOKAHEAD} days "
            f"for {exchange}."
        )

    def _next_boundary(
        self,
        now: datetime,
        config: ExchangeCalendarConfig,
        exchange: Exchange,
        boundary: time,
    ) -> datetime:
        """Return the next occurrence of a daily session boundary.

        Args:
            now: The current instant, in the exchange timezone.
            config: The exchange configuration.
            exchange: The exchange being queried.
            boundary: The local time of the boundary (session start or end).

        Returns:
            The next timezone-aware instant at ``boundary`` on a trading day.
        """
        tz = ZoneInfo(config.timezone)
        today = now.date()
        if self._is_trading_day(today, config, exchange):
            candidate = datetime.combine(today, boundary, tzinfo=tz)
            if candidate > now:
                return candidate
        next_day = self._next_trading_day(today, config, exchange, inclusive=False)
        return datetime.combine(next_day, boundary, tzinfo=tz)
