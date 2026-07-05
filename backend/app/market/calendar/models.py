"""Immutable value objects for the market-calendar domain."""

from __future__ import annotations

from datetime import datetime, time

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.market.calendar.enums import SessionType
from app.market.enums import Exchange


class CalendarModel(BaseModel):
    """Base class for immutable calendar value objects."""

    model_config = ConfigDict(frozen=True)


class SessionWindow(CalendarModel):
    """A single intraday trading session, expressed in local wall-clock time."""

    session_type: SessionType = Field(description="The kind of session.")
    start: time = Field(description="Local session start (inclusive).")
    end: time = Field(description="Local session end (exclusive).")

    @model_validator(mode="after")
    def _validate_order(self) -> SessionWindow:
        """Ensure the session start precedes its end.

        Raises:
            ValueError: If ``end`` is not strictly after ``start``.
        """
        if self.end <= self.start:
            raise ValueError("Session 'end' must be strictly after 'start'.")
        return self

    def contains(self, moment: time) -> bool:
        """Return whether a local time falls within this session.

        Args:
            moment: A naive local time.

        Returns:
            ``True`` if ``start <= moment < end``.
        """
        return self.start <= moment < self.end


class MaintenanceWindow(CalendarModel):
    """A scheduled maintenance period during which trading is suspended."""

    start: datetime = Field(description="Maintenance start (timezone-aware).")
    end: datetime = Field(description="Maintenance end (timezone-aware).")

    @model_validator(mode="after")
    def _validate_order(self) -> MaintenanceWindow:
        """Ensure the window start precedes its end.

        Raises:
            ValueError: If ``end`` is not strictly after ``start``.
        """
        if self.end <= self.start:
            raise ValueError("Maintenance 'end' must be strictly after 'start'.")
        return self

    def contains(self, moment: datetime) -> bool:
        """Return whether an instant falls within this window.

        Args:
            moment: A timezone-aware instant.

        Returns:
            ``True`` if ``start <= moment < end``.
        """
        return self.start <= moment < self.end


class ExchangeCalendarConfig(CalendarModel):
    """Timing configuration for a single exchange.

    Trading hours are data, not code: they live here so future exchanges are
    added by configuration rather than by changing calendar logic.
    """

    exchange: Exchange = Field(description="The exchange this config describes.")
    timezone: str = Field(description="IANA timezone name for the exchange.")
    weekend_days: frozenset[int] = Field(
        description="Non-trading weekdays (Monday=0 .. Sunday=6)."
    )
    sessions: tuple[SessionWindow, ...] = Field(
        description="Intraday sessions, ordered by start time."
    )
    maintenance_windows: tuple[MaintenanceWindow, ...] = Field(
        default_factory=tuple, description="Scheduled maintenance windows."
    )

    @model_validator(mode="after")
    def _validate_sessions(self) -> ExchangeCalendarConfig:
        """Validate that sessions are present, ordered and non-overlapping.

        Raises:
            ValueError: If sessions are empty, unordered, overlapping, or the
                mandatory regular session is missing.
        """
        if not self.sessions:
            raise ValueError("At least one session must be configured.")
        previous_end: time | None = None
        for session in self.sessions:
            if previous_end is not None and session.start < previous_end:
                raise ValueError("Sessions must be ordered and non-overlapping.")
            previous_end = session.end
        if not any(s.session_type is SessionType.REGULAR for s in self.sessions):
            raise ValueError("A REGULAR session is required.")
        return self

    @property
    def regular_session(self) -> SessionWindow:
        """Return the mandatory regular trading session."""
        return next(s for s in self.sessions if s.session_type is SessionType.REGULAR)
