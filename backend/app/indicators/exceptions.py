"""Indicator engine exceptions."""

from __future__ import annotations


class IndicatorError(Exception):
    """Base class for all indicator-engine errors."""


class UnknownIndicatorError(IndicatorError):
    """Raised when an indicator name is not registered."""


class InvalidPeriodError(IndicatorError):
    """Raised when a period or parameter is out of range."""


class InsufficientHistoryError(IndicatorError):
    """Raised when there are too few candles to compute an indicator."""


class ComputationError(IndicatorError):
    """Raised when a computation produces a non-finite (NaN/inf) value."""
