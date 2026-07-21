"""Exceptions for the historical import engine."""

from __future__ import annotations


class OutOfWindowDataError(Exception):
    """A provider returned candles outside the requested window.

    Raised when a fetched batch contains any candle whose timestamp falls
    outside ``[requested_start, requested_end]``. The batch is rejected wholesale
    — never stored, never relabelled — because a provider that serves the wrong
    date range cannot be trusted to have served the right prices either.
    """
