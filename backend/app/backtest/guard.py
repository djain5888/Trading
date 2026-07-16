"""A no-look-ahead data view for the backtester.

The single most dangerous bug in a backtest is look-ahead: letting the strategy
see prices it could not have known at decision time. :class:`LookaheadGuard`
makes that a *structural* error. It wraps the real :class:`HistoricalDataEngine`
and, once frozen at a simulation date, raises :class:`LookaheadError` if any read
returns a candle dated after that date. The backtest freezes the guard to each
day before running the exact same strategy/regime engines the live path uses, so
a future-data leak fails the run loudly instead of silently inflating returns.
"""

from __future__ import annotations

from datetime import date, datetime

from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.models import Candle, SeriesKey


class LookaheadError(RuntimeError):
    """Raised when a read would expose candles dated after the simulation date."""


class LookaheadGuard(HistoricalDataEngine):
    """A read-through data engine that refuses to return future candles."""

    def __init__(self, inner: HistoricalDataEngine) -> None:
        """Wrap ``inner`` and share its repository and validator.

        Args:
            inner: The real data engine whose store is read through the guard.
        """
        super().__init__(inner._repository, inner._validator)
        self._as_of: date | None = None

    def freeze_at(self, day: date) -> None:
        """Set the simulation date; later reads may not return candles past it."""
        self._as_of = day

    async def get_candles(
        self, key: SeriesKey, start: datetime, end: datetime
    ) -> list[Candle]:
        """Return candles in ``[start, end]``, asserting none are in the future."""
        candles = await super().get_candles(key, start, end)
        self._verify(key, candles)
        return candles

    async def get_latest(self, key: SeriesKey) -> Candle | None:
        """Return the latest candle, asserting it is not in the future."""
        candle = await super().get_latest(key)
        if candle is not None:
            self._verify(key, (candle,))
        return candle

    async def get_candle(self, key: SeriesKey, timestamp: datetime) -> Candle | None:
        """Return the candle at ``timestamp``, asserting it is not in the future."""
        candle = await super().get_candle(key, timestamp)
        if candle is not None:
            self._verify(key, (candle,))
        return candle

    def _verify(
        self, key: SeriesKey, candles: tuple[Candle, ...] | list[Candle]
    ) -> None:
        """Raise if any candle is dated after the frozen simulation date."""
        if self._as_of is None:
            return
        for candle in candles:
            if candle.timestamp.date() > self._as_of:
                raise LookaheadError(
                    f"Look-ahead detected: {key.symbol} candle dated "
                    f"{candle.timestamp.date()} is after the simulation date "
                    f"{self._as_of}."
                )
