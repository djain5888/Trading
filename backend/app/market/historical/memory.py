"""In-memory candle repository.

A storage-independent reference adapter used as the default binding and in
tests. It is not a database implementation; it keeps candles in process memory,
indexed by series and timestamp.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from app.market.historical.models import Candle, SeriesKey
from app.market.historical.repository import CandleRepository


class InMemoryCandleRepository(CandleRepository):
    """A :class:`CandleRepository` backed by in-process dictionaries."""

    def __init__(self) -> None:
        """Initialise an empty repository."""
        self._store: dict[SeriesKey, dict[datetime, Candle]] = {}

    def _series(self, key: SeriesKey) -> dict[datetime, Candle]:
        """Return (creating if needed) the timestamp map for a series."""
        return self._store.setdefault(key, {})

    async def save(self, candle: Candle) -> None:
        """Persist a single candle, overwriting any existing one at its key."""
        self._series(candle.series_key)[candle.timestamp] = candle

    async def save_many(self, candles: Sequence[Candle]) -> None:
        """Persist several candles."""
        for candle in candles:
            await self.save(candle)

    async def get(self, key: SeriesKey, timestamp: datetime) -> Candle | None:
        """Return the candle at ``timestamp`` for ``key``, or ``None``."""
        return self._store.get(key, {}).get(timestamp)

    async def get_range(
        self, key: SeriesKey, start: datetime, end: datetime
    ) -> list[Candle]:
        """Return the ordered candles for ``key`` within ``[start, end]``."""
        series = self._store.get(key, {})
        selected = [
            candle for timestamp, candle in series.items() if start <= timestamp <= end
        ]
        return sorted(selected, key=lambda candle: candle.timestamp)

    async def latest(self, key: SeriesKey) -> Candle | None:
        """Return the most recent candle for ``key``, or ``None``."""
        series = self._store.get(key, {})
        if not series:
            return None
        return max(series.values(), key=lambda candle: candle.timestamp)

    async def exists(self, key: SeriesKey, timestamp: datetime) -> bool:
        """Return whether a candle exists at ``timestamp`` for ``key``."""
        return timestamp in self._store.get(key, {})

    async def delete(self, key: SeriesKey, timestamp: datetime) -> bool:
        """Delete the candle at ``timestamp``; return whether one was removed."""
        series = self._store.get(key)
        if series is None or timestamp not in series:
            return False
        del series[timestamp]
        return True

    async def count(self, key: SeriesKey) -> int:
        """Return the number of stored candles for ``key``."""
        return len(self._store.get(key, {}))
