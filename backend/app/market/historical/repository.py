"""The candle repository port.

This is a storage-independent interface. Concrete adapters (PostgreSQL, DuckDB,
in-memory) implement it without the engine knowing which backing store is used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

from app.market.historical.models import Candle, SeriesKey


class CandleRepository(ABC):
    """Persistence port for OHLCV candles."""

    @abstractmethod
    async def save(self, candle: Candle) -> None:
        """Persist a single candle, overwriting any existing one at its key."""

    @abstractmethod
    async def save_many(self, candles: Sequence[Candle]) -> None:
        """Persist several candles."""

    @abstractmethod
    async def get(self, key: SeriesKey, timestamp: datetime) -> Candle | None:
        """Return the candle at ``timestamp`` for ``key``, or ``None``."""

    @abstractmethod
    async def get_range(
        self, key: SeriesKey, start: datetime, end: datetime
    ) -> list[Candle]:
        """Return the ordered candles for ``key`` within ``[start, end]``."""

    @abstractmethod
    async def latest(self, key: SeriesKey) -> Candle | None:
        """Return the most recent candle for ``key``, or ``None``."""

    @abstractmethod
    async def exists(self, key: SeriesKey, timestamp: datetime) -> bool:
        """Return whether a candle exists at ``timestamp`` for ``key``."""

    @abstractmethod
    async def delete(self, key: SeriesKey, timestamp: datetime) -> bool:
        """Delete the candle at ``timestamp``; return whether one was removed."""

    @abstractmethod
    async def count(self, key: SeriesKey) -> int:
        """Return the number of stored candles for ``key``."""
