"""DuckDB implementation of the candle repository.

The primary analytical store for Titan's historical candles. Timestamps are
normalised to UTC on write and re-tagged as UTC-aware on read, preserving the
instant without depending on optional timezone libraries. All blocking DuckDB
calls are offloaded to a worker thread and serialised with a lock, so the
synchronous engine stays async-friendly and the single connection is used
safely.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import duckdb

from app.core.logging import get_logger
from app.market.enums import Exchange, Interval
from app.market.historical.models import Candle, SeriesKey
from app.market.historical.repository import CandleRepository
from app.market.historical.storage.exceptions import (
    DuplicateCandleError,
    QueryError,
    SchemaInitializationError,
    StorageConnectionError,
)

logger = get_logger(__name__)

#: Number of rows fetched per chunk when streaming large result sets.
_FETCH_CHUNK_SIZE = 10_000

#: Ordered column list shared by all statements.
_COLUMNS = (
    "symbol, exchange, interval, timestamp, open, high, low, close, "
    "volume, open_interest"
)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS candles (
    symbol        VARCHAR      NOT NULL,
    exchange      VARCHAR      NOT NULL,
    interval      VARCHAR      NOT NULL,
    timestamp     TIMESTAMP    NOT NULL,
    open          DECIMAL(18, 6) NOT NULL,
    high          DECIMAL(18, 6) NOT NULL,
    low           DECIMAL(18, 6) NOT NULL,
    close         DECIMAL(18, 6) NOT NULL,
    volume        BIGINT       NOT NULL,
    open_interest BIGINT,
    PRIMARY KEY (symbol, exchange, interval, timestamp)
);
"""

# Secondary index accelerating range scans within a series (the PK already
# covers exact-key lookups; this favours ORDER BY timestamp range queries).
_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_candles_series_time
ON candles (symbol, exchange, interval, timestamp);
"""

_KEY_PREDICATE = "symbol = ? AND exchange = ? AND interval = ?"


class DuckDBCandleRepository(CandleRepository):
    """A :class:`CandleRepository` backed by a single DuckDB database file."""

    def __init__(self, database_path: str) -> None:
        """Open the database and ensure the schema exists.

        Args:
            database_path: Path to the DuckDB file, or ``":memory:"``.

        Raises:
            StorageConnectionError: If the database cannot be opened.
            SchemaInitializationError: If the schema cannot be created.
        """
        self._lock = asyncio.Lock()
        try:
            self._connection = duckdb.connect(database_path)
        except duckdb.Error as exc:
            raise StorageConnectionError(
                f"Could not open DuckDB at {database_path!r}.", cause=exc
            ) from exc
        self._initialize_schema()

    # -- Schema ------------------------------------------------------------

    def _initialize_schema(self) -> None:
        """Create the candles table and index if they do not exist."""
        try:
            self._connection.execute(_CREATE_TABLE)
            self._connection.execute(_CREATE_INDEX)
        except duckdb.Error as exc:
            raise SchemaInitializationError(
                "Failed to initialise the candle schema.", cause=exc
            ) from exc

    # -- Row mapping -------------------------------------------------------

    @staticmethod
    def _to_row(
        candle: Candle,
    ) -> tuple[
        str, str, str, datetime, Decimal, Decimal, Decimal, Decimal, int, int | None
    ]:
        """Convert a candle into a positional row with a UTC-naive timestamp."""
        return (
            candle.symbol,
            candle.exchange.value,
            candle.interval.value,
            candle.timestamp.astimezone(UTC).replace(tzinfo=None),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.open_interest,
        )

    @staticmethod
    def _from_row(row: Any) -> Candle:
        """Convert a database row back into a candle with a UTC-aware timestamp."""
        return Candle(
            symbol=row[0],
            exchange=Exchange(row[1]),
            interval=Interval(row[2]),
            timestamp=row[3].replace(tzinfo=UTC),
            open=row[4],
            high=row[5],
            low=row[6],
            close=row[7],
            volume=row[8],
            open_interest=row[9],
        )

    @staticmethod
    def _key_params(key: SeriesKey) -> list[Any]:
        """Return the positional parameters identifying a series."""
        return [key.symbol, key.exchange.value, key.interval.value]

    # -- Writes ------------------------------------------------------------

    async def save(self, candle: Candle) -> None:
        """Persist a single candle, overwriting any existing one at its key."""
        async with self._lock:
            await asyncio.to_thread(self._save_sync, candle)

    def _save_sync(self, candle: Candle) -> None:
        placeholders = ", ".join(["?"] * 10)
        try:
            self._connection.execute(
                f"INSERT OR REPLACE INTO candles ({_COLUMNS}) VALUES ({placeholders})",
                list(self._to_row(candle)),
            )
        except duckdb.Error as exc:
            raise QueryError("Failed to save candle.", cause=exc) from exc

    async def save_many(self, candles: Sequence[Candle]) -> None:
        """Persist several candles atomically as new rows.

        Raises:
            DuplicateCandleError: If any candle collides with an existing key.
            QueryError: If the bulk insert otherwise fails.
        """
        rows = [list(self._to_row(candle)) for candle in candles]
        if not rows:
            return
        async with self._lock:
            await asyncio.to_thread(self._save_many_sync, rows)

    def _save_many_sync(self, rows: list[list[Any]]) -> None:
        placeholders = ", ".join(["?"] * 10)
        statement = f"INSERT INTO candles ({_COLUMNS}) VALUES ({placeholders})"
        try:
            self._connection.execute("BEGIN TRANSACTION")
            self._connection.executemany(statement, rows)
            self._connection.execute("COMMIT")
        except duckdb.ConstraintException as exc:
            self._connection.execute("ROLLBACK")
            raise DuplicateCandleError(
                "Bulk insert aborted: duplicate candle key.", cause=exc
            ) from exc
        except duckdb.Error as exc:
            self._connection.execute("ROLLBACK")
            raise QueryError(
                "Bulk insert failed and was rolled back.", cause=exc
            ) from exc

    async def delete(self, key: SeriesKey, timestamp: datetime) -> bool:
        """Delete the candle at ``timestamp``; return whether one was removed."""
        async with self._lock:
            return await asyncio.to_thread(self._delete_sync, key, timestamp)

    def _delete_sync(self, key: SeriesKey, timestamp: datetime) -> bool:
        params = [*self._key_params(key), self._to_utc_naive(timestamp)]
        try:
            existed = (
                self._connection.execute(
                    f"SELECT 1 FROM candles WHERE {_KEY_PREDICATE} "
                    "AND timestamp = ? LIMIT 1",
                    params,
                ).fetchone()
                is not None
            )
            if existed:
                self._connection.execute(
                    f"DELETE FROM candles WHERE {_KEY_PREDICATE} AND timestamp = ?",
                    params,
                )
            return existed
        except duckdb.Error as exc:
            raise QueryError("Failed to delete candle.", cause=exc) from exc

    # -- Reads -------------------------------------------------------------

    async def get(self, key: SeriesKey, timestamp: datetime) -> Candle | None:
        """Return the candle at ``timestamp`` for ``key``, or ``None``."""
        async with self._lock:
            return await asyncio.to_thread(self._get_sync, key, timestamp)

    def _get_sync(self, key: SeriesKey, timestamp: datetime) -> Candle | None:
        params = [*self._key_params(key), self._to_utc_naive(timestamp)]
        try:
            row = self._connection.execute(
                f"SELECT {_COLUMNS} FROM candles WHERE {_KEY_PREDICATE} "
                "AND timestamp = ?",
                params,
            ).fetchone()
        except duckdb.Error as exc:
            raise QueryError("Failed to fetch candle.", cause=exc) from exc
        return self._from_row(row) if row is not None else None

    async def get_range(
        self, key: SeriesKey, start: datetime, end: datetime
    ) -> list[Candle]:
        """Return the ordered candles for ``key`` within ``[start, end]``."""
        async with self._lock:
            return await asyncio.to_thread(self._get_range_sync, key, start, end)

    def _get_range_sync(
        self, key: SeriesKey, start: datetime, end: datetime
    ) -> list[Candle]:
        params = [
            *self._key_params(key),
            self._to_utc_naive(start),
            self._to_utc_naive(end),
        ]
        try:
            cursor = self._connection.execute(
                f"SELECT {_COLUMNS} FROM candles WHERE {_KEY_PREDICATE} "
                "AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
                params,
            )
            candles: list[Candle] = []
            while True:
                chunk = cursor.fetchmany(_FETCH_CHUNK_SIZE)
                if not chunk:
                    break
                candles.extend(self._from_row(row) for row in chunk)
            return candles
        except duckdb.Error as exc:
            raise QueryError("Failed to fetch candle range.", cause=exc) from exc

    async def latest(self, key: SeriesKey) -> Candle | None:
        """Return the most recent candle for ``key``, or ``None``."""
        async with self._lock:
            return await asyncio.to_thread(self._latest_sync, key)

    def _latest_sync(self, key: SeriesKey) -> Candle | None:
        try:
            row = self._connection.execute(
                f"SELECT {_COLUMNS} FROM candles WHERE {_KEY_PREDICATE} "
                "ORDER BY timestamp DESC LIMIT 1",
                self._key_params(key),
            ).fetchone()
        except duckdb.Error as exc:
            raise QueryError("Failed to fetch latest candle.", cause=exc) from exc
        return self._from_row(row) if row is not None else None

    async def exists(self, key: SeriesKey, timestamp: datetime) -> bool:
        """Return whether a candle exists at ``timestamp`` for ``key``."""
        async with self._lock:
            return await asyncio.to_thread(self._exists_sync, key, timestamp)

    def _exists_sync(self, key: SeriesKey, timestamp: datetime) -> bool:
        params = [*self._key_params(key), self._to_utc_naive(timestamp)]
        try:
            row = self._connection.execute(
                f"SELECT 1 FROM candles WHERE {_KEY_PREDICATE} "
                "AND timestamp = ? LIMIT 1",
                params,
            ).fetchone()
        except duckdb.Error as exc:
            raise QueryError("Failed to check candle existence.", cause=exc) from exc
        return row is not None

    async def count(self, key: SeriesKey) -> int:
        """Return the number of stored candles for ``key``."""
        async with self._lock:
            return await asyncio.to_thread(self._count_sync, key)

    def _count_sync(self, key: SeriesKey) -> int:
        try:
            row = self._connection.execute(
                f"SELECT COUNT(*) FROM candles WHERE {_KEY_PREDICATE}",
                self._key_params(key),
            ).fetchone()
        except duckdb.Error as exc:
            raise QueryError("Failed to count candles.", cause=exc) from exc
        return int(row[0]) if row is not None else 0

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._connection.close()

    @staticmethod
    def _to_utc_naive(timestamp: datetime) -> datetime:
        """Normalise an aware timestamp to a UTC-naive value for storage."""
        return timestamp.astimezone(UTC).replace(tzinfo=None)
