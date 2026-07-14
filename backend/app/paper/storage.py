"""DuckDB implementation of the paper-trade repository.

Persists paper trades in the existing analytical store. Timestamps are stored
UTC-naive and re-tagged UTC-aware on read (matching the candle store), so the
recorded instant survives round-trips without an extra dependency. Blocking
DuckDB calls run in a worker thread and are serialised with a lock.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import duckdb

from app.core.logging import get_logger
from app.market.enums import Exchange, Interval
from app.market.historical.storage.exceptions import (
    QueryError,
    SchemaInitializationError,
    StorageConnectionError,
)
from app.paper.models import ExitReason, PaperTrade, TradeStatus
from app.paper.repository import PaperTradeRepository

logger = get_logger(__name__)

_COLUMNS = (
    "id, symbol, strategy, exchange, interval, entry_price, stop_price, "
    "target_price, size, opened_at, status, last_price, pnl, exit_price, "
    "exit_reason, closed_at"
)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id           VARCHAR PRIMARY KEY,
    symbol       VARCHAR NOT NULL,
    strategy     VARCHAR NOT NULL,
    exchange     VARCHAR NOT NULL,
    interval     VARCHAR NOT NULL,
    entry_price  DOUBLE  NOT NULL,
    stop_price   DOUBLE  NOT NULL,
    target_price DOUBLE  NOT NULL,
    size         DOUBLE  NOT NULL,
    opened_at    TIMESTAMP NOT NULL,
    status       VARCHAR NOT NULL,
    last_price   DOUBLE  NOT NULL,
    pnl          DOUBLE  NOT NULL,
    exit_price   DOUBLE,
    exit_reason  VARCHAR,
    closed_at    TIMESTAMP
);
"""


class DuckDBPaperTradeRepository(PaperTradeRepository):
    """A :class:`PaperTradeRepository` backed by a DuckDB database file."""

    def __init__(self, database_path: str) -> None:
        """Open the database and ensure the schema exists.

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
        try:
            self._connection.execute(_CREATE_TABLE)
        except duckdb.Error as exc:
            raise SchemaInitializationError(
                "Failed to initialise the paper-trades schema.", cause=exc
            ) from exc

    async def upsert(self, trade: PaperTrade) -> None:
        """Insert or replace a trade by its id."""
        async with self._lock:
            await asyncio.to_thread(self._upsert_sync, trade)

    def _upsert_sync(self, trade: PaperTrade) -> None:
        placeholders = ", ".join(["?"] * 16)
        try:
            self._connection.execute(
                f"INSERT OR REPLACE INTO paper_trades ({_COLUMNS}) "
                f"VALUES ({placeholders})",
                self._to_row(trade),
            )
        except duckdb.Error as exc:
            raise QueryError("Failed to save paper trade.", cause=exc) from exc

    async def all(self) -> list[PaperTrade]:
        """Return every stored trade ordered by open time."""
        async with self._lock:
            return await asyncio.to_thread(self._all_sync)

    def _all_sync(self) -> list[PaperTrade]:
        try:
            rows = self._connection.execute(
                f"SELECT {_COLUMNS} FROM paper_trades ORDER BY opened_at"
            ).fetchall()
        except duckdb.Error as exc:
            raise QueryError("Failed to fetch paper trades.", cause=exc) from exc
        return [self._from_row(row) for row in rows]

    def close(self) -> None:
        """Close the underlying database connection."""
        self._connection.close()

    @staticmethod
    def _to_row(trade: PaperTrade) -> list[Any]:
        return [
            trade.id,
            trade.symbol,
            trade.strategy,
            trade.exchange.value,
            trade.interval.value,
            trade.entry_price,
            trade.stop_price,
            trade.target_price,
            trade.size,
            _naive(trade.opened_at),
            trade.status.value,
            trade.last_price,
            trade.pnl,
            trade.exit_price,
            trade.exit_reason.value if trade.exit_reason is not None else None,
            _naive(trade.closed_at) if trade.closed_at is not None else None,
        ]

    @staticmethod
    def _from_row(row: Any) -> PaperTrade:
        return PaperTrade(
            id=row[0],
            symbol=row[1],
            strategy=row[2],
            exchange=Exchange(row[3]),
            interval=Interval(row[4]),
            entry_price=row[5],
            stop_price=row[6],
            target_price=row[7],
            size=row[8],
            opened_at=row[9].replace(tzinfo=UTC),
            status=TradeStatus(row[10]),
            last_price=row[11],
            pnl=row[12],
            exit_price=row[13],
            exit_reason=ExitReason(row[14]) if row[14] is not None else None,
            closed_at=row[15].replace(tzinfo=UTC) if row[15] is not None else None,
        )


def _naive(timestamp: datetime) -> datetime:
    """Normalise an aware timestamp to a UTC-naive value for storage."""
    return timestamp.astimezone(UTC).replace(tzinfo=None)
