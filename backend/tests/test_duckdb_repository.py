"""Integration tests for the DuckDB candle repository."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from app.core.timezone import INDIA_TZ
from app.market.enums import Exchange, Interval
from app.market.historical.models import Candle, SeriesKey
from app.market.historical.storage.duckdb_repository import DuckDBCandleRepository
from app.market.historical.storage.exceptions import DuplicateCandleError

_BASE = datetime(2025, 1, 6, 9, 15, tzinfo=INDIA_TZ)
_KEY = SeriesKey(symbol="RELIANCE", exchange=Exchange.NSE, interval=Interval.ONE_MINUTE)


def _candle(minute_offset: int = 0, *, close: str = "101") -> Candle:
    """Build a valid candle offset from the base timestamp."""
    return Candle(
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        interval=Interval.ONE_MINUTE,
        timestamp=_BASE + timedelta(minutes=minute_offset),
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("99"),
        close=Decimal(close),
        volume=1000,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[DuckDBCandleRepository]:
    """Return a repository backed by a temporary DuckDB file."""
    repository = DuckDBCandleRepository(str(tmp_path / "candles.duckdb"))
    try:
        yield repository
    finally:
        repository.close()


# -- Schema ----------------------------------------------------------------


def test_schema_is_created(tmp_path: Path) -> None:
    """Opening the repository creates the candles table and index."""
    repository = DuckDBCandleRepository(str(tmp_path / "s.duckdb"))
    try:
        tables = repository._connection.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
        assert ("candles",) in tables
    finally:
        repository.close()


def test_reopen_is_idempotent(tmp_path: Path) -> None:
    """Reopening an existing database does not fail on schema creation."""
    path = str(tmp_path / "re.duckdb")
    first = DuckDBCandleRepository(path)
    first.close()
    second = DuckDBCandleRepository(path)
    second.close()


# -- Single insert ---------------------------------------------------------


async def test_single_insert_and_get(repo: DuckDBCandleRepository) -> None:
    """A saved candle is retrievable and equal by instant."""
    candle = _candle()
    await repo.save(candle)
    fetched = await repo.get(_KEY, candle.timestamp)
    assert fetched == candle


async def test_get_missing_returns_none(repo: DuckDBCandleRepository) -> None:
    """Fetching an absent candle returns None."""
    assert await repo.get(_KEY, _BASE) is None


async def test_save_overwrites(repo: DuckDBCandleRepository) -> None:
    """Saving the same key overwrites the stored candle."""
    await repo.save(_candle(close="101"))
    await repo.save(_candle(close="104"))
    fetched = await repo.get(_KEY, _BASE)
    assert fetched is not None and fetched.close == Decimal("104")
    assert await repo.count(_KEY) == 1


async def test_timestamp_roundtrip_is_timezone_aware(
    repo: DuckDBCandleRepository,
) -> None:
    """Retrieved timestamps are timezone-aware and preserve the instant."""
    candle = _candle()
    await repo.save(candle)
    fetched = await repo.get(_KEY, candle.timestamp)
    assert fetched is not None
    assert fetched.timestamp.tzinfo is not None
    assert fetched.timestamp == candle.timestamp


# -- Bulk insert -----------------------------------------------------------


async def test_bulk_insert(repo: DuckDBCandleRepository) -> None:
    """Bulk insert stores every candle."""
    await repo.save_many([_candle(i) for i in range(50)])
    assert await repo.count(_KEY) == 50


async def test_bulk_insert_empty_is_noop(repo: DuckDBCandleRepository) -> None:
    """Bulk inserting nothing is a no-op."""
    await repo.save_many([])
    assert await repo.count(_KEY) == 0


# -- Duplicate handling / transactions -------------------------------------


async def test_bulk_duplicate_raises_and_rolls_back(
    repo: DuckDBCandleRepository,
) -> None:
    """A duplicate in a bulk insert aborts the whole batch (no partial write)."""
    await repo.save_many([_candle(0), _candle(1)])
    with pytest.raises(DuplicateCandleError):
        # _candle(1) already exists; _candle(2) is new but must not be written.
        await repo.save_many([_candle(2), _candle(1)])
    assert await repo.count(_KEY) == 2
    assert await repo.exists(_KEY, _BASE + timedelta(minutes=2)) is False


async def test_exists(repo: DuckDBCandleRepository) -> None:
    """Existence reflects stored candles."""
    await repo.save(_candle())
    assert await repo.exists(_KEY, _BASE) is True
    assert await repo.exists(_KEY, _BASE + timedelta(minutes=5)) is False


# -- Range queries / latest ------------------------------------------------


async def test_range_query_is_ordered_and_inclusive(
    repo: DuckDBCandleRepository,
) -> None:
    """Range queries return inclusive, chronologically ordered candles."""
    await repo.save_many([_candle(i) for i in range(5)])
    result = await repo.get_range(_KEY, _BASE, _BASE + timedelta(minutes=2))
    assert [c.timestamp for c in result] == [
        _BASE,
        _BASE + timedelta(minutes=1),
        _BASE + timedelta(minutes=2),
    ]


async def test_latest(repo: DuckDBCandleRepository) -> None:
    """Latest returns the most recent candle."""
    await repo.save_many([_candle(i) for i in range(3)])
    latest = await repo.latest(_KEY)
    assert latest is not None
    assert latest.timestamp == _BASE + timedelta(minutes=2)


async def test_latest_empty_returns_none(repo: DuckDBCandleRepository) -> None:
    """Latest on an empty series returns None."""
    assert await repo.latest(_KEY) is None


# -- Delete ----------------------------------------------------------------


async def test_delete(repo: DuckDBCandleRepository) -> None:
    """Deleting an existing candle removes it and reports success."""
    await repo.save(_candle())
    assert await repo.delete(_KEY, _BASE) is True
    assert await repo.count(_KEY) == 0


async def test_delete_missing_returns_false(repo: DuckDBCandleRepository) -> None:
    """Deleting an absent candle returns False."""
    assert await repo.delete(_KEY, _BASE) is False


# -- Large dataset ---------------------------------------------------------


async def test_large_dataset(repo: DuckDBCandleRepository) -> None:
    """A large bulk insert stores and ranges over many candles efficiently."""
    count = 20_000
    await repo.save_many([_candle(i) for i in range(count)])
    assert await repo.count(_KEY) == count
    window = await repo.get_range(_KEY, _BASE, _BASE + timedelta(minutes=count - 1))
    assert len(window) == count


# -- Interfaces & DI -------------------------------------------------------


def test_repository_is_candle_repository(repo: DuckDBCandleRepository) -> None:
    """The DuckDB repository satisfies the domain interface."""
    from app.market.historical.repository import CandleRepository

    assert isinstance(repo, CandleRepository)


def test_di_returns_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DuckDB DI accessor returns a cached singleton engine."""
    import app.config.settings as settings_module
    from app.market.historical.storage import dependencies as di

    monkeypatch.setenv("DUCKDB_PATH", ":memory:")
    settings_module.get_settings.cache_clear()
    di.get_duckdb_candle_repository.cache_clear()
    di._duckdb_engine_singleton.cache_clear()
    try:
        first = di.get_duckdb_historical_data_engine()
        second = di.get_duckdb_historical_data_engine()
        assert first is second
    finally:
        di._duckdb_engine_singleton.cache_clear()
        di.get_duckdb_candle_repository.cache_clear()
        settings_module.get_settings.cache_clear()
