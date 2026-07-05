"""Unit tests for the historical market-data engine."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from app.core.timezone import INDIA_TZ
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.enums import DuplicatePolicy, IssueType
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import Candle, SeriesKey
from app.market.historical.validation import ValidationEngine
from pydantic import ValidationError

_BASE = datetime(2025, 1, 6, 9, 15, tzinfo=INDIA_TZ)


def _candle(
    minute_offset: int = 0,
    *,
    symbol: str = "RELIANCE",
    interval: Interval = Interval.ONE_MINUTE,
    close: str = "101",
    volume: int = 1000,
) -> Candle:
    """Build a valid candle offset from the base timestamp."""
    return Candle(
        symbol=symbol,
        exchange=Exchange.NSE,
        interval=interval,
        timestamp=_BASE + timedelta(minutes=minute_offset),
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("99"),
        close=Decimal(close),
        volume=volume,
    )


@pytest.fixture
def engine() -> HistoricalDataEngine:
    """Return an engine backed by a fresh in-memory repository."""
    return HistoricalDataEngine(
        repository=InMemoryCandleRepository(), validator=ValidationEngine()
    )


# -- Candle model ----------------------------------------------------------


def test_candle_normalizes_symbol() -> None:
    """Symbols are trimmed and upper-cased."""
    assert _candle(symbol="  reliance ").symbol == "RELIANCE"


def test_candle_rejects_naive_timestamp() -> None:
    """A naive timestamp is rejected."""
    with pytest.raises(ValidationError):
        Candle(
            symbol="X",
            exchange=Exchange.NSE,
            interval=Interval.ONE_MINUTE,
            timestamp=datetime(2025, 1, 6, 9, 15),
            open=Decimal("1"),
            high=Decimal("2"),
            low=Decimal("1"),
            close=Decimal("2"),
            volume=1,
        )


def test_candle_rejects_invalid_ohlc() -> None:
    """A high below the open is rejected."""
    with pytest.raises(ValidationError):
        Candle(
            symbol="X",
            exchange=Exchange.NSE,
            interval=Interval.ONE_MINUTE,
            timestamp=_BASE,
            open=Decimal("10"),
            high=Decimal("5"),
            low=Decimal("1"),
            close=Decimal("4"),
            volume=1,
        )


def test_candle_rejects_negative_volume() -> None:
    """Negative volume is rejected."""
    with pytest.raises(ValidationError):
        _candle(volume=-1)


def test_candle_is_frozen() -> None:
    """Candles are immutable."""
    candle = _candle()
    with pytest.raises(ValidationError):
        candle.close = Decimal("1")  # type: ignore[misc]


# -- Repository ------------------------------------------------------------


async def test_repository_crud() -> None:
    """The in-memory repository stores and retrieves candles."""
    repo = InMemoryCandleRepository()
    key = SeriesKey(
        symbol="RELIANCE", exchange=Exchange.NSE, interval=Interval.ONE_MINUTE
    )
    candles = [_candle(i) for i in range(3)]
    await repo.save_many(candles)

    assert await repo.count(key) == 3
    assert await repo.exists(key, _BASE) is True
    assert await repo.get(key, _BASE) == candles[0]
    latest = await repo.latest(key)
    assert latest is not None and latest.timestamp == candles[-1].timestamp
    ranged = await repo.get_range(key, _BASE, _BASE + timedelta(minutes=1))
    assert len(ranged) == 2
    assert await repo.delete(key, _BASE) is True
    assert await repo.count(key) == 2


# -- Validation engine -----------------------------------------------------


def test_validate_candle_flags_invalid_ohlc() -> None:
    """A bypassed-construction candle is flagged as invalid OHLC."""
    bad = Candle.model_construct(
        symbol="X",
        exchange=Exchange.NSE,
        interval=Interval.ONE_MINUTE,
        timestamp=_BASE,
        open=Decimal("10"),
        high=Decimal("1"),
        low=Decimal("5"),
        close=Decimal("8"),
        volume=10,
        open_interest=None,
    )
    issues = ValidationEngine().validate_candle(bad)
    assert any(i.issue_type is IssueType.INVALID_OHLC for i in issues)


def test_detect_duplicates() -> None:
    """Duplicate timestamps are detected."""
    issues = ValidationEngine().detect_duplicates([_candle(0), _candle(0)])
    assert len(issues) == 1
    assert issues[0].issue_type is IssueType.DUPLICATE


def test_detect_unordered() -> None:
    """Out-of-order candles are detected."""
    issues = ValidationEngine().detect_unordered([_candle(2), _candle(1)])
    assert issues and issues[0].issue_type is IssueType.UNORDERED


def test_detect_overlaps() -> None:
    """Candles closer than one interval overlap."""
    candles = [
        _candle(0, interval=Interval.FIVE_MINUTE),
        _candle(2, interval=Interval.FIVE_MINUTE),
    ]
    issues = ValidationEngine().detect_overlaps(candles, Interval.FIVE_MINUTE)
    assert issues and issues[0].issue_type is IssueType.OVERLAP


def test_detect_gaps() -> None:
    """Missing candle slots are counted."""
    candles = [_candle(0), _candle(1), _candle(3)]
    issues, missing = ValidationEngine().detect_gaps(candles, Interval.ONE_MINUTE)
    assert missing == 1
    assert issues and issues[0].issue_type is IssueType.GAP


# -- Engine import ---------------------------------------------------------


async def test_import_fresh_series(engine: HistoricalDataEngine) -> None:
    """A clean series imports fully with no issues."""
    report = await engine.import_candles([_candle(i) for i in range(3)])
    assert report.imported == 3
    assert report.duplicates == 0
    assert report.gaps_found == 0


async def test_import_detects_batch_duplicate(engine: HistoricalDataEngine) -> None:
    """A duplicate within the batch is skipped and reported."""
    report = await engine.import_candles([_candle(0), _candle(0), _candle(1)])
    assert report.imported == 2
    assert report.duplicates == 1
    assert report.skipped == 1


async def test_import_skips_existing_by_default(
    engine: HistoricalDataEngine,
) -> None:
    """Re-importing an existing candle is skipped under the SKIP policy."""
    await engine.import_candles([_candle(0)])
    report = await engine.import_candles([_candle(0, close="104")])
    assert report.imported == 0
    assert report.duplicates == 1
    key = SeriesKey(
        symbol="RELIANCE", exchange=Exchange.NSE, interval=Interval.ONE_MINUTE
    )
    stored = await engine.get_candle(key, _BASE)
    assert stored is not None and stored.close == Decimal("101")  # unchanged


async def test_import_updates_existing_when_requested(
    engine: HistoricalDataEngine,
) -> None:
    """The UPDATE policy overwrites and counts repaired candles."""
    await engine.import_candles([_candle(0)])
    report = await engine.import_candles(
        [_candle(0, close="104")], on_duplicate=DuplicatePolicy.UPDATE
    )
    assert report.repaired == 1
    assert report.imported == 0
    key = SeriesKey(
        symbol="RELIANCE", exchange=Exchange.NSE, interval=Interval.ONE_MINUTE
    )
    stored = await engine.get_candle(key, _BASE)
    assert stored is not None and stored.close == Decimal("104")


async def test_import_reports_gaps(engine: HistoricalDataEngine) -> None:
    """Gaps are reported during import."""
    report = await engine.import_candles([_candle(0), _candle(1), _candle(3)])
    assert report.gaps_found == 1
    assert report.imported == 3


async def test_import_counts_invalid(engine: HistoricalDataEngine) -> None:
    """Invalid candles are counted and not stored."""
    bad = Candle.model_construct(
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        interval=Interval.ONE_MINUTE,
        timestamp=_BASE + timedelta(minutes=1),
        open=Decimal("10"),
        high=Decimal("1"),
        low=Decimal("5"),
        close=Decimal("8"),
        volume=-5,
        open_interest=None,
    )
    report = await engine.import_candles([_candle(0), bad])
    assert report.invalid == 1
    assert report.imported == 1
    assert report.skipped == 1


async def test_import_detects_unordered_input(
    engine: HistoricalDataEngine,
) -> None:
    """Unordered input is reported but still stored in order."""
    report = await engine.import_candles([_candle(2), _candle(0), _candle(1)])
    assert any(i.issue_type is IssueType.UNORDERED for i in report.issues)
    assert report.imported == 3


async def test_import_rejects_multiple_series(
    engine: HistoricalDataEngine,
) -> None:
    """Mixing series in one import is rejected."""
    with pytest.raises(ValueError):
        await engine.import_candles([_candle(0), _candle(1, symbol="TCS")])


async def test_import_empty_returns_empty_report(
    engine: HistoricalDataEngine,
) -> None:
    """Importing nothing yields an empty report."""
    report = await engine.import_candles([])
    assert report.imported == 0
    assert report.issues == ()


# -- Retrieval & DI --------------------------------------------------------


async def test_engine_retrieval(engine: HistoricalDataEngine) -> None:
    """Retrieval helpers delegate to the repository."""
    await engine.import_candles([_candle(i) for i in range(3)])
    key = SeriesKey(
        symbol="RELIANCE", exchange=Exchange.NSE, interval=Interval.ONE_MINUTE
    )
    assert await engine.count(key) == 3
    latest = await engine.get_latest(key)
    assert latest is not None
    got = await engine.get_candles(key, _BASE, _BASE + timedelta(minutes=5))
    assert len(got) == 3


def test_di_returns_singleton() -> None:
    """The DI accessor returns a cached singleton engine."""
    from app.market.historical.dependencies import (
        _engine_singleton,
        get_historical_data_engine,
    )

    _engine_singleton.cache_clear()
    try:
        assert get_historical_data_engine() is get_historical_data_engine()
    finally:
        _engine_singleton.cache_clear()
