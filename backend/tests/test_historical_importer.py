"""Unit tests for the historical import engine (no real APIs)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from app.core.clock import FakeClock
from app.core.timezone import INDIA_TZ
from app.market.calendar.config import InMemoryCalendarConfigProvider
from app.market.calendar.holidays import StaticHolidayProvider
from app.market.calendar.service import MarketCalendarService
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.importer.engine import HistoricalImportEngine
from app.market.historical.importer.models import (
    ImportConfig,
    ImportMode,
    ImportProgress,
    ImportSummary,
)
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import SeriesKey
from app.market.historical.storage.exceptions import QueryError
from app.market.historical.validation import ValidationEngine
from app.market.models import Candle as Bar
from app.market.models import (
    HistoricalData,
    MarketDepth,
    MarketStatusInfo,
    Quote,
    SymbolSearchResponse,
)
from app.providers.base import MarketDataProvider
from app.providers.exceptions import (
    AuthenticationError,
    InvalidSymbolError,
    NetworkError,
    ProviderError,
)

_BASE = datetime(2025, 1, 6, 9, 15, tzinfo=INDIA_TZ)  # Monday, a trading day.
_KEY = SeriesKey(symbol="RELIANCE", exchange=Exchange.NSE, interval=Interval.ONE_MINUTE)
_DAILY_KEY = SeriesKey(
    symbol="RELIANCE", exchange=Exchange.NSE, interval=Interval.ONE_DAY
)


def _daily_bar(day_offset: int) -> Bar:
    """Build a daily provider bar offset from the base date (a trading day)."""
    return Bar(
        timestamp=_BASE + timedelta(days=day_offset),
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("99"),
        close=Decimal("101"),
        volume=1000,
    )


def _bar(
    minute: int,
    *,
    high: str = "105",
    low: str = "99",
) -> Bar:
    """Build a provider bar offset from the base timestamp."""
    return Bar(
        timestamp=_BASE + timedelta(minutes=minute),
        open=Decimal("100"),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal("101"),
        volume=1000,
    )


class FakeMarketDataProvider(MarketDataProvider):
    """In-memory provider returning canned bars; never touches a network."""

    def __init__(self, bars: dict[str, list[Bar]]) -> None:
        self._bars = bars
        self.calls = 0
        self.failures: list[Exception] = []

    async def get_historical_data(
        self,
        symbol: str,
        interval: Interval,
        start_date: datetime,
        end_date: datetime,
        exchange: Exchange = Exchange.NSE,
    ) -> HistoricalData:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        selected = [
            bar
            for bar in self._bars.get(symbol, [])
            if start_date <= bar.timestamp <= end_date
        ]
        return HistoricalData(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            candles=tuple(selected),
        )

    async def get_market_status(
        self, exchange: Exchange = Exchange.NSE
    ) -> MarketStatusInfo:
        raise NotImplementedError

    async def get_quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Quote:
        raise NotImplementedError

    async def get_quotes(
        self, symbols: Sequence[str], exchange: Exchange = Exchange.NSE
    ) -> list[Quote]:
        raise NotImplementedError

    async def search_symbol(
        self, query: str, exchange: Exchange | None = None
    ) -> SymbolSearchResponse:
        raise NotImplementedError

    async def get_market_depth(
        self, symbol: str, exchange: Exchange = Exchange.NSE
    ) -> MarketDepth:
        raise NotImplementedError


class RecordingSleeper:
    """An async sleeper that records delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _calendar() -> MarketCalendarService:
    """Return a calendar with default NSE config and no holidays."""
    return MarketCalendarService(
        clock=FakeClock(_BASE),
        config_provider=InMemoryCalendarConfigProvider.with_defaults(),
        holiday_provider=StaticHolidayProvider({}),
    )


def _make_engine(
    provider: MarketDataProvider,
    *,
    data_engine: HistoricalDataEngine | None = None,
    config: ImportConfig | None = None,
    sleeper: RecordingSleeper | None = None,
) -> HistoricalImportEngine:
    """Build an import engine wired to fakes."""
    return HistoricalImportEngine(
        provider=provider,
        data_engine=data_engine
        or HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine()),
        calendar=_calendar(),
        clock=FakeClock(_BASE),
        config=config or ImportConfig(batch_size=100),
        sleeper=sleeper or RecordingSleeper(),
    )


# -- Full import -----------------------------------------------------------


async def test_full_import() -> None:
    """A full import downloads and stores the whole series."""
    provider = FakeMarketDataProvider({"RELIANCE": [_bar(i) for i in range(10)]})
    data_engine = HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())
    engine = _make_engine(provider, data_engine=data_engine)

    summary = await engine.import_history(
        ["RELIANCE"], Interval.ONE_MINUTE, _BASE, _BASE + timedelta(minutes=9)
    )

    assert summary.symbols_processed == 1
    assert summary.candles_downloaded == 10
    assert summary.candles_imported == 10
    assert await data_engine.count(_KEY) == 10


async def test_full_import_skips_when_already_covered() -> None:
    """Re-running a full import over stored data downloads nothing."""
    provider = FakeMarketDataProvider({"RELIANCE": [_bar(i) for i in range(10)]})
    data_engine = HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())
    engine = _make_engine(provider, data_engine=data_engine)
    end = _BASE + timedelta(minutes=9)

    await engine.import_history(["RELIANCE"], Interval.ONE_MINUTE, _BASE, end)
    provider.calls = 0
    summary = await engine.import_history(["RELIANCE"], Interval.ONE_MINUTE, _BASE, end)

    assert provider.calls == 0
    assert summary.candles_downloaded == 0


# -- Incremental import ----------------------------------------------------


async def test_incremental_import_only_fetches_new() -> None:
    """Incremental mode imports only candles newer than the latest stored."""
    all_bars = [_bar(i) for i in range(10)]
    provider = FakeMarketDataProvider({"RELIANCE": all_bars})
    data_engine = HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())
    engine = _make_engine(provider, data_engine=data_engine)

    # Seed the first 6 candles.
    await engine.import_history(
        ["RELIANCE"], Interval.ONE_MINUTE, _BASE, _BASE + timedelta(minutes=5)
    )

    summary = await engine.import_history(
        ["RELIANCE"],
        Interval.ONE_MINUTE,
        _BASE,
        _BASE + timedelta(minutes=9),
        mode=ImportMode.INCREMENTAL,
    )

    assert summary.candles_downloaded == 4  # candles 6..9 only
    assert summary.candles_imported == 4
    assert await data_engine.count(_KEY) == 10


# -- Resume + duplicates ---------------------------------------------------


async def test_resume_skips_covered_and_counts_duplicates() -> None:
    """Resuming a partial import skips done windows and dedupes the boundary."""
    all_bars = [_bar(i) for i in range(10)]
    provider = FakeMarketDataProvider({"RELIANCE": all_bars})
    repo = InMemoryCandleRepository()
    data_engine = HistoricalDataEngine(repo, ValidationEngine())
    engine = _make_engine(
        provider, data_engine=data_engine, config=ImportConfig(batch_size=5)
    )
    end = _BASE + timedelta(minutes=9)

    # Simulate an interrupted import that stored candles 0..6.
    from app.market.historical.models import Candle

    await data_engine.import_candles(
        [
            Candle(
                symbol="RELIANCE",
                exchange=Exchange.NSE,
                interval=Interval.ONE_MINUTE,
                timestamp=_BASE + timedelta(minutes=i),
                open=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("99"),
                close=Decimal("101"),
                volume=1000,
            )
            for i in range(7)
        ]
    )

    summary = await engine.import_history(
        ["RELIANCE"], Interval.ONE_MINUTE, _BASE, end, mode=ImportMode.FULL
    )

    # Window [0..4] is covered and skipped; window [5..9] is refetched: 5,6 are
    # duplicates and 7,8,9 are imported.
    assert summary.candles_downloaded == 5
    assert summary.duplicates == 2
    assert summary.candles_imported == 3
    assert await data_engine.count(_KEY) == 10


# -- Retry logic -----------------------------------------------------------


async def test_retry_then_success() -> None:
    """Retryable failures are retried with backoff until success."""
    provider = FakeMarketDataProvider({"RELIANCE": [_bar(i) for i in range(5)]})
    provider.failures = [NetworkError("blip"), NetworkError("blip")]
    sleeper = RecordingSleeper()
    engine = _make_engine(provider, sleeper=sleeper)

    summary = await engine.import_history(
        ["RELIANCE"], Interval.ONE_MINUTE, _BASE, _BASE + timedelta(minutes=4)
    )

    assert summary.retries == 2
    assert summary.candles_imported == 5
    assert len(sleeper.delays) == 2
    assert sleeper.delays[0] < sleeper.delays[1]  # exponential growth


async def test_retry_exhausted_marks_failed() -> None:
    """Exhausting retries records a failed request and imports nothing."""
    provider = FakeMarketDataProvider({"RELIANCE": [_bar(0)]})
    provider.failures = [NetworkError("x"), NetworkError("x")]
    engine = _make_engine(provider, config=ImportConfig(batch_size=100, max_retries=1))

    summary = await engine.import_history(
        ["RELIANCE"], Interval.ONE_MINUTE, _BASE, _BASE
    )

    assert summary.retries == 1
    assert summary.failed_requests == 1
    assert summary.candles_imported == 0


# -- Provider / storage / validation failures ------------------------------


async def test_non_retryable_provider_error() -> None:
    """A non-retryable provider error fails the window without retrying."""
    provider = FakeMarketDataProvider({"RELIANCE": [_bar(0)]})
    provider.failures = [InvalidSymbolError("unknown", symbol="RELIANCE")]
    engine = _make_engine(provider)

    summary = await engine.import_history(
        ["RELIANCE"], Interval.ONE_MINUTE, _BASE, _BASE
    )

    assert summary.failed_requests == 1
    assert summary.retries == 0
    assert summary.candles_imported == 0


async def test_daily_import_uses_a_single_provider_call() -> None:
    """A daily import folds a lone trailing day into one call (double-call fix).

    A history window that leaves a one-point remainder past a full batch used to
    plan a second, degenerate ``[end, end]`` window whose live SDK call failed
    even though the first window had already stored the data.
    """
    provider = FakeMarketDataProvider({"RELIANCE": [_daily_bar(i) for i in range(5)]})
    data_engine = HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())
    engine = _make_engine(
        provider, data_engine=data_engine, config=ImportConfig(batch_size=4)
    )

    summary = await engine.import_history(
        ["RELIANCE"], Interval.ONE_DAY, _BASE, _BASE + timedelta(days=4)
    )

    assert provider.calls == 1  # one window, not a trailing degenerate second
    assert summary.failed_requests == 0
    assert summary.candles_imported == 5
    assert await data_engine.count(_DAILY_KEY) == 5


async def test_provider_error_detail_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-retryable provider error logs the real underlying detail (BUG 2)."""
    provider = FakeMarketDataProvider({"RELIANCE": [_bar(0)]})
    provider.failures = [
        ProviderError(
            "Groww SDK historical candles failed.",
            details="end_time must be after start_time",
        )
    ]
    engine = _make_engine(provider)

    with caplog.at_level(logging.ERROR):
        summary = await engine.import_history(
            ["RELIANCE"], Interval.ONE_MINUTE, _BASE, _BASE
        )

    assert summary.failed_requests == 1
    # The real SDK message is surfaced, not just the wrapper "…failed." text.
    assert "end_time must be after start_time" in caplog.text


async def test_authentication_error_propagates() -> None:
    """Authentication failures abort the import."""
    provider = FakeMarketDataProvider({"RELIANCE": [_bar(0)]})
    provider.failures = [AuthenticationError("bad key")]
    engine = _make_engine(provider)

    with pytest.raises(AuthenticationError):
        await engine.import_history(["RELIANCE"], Interval.ONE_MINUTE, _BASE, _BASE)


async def test_storage_failure_is_handled() -> None:
    """A storage error is caught and recorded, not raised."""

    class FailingRepo(InMemoryCandleRepository):
        async def save_many(self, candles: Sequence[object]) -> None:
            raise QueryError("disk full")

    provider = FakeMarketDataProvider({"RELIANCE": [_bar(i) for i in range(3)]})
    data_engine = HistoricalDataEngine(FailingRepo(), ValidationEngine())
    engine = _make_engine(provider, data_engine=data_engine)

    summary = await engine.import_history(
        ["RELIANCE"], Interval.ONE_MINUTE, _BASE, _BASE + timedelta(minutes=2)
    )

    assert summary.failed_requests == 1
    assert summary.candles_imported == 0
    assert summary.candles_downloaded == 3


async def test_validation_failure_counts_invalid() -> None:
    """Bars failing candle validation are counted and skipped."""
    bars = [_bar(0), _bar(1, high="50", low="99"), _bar(2)]  # middle bar: high < low
    provider = FakeMarketDataProvider({"RELIANCE": bars})
    data_engine = HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())
    engine = _make_engine(provider, data_engine=data_engine)

    summary = await engine.import_history(
        ["RELIANCE"], Interval.ONE_MINUTE, _BASE, _BASE + timedelta(minutes=2)
    )

    assert summary.invalid_candles == 1
    assert summary.candles_imported == 2
    assert await data_engine.count(_KEY) == 2


# -- Progress callback & summary -------------------------------------------


async def test_progress_callback_invoked_per_batch() -> None:
    """The progress callback fires once per fetched batch."""
    provider = FakeMarketDataProvider({"RELIANCE": [_bar(i) for i in range(10)]})
    engine = _make_engine(provider, config=ImportConfig(batch_size=5))
    events: list[ImportProgress] = []

    await engine.import_history(
        ["RELIANCE"],
        Interval.ONE_MINUTE,
        _BASE,
        _BASE + timedelta(minutes=9),
        on_progress=events.append,
    )

    assert len(events) == 2  # two batches of five
    assert events[-1].candles_imported == 10
    assert events[-1].fraction == 1.0


async def test_summary_shape() -> None:
    """The summary exposes all required fields."""
    provider = FakeMarketDataProvider({"RELIANCE": [_bar(i) for i in range(3)]})
    engine = _make_engine(provider)

    summary = await engine.import_history(
        ["RELIANCE"], Interval.ONE_MINUTE, _BASE, _BASE + timedelta(minutes=2)
    )

    assert isinstance(summary, ImportSummary)
    assert summary.elapsed_time >= 0.0
    assert summary.symbols_processed == 1


async def test_invalid_range_rejected() -> None:
    """An inverted range is rejected."""
    provider = FakeMarketDataProvider({"RELIANCE": []})
    engine = _make_engine(provider)

    with pytest.raises(ValueError):
        await engine.import_history(
            ["RELIANCE"], Interval.ONE_MINUTE, _BASE + timedelta(minutes=1), _BASE
        )


def test_di_returns_singleton() -> None:
    """The DI accessor returns a cached singleton engine."""
    from app.market.historical.importer.dependencies import (
        _import_engine_singleton,
        get_historical_import_engine,
    )

    _import_engine_singleton.cache_clear()
    try:
        assert get_historical_import_engine() is get_historical_import_engine()
    finally:
        _import_engine_singleton.cache_clear()
