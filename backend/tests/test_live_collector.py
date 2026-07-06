"""Unit tests for the live market collector (no real HTTP)."""

from __future__ import annotations

import asyncio
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
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import SeriesKey
from app.market.historical.repository import CandleRepository
from app.market.live.collector import LiveMarketCollector
from app.market.live.enums import CollectionMode
from app.market.live.models import CollectorConfig
from app.market.models import (
    HistoricalData,
    MarketDepth,
    MarketStatusInfo,
    Quote,
    SymbolSearchResponse,
)
from app.providers.base import MarketDataProvider
from app.providers.exceptions import AuthenticationError, NetworkError

# Monday 10:00 IST is an open trading session; Saturday is closed.
_OPEN = datetime(2025, 1, 6, 10, 0, tzinfo=INDIA_TZ)
_CLOSED = datetime(2025, 1, 11, 10, 0, tzinfo=INDIA_TZ)


def _quote(symbol: str, price: str, when: datetime) -> Quote:
    """Build a quote."""
    return Quote(
        symbol=symbol,
        exchange=Exchange.NSE,
        last_price=Decimal(price),
        volume=100,
        timestamp=when,
    )


class FakeMarketDataProvider(MarketDataProvider):
    """In-memory provider returning scripted quotes; no network."""

    def __init__(self) -> None:
        self.batches: list[list[Quote]] = []
        self.failures: list[Exception] = []
        self.calls = 0

    def queue(self, quotes: list[Quote]) -> None:
        """Queue a batch of quotes to be returned by successive calls."""
        self.batches.append(quotes)

    async def get_quotes(
        self, symbols: Sequence[str], exchange: Exchange = Exchange.NSE
    ) -> list[Quote]:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        if self.batches:
            return self.batches.pop(0)
        return []

    async def get_quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Quote:
        raise NotImplementedError

    async def get_market_status(
        self, exchange: Exchange = Exchange.NSE
    ) -> MarketStatusInfo:
        raise NotImplementedError

    async def get_historical_data(
        self,
        symbol: str,
        interval: Interval,
        start_date: datetime,
        end_date: datetime,
        exchange: Exchange = Exchange.NSE,
    ) -> HistoricalData:
        raise NotImplementedError

    async def search_symbol(
        self, query: str, exchange: Exchange | None = None
    ) -> SymbolSearchResponse:
        raise NotImplementedError

    async def get_market_depth(
        self, symbol: str, exchange: Exchange = Exchange.NSE
    ) -> MarketDepth:
        raise NotImplementedError


def _calendar(clock: FakeClock) -> MarketCalendarService:
    """Return a calendar with default NSE config and no holidays."""
    return MarketCalendarService(
        clock=clock,
        config_provider=InMemoryCalendarConfigProvider.with_defaults(),
        holiday_provider=StaticHolidayProvider({}),
    )


def _collector(
    provider: MarketDataProvider,
    clock: FakeClock,
    *,
    repository: CandleRepository | None = None,
    watchlist: tuple[str, ...] = ("RELIANCE",),
    mode: CollectionMode = CollectionMode.LIVE,
    stale_after: float = 30.0,
    max_retries: int = 3,
    sleeper: object | None = None,
) -> LiveMarketCollector:
    """Build a collector wired to fakes."""
    config = CollectorConfig(
        watchlist=watchlist,
        interval=Interval.ONE_MINUTE,
        mode=mode,
        poll_interval_seconds=1.0,
        stale_after_seconds=stale_after,
        max_retries=max_retries,
    )
    return LiveMarketCollector(
        provider=provider,
        calendar=_calendar(clock),
        clock=clock,
        repository=repository or InMemoryCandleRepository(),
        config=config,
        sleeper=sleeper,  # type: ignore[arg-type]
    )


class StopAfterFirstSleeper:
    """A sleeper that stops the collector on its first invocation."""

    def __init__(self) -> None:
        self.collector: LiveMarketCollector | None = None
        self.calls = 0

    async def __call__(self, delay: float) -> None:
        self.calls += 1
        if self.collector is not None:
            self.collector.request_stop()


# -- poll_once: persistence & data quality ---------------------------------


async def test_poll_persists_snapshot() -> None:
    """A polled quote is persisted as a snapshot candle."""
    provider = FakeMarketDataProvider()
    provider.queue([_quote("RELIANCE", "2500", _OPEN)])
    repo = InMemoryCandleRepository()
    collector = _collector(provider, FakeClock(_OPEN), repository=repo)

    await collector.poll_once()

    key = SeriesKey(
        symbol="RELIANCE", exchange=Exchange.NSE, interval=Interval.ONE_MINUTE
    )
    assert await repo.count(key) == 1
    assert collector.metrics().successful_updates == 1


async def test_duplicate_tick_filtered() -> None:
    """A repeated timestamp is counted as a duplicate and not re-stored."""
    provider = FakeMarketDataProvider()
    provider.queue([_quote("RELIANCE", "2500", _OPEN)])
    provider.queue([_quote("RELIANCE", "2500", _OPEN)])  # same timestamp
    collector = _collector(provider, FakeClock(_OPEN))

    await collector.poll_once()
    await collector.poll_once()

    metrics = collector.metrics()
    assert metrics.successful_updates == 1
    assert metrics.duplicate_ticks == 1


async def test_out_of_order_filtered() -> None:
    """An earlier timestamp after a later one is rejected."""
    provider = FakeMarketDataProvider()
    provider.queue([_quote("RELIANCE", "2500", _OPEN)])
    provider.queue([_quote("RELIANCE", "2490", _OPEN - timedelta(minutes=1))])
    collector = _collector(provider, FakeClock(_OPEN))

    await collector.poll_once()
    await collector.poll_once()

    assert collector.metrics().out_of_order == 1


async def test_missing_update_counted() -> None:
    """A watchlist symbol absent from the response is counted as missing."""
    provider = FakeMarketDataProvider()
    provider.queue([_quote("RELIANCE", "2500", _OPEN)])  # TCS missing
    collector = _collector(provider, FakeClock(_OPEN), watchlist=("RELIANCE", "TCS"))

    await collector.poll_once()

    assert collector.metrics().missing_updates == 1


async def test_stale_detection() -> None:
    """An unchanged price beyond the threshold is flagged stale."""
    provider = FakeMarketDataProvider()
    clock = FakeClock(_OPEN)
    provider.queue([_quote("RELIANCE", "2500", _OPEN)])
    provider.queue([_quote("RELIANCE", "2500", _OPEN + timedelta(minutes=2))])
    collector = _collector(provider, clock, stale_after=30.0)

    await collector.poll_once()
    clock.advance(timedelta(seconds=60))  # price unchanged for > 30s
    await collector.poll_once()

    assert collector.metrics().stale_detections == 1


# -- Failure handling ------------------------------------------------------


async def test_retry_then_success() -> None:
    """Retryable failures are retried and counted."""
    provider = FakeMarketDataProvider()
    provider.failures = [NetworkError("blip"), NetworkError("blip")]
    provider.queue([_quote("RELIANCE", "2500", _OPEN)])
    sleeper = StopAfterFirstSleeper()  # used only as an async no-op here
    collector = LiveMarketCollector(
        provider=provider,
        calendar=_calendar(FakeClock(_OPEN)),
        clock=FakeClock(_OPEN),
        repository=InMemoryCandleRepository(),
        config=CollectorConfig(watchlist=("RELIANCE",), max_retries=3),
        sleeper=sleeper,
    )

    await collector.poll_once()

    metrics = collector.metrics()
    assert metrics.retry_count == 2
    assert metrics.successful_updates == 1


async def test_failed_cycle_counted() -> None:
    """A cycle that fails after retries increments failed_updates."""
    provider = FakeMarketDataProvider()
    provider.failures = [NetworkError("x"), NetworkError("x")]

    async def _noop(_delay: float) -> None:
        return None

    collector = _collector(provider, FakeClock(_OPEN), max_retries=1, sleeper=_noop)

    await collector.poll_once()

    assert collector.metrics().failed_updates == 1
    assert collector.metrics().successful_updates == 0


async def test_authentication_error_propagates() -> None:
    """Authentication failures are not recoverable and propagate."""
    provider = FakeMarketDataProvider()
    provider.failures = [AuthenticationError("bad key")]
    collector = _collector(provider, FakeClock(_OPEN))

    with pytest.raises(AuthenticationError):
        await collector.poll_once()


# -- Market open / closed loop ---------------------------------------------


async def test_loop_collects_when_open() -> None:
    """The loop polls when the market is open."""
    provider = FakeMarketDataProvider()
    provider.queue([_quote("RELIANCE", "2500", _OPEN)])
    sleeper = StopAfterFirstSleeper()
    collector = _collector(provider, FakeClock(_OPEN), sleeper=sleeper)
    sleeper.collector = collector

    await collector.run()  # polls once, then sleeper stops it

    assert provider.calls == 1
    assert collector.metrics().successful_updates == 1


async def test_loop_does_not_collect_when_closed() -> None:
    """The loop waits for the next session instead of polling when closed."""
    provider = FakeMarketDataProvider()
    sleeper = StopAfterFirstSleeper()
    collector = _collector(provider, FakeClock(_CLOSED), sleeper=sleeper)
    sleeper.collector = collector

    await collector.run()  # closed -> waits for next open, sleeper stops it

    assert provider.calls == 0
    assert collector.metrics().requests_made == 0


# -- Lifecycle: cancellation, graceful shutdown, restart -------------------


async def test_cancellation() -> None:
    """Cancelling the run task raises CancelledError out of the loop."""
    provider = FakeMarketDataProvider()

    async def _slow(_delay: float) -> None:
        await asyncio.sleep(3600)

    collector = _collector(provider, FakeClock(_OPEN), sleeper=_slow)
    task = asyncio.create_task(collector.run())
    await asyncio.sleep(0.02)  # let it poll and enter the sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_graceful_shutdown_via_stop() -> None:
    """stop() ends the loop cleanly and leaves it not running."""
    provider = FakeMarketDataProvider()
    for _ in range(100):
        provider.queue([_quote("RELIANCE", "2500", _OPEN)])

    async def _slow(_delay: float) -> None:
        await asyncio.sleep(3600)

    collector = _collector(provider, FakeClock(_OPEN), sleeper=_slow)
    collector.start()
    await asyncio.sleep(0.02)
    await collector.stop()

    assert not collector.is_running
    assert collector.metrics().successful_updates >= 1


async def test_restart() -> None:
    """A collector can be restarted after stopping."""
    provider = FakeMarketDataProvider()
    for _ in range(10):
        provider.queue([_quote("RELIANCE", "2500", _OPEN)])

    sleeper = StopAfterFirstSleeper()
    collector = _collector(provider, FakeClock(_OPEN), sleeper=sleeper)
    sleeper.collector = collector

    await collector.run()  # first session: one poll
    first = collector.metrics().requests_made
    await collector.run()  # restart: another poll
    assert collector.metrics().requests_made > first


# -- Modes -----------------------------------------------------------------


async def test_backtest_replay_is_stub() -> None:
    """Backtest replay mode is a stub and refuses to run."""
    provider = FakeMarketDataProvider()
    collector = _collector(
        provider, FakeClock(_OPEN), mode=CollectionMode.BACKTEST_REPLAY
    )
    with pytest.raises(NotImplementedError):
        await collector.poll_once()


async def test_paper_mode_collects() -> None:
    """Paper mode collects exactly like live mode."""
    provider = FakeMarketDataProvider()
    provider.queue([_quote("RELIANCE", "2500", _OPEN)])
    collector = _collector(provider, FakeClock(_OPEN), mode=CollectionMode.PAPER)

    await collector.poll_once()

    assert collector.metrics().successful_updates == 1


# -- Metrics ---------------------------------------------------------------


async def test_metrics_snapshot() -> None:
    """Metrics reflect requests, updates and uptime."""
    provider = FakeMarketDataProvider()
    provider.queue([_quote("RELIANCE", "2500", _OPEN)])
    clock = FakeClock(_OPEN)
    collector = _collector(provider, clock)

    # Seed started_at via run bookkeeping without the loop.
    await collector.poll_once()
    metrics = collector.metrics()

    assert metrics.requests_made == 1
    assert metrics.successful_updates == 1
    assert metrics.uptime_seconds >= 0.0
