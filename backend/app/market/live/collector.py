"""The live market collector.

Continuously polls a :class:`MarketDataProvider` for the configured watchlist,
persists each accepted quote as a snapshot candle, and exposes runtime metrics.
It collects only while the market is open (per :class:`MarketCalendarService`),
sleeps until the next session when closed, and supports drift-free scheduling,
graceful shutdown and cancellation. It contains no trading logic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from app.core.clock import Clock
from app.core.logging import get_logger
from app.market.calendar.service import MarketCalendarService
from app.market.historical.models import Candle
from app.market.historical.repository import CandleRepository
from app.market.historical.storage.exceptions import StorageError
from app.market.live.enums import CollectionMode
from app.market.live.models import CollectorConfig, CollectorMetricsSnapshot
from app.market.models import Quote
from app.providers.base import MarketDataProvider
from app.providers.exceptions import (
    AuthenticationError,
    NetworkError,
    ProviderError,
    RateLimitError,
)

logger = get_logger(__name__)

#: Async sleep signature, injectable so timing is controllable in tests.
Sleeper = Callable[[float], Awaitable[None]]


@dataclass
class _Metrics:
    """Mutable running metrics folded into a snapshot on demand."""

    requests_made: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    missing_updates: int = 0
    duplicate_ticks: int = 0
    out_of_order: int = 0
    stale_detections: int = 0
    retry_count: int = 0
    latency_total: float = 0.0
    latency_samples: int = 0
    last_successful_update: datetime | None = None
    started_at: datetime | None = None

    def record_latency(self, seconds: float) -> None:
        """Accumulate a latency sample."""
        self.latency_total += max(seconds, 0.0)
        self.latency_samples += 1

    def snapshot(self, uptime_seconds: float) -> CollectorMetricsSnapshot:
        """Return an immutable snapshot with the current uptime."""
        average = (
            self.latency_total / self.latency_samples if self.latency_samples else 0.0
        )
        return CollectorMetricsSnapshot(
            requests_made=self.requests_made,
            successful_updates=self.successful_updates,
            failed_updates=self.failed_updates,
            missing_updates=self.missing_updates,
            duplicate_ticks=self.duplicate_ticks,
            out_of_order=self.out_of_order,
            stale_detections=self.stale_detections,
            retry_count=self.retry_count,
            average_latency_seconds=average,
            last_successful_update=self.last_successful_update,
            uptime_seconds=uptime_seconds,
        )


@dataclass
class _SymbolState:
    """Per-symbol data-quality state."""

    last_timestamp: datetime | None = None
    last_price: Decimal | None = None
    last_change_at: datetime | None = None


class LiveMarketCollector:
    """Polls live market data and persists snapshots while the market is open."""

    def __init__(
        self,
        provider: MarketDataProvider,
        calendar: MarketCalendarService,
        clock: Clock,
        repository: CandleRepository,
        config: CollectorConfig,
        *,
        sleeper: Sleeper | None = None,
    ) -> None:
        """Initialise the collector.

        Args:
            provider: Live market-data source (interface only).
            calendar: Trading-calendar authority for open/closed decisions.
            clock: Time source.
            repository: Sink for persisted snapshot candles.
            config: Watchlist, interval, mode and timing configuration.
            sleeper: Async sleep used for scheduling (injectable for tests).
        """
        self._provider = provider
        self._calendar = calendar
        self._clock = clock
        self._repository = repository
        self._config = config
        self._sleeper = sleeper or asyncio.sleep
        self._metrics = _Metrics()
        self._states: dict[str, _SymbolState] = {
            symbol: _SymbolState() for symbol in config.watchlist
        }
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    # -- Lifecycle ---------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Return whether the collector loop is active."""
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the collection loop as a background task."""
        if self.is_running:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self.run())

    def request_stop(self) -> None:
        """Signal the loop to stop without awaiting it."""
        self._stop.set()

    async def stop(self) -> None:
        """Signal a graceful stop and await the loop's completion."""
        self.request_stop()
        if self._task is not None:
            await self._task
            self._task = None

    async def run(self) -> None:
        """Run the collection loop until stopped or cancelled."""
        if self._config.mode is CollectionMode.BACKTEST_REPLAY:
            raise NotImplementedError("Backtest replay mode is a stub.")
        self._stop.clear()
        self._metrics.started_at = self._clock.now()
        next_at = self._clock.now()
        try:
            while not self._stop.is_set():
                if not self._calendar.is_market_open(self._config.exchange):
                    await self._wait_for_next_session()
                    next_at = self._clock.now()
                    continue
                await self.poll_once()
                next_at = next_at + timedelta(
                    seconds=self._config.poll_interval_seconds
                )
                await self._sleep_until(next_at)
        except asyncio.CancelledError:
            logger.info("Live collector cancelled.")
            raise
        finally:
            logger.info("Live collector stopped.")

    # -- Polling -----------------------------------------------------------

    async def poll_once(self) -> None:
        """Run a single collection cycle over the watchlist."""
        if self._config.mode is CollectionMode.BACKTEST_REPLAY:
            raise NotImplementedError("Backtest replay mode is a stub.")
        quotes = await self._fetch()
        if quotes is None:
            self._metrics.failed_updates += 1
            return
        by_symbol = {quote.symbol.strip().upper(): quote for quote in quotes}
        updated = False
        for symbol in self._config.watchlist:
            quote = by_symbol.get(symbol)
            if quote is None:
                self._metrics.missing_updates += 1
                continue
            if await self._process(symbol, quote):
                updated = True
        if updated:
            self._metrics.last_successful_update = self._clock.now()

    async def _fetch(self) -> list[Quote] | None:
        """Fetch the watchlist quotes, retrying only retryable failures.

        Returns:
            The quotes, or ``None`` if the cycle failed after retries.

        Raises:
            AuthenticationError: Propagated immediately (non-recoverable).
        """
        attempt = 0
        while True:
            self._metrics.requests_made += 1
            started = self._clock.now()
            try:
                quotes = await self._provider.get_quotes(
                    list(self._config.watchlist), self._config.exchange
                )
            except AuthenticationError:
                logger.error("Authentication failure during collection.")
                raise
            except (NetworkError, RateLimitError) as exc:
                if attempt >= self._config.max_retries:
                    logger.error("Giving up cycle after %d retries: %s", attempt, exc)
                    return None
                self._metrics.retry_count += 1
                await self._sleeper(self._backoff(attempt, exc))
                attempt += 1
                continue
            except ProviderError as exc:
                logger.error("Non-retryable provider error: %s", exc)
                return None
            self._metrics.record_latency((self._clock.now() - started).total_seconds())
            return quotes

    async def _process(self, symbol: str, quote: Quote) -> bool:
        """Validate, deduplicate and persist a single quote.

        Returns:
            ``True`` if a snapshot was persisted.
        """
        state = self._states[symbol]
        timestamp = quote.timestamp
        if state.last_timestamp is not None:
            if timestamp == state.last_timestamp:
                self._metrics.duplicate_ticks += 1
                return False
            if timestamp < state.last_timestamp:
                self._metrics.out_of_order += 1
                logger.warning("Out-of-order tick for %s at %s.", symbol, timestamp)
                return False

        now = self._clock.now()
        if state.last_price is not None and quote.last_price == state.last_price:
            if (
                state.last_change_at is not None
                and (now - state.last_change_at).total_seconds()
                > self._config.stale_after_seconds
            ):
                self._metrics.stale_detections += 1
                logger.warning("Stale price for %s: %s.", symbol, quote.last_price)
        else:
            state.last_change_at = now
        if state.last_change_at is None:
            state.last_change_at = now

        if not await self._persist(symbol, quote):
            return False
        state.last_timestamp = timestamp
        state.last_price = quote.last_price
        self._metrics.successful_updates += 1
        return True

    async def _persist(self, symbol: str, quote: Quote) -> bool:
        """Persist a quote as a flat snapshot candle.

        Returns:
            ``True`` on success; ``False`` if storage failed.
        """
        price = quote.last_price
        candle = Candle(
            symbol=symbol,
            exchange=self._config.exchange,
            interval=self._config.interval,
            timestamp=quote.timestamp,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=quote.volume or 0,
        )
        try:
            await self._repository.save(candle)
        except StorageError as exc:
            self._metrics.failed_updates += 1
            logger.error("Failed to persist snapshot for %s: %s", symbol, exc)
            return False
        return True

    # -- Scheduling --------------------------------------------------------

    async def _sleep_until(self, deadline: datetime) -> None:
        """Sleep until ``deadline`` without accumulating drift."""
        delay = (deadline - self._clock.now()).total_seconds()
        await self._sleep(max(delay, 0.0))

    async def _wait_for_next_session(self) -> None:
        """Sleep until the next market open."""
        next_open = self._calendar.next_market_open(self._config.exchange)
        delay = (next_open - self._clock.now()).total_seconds()
        logger.info("Market closed; sleeping %.0fs until next session.", max(delay, 0))
        await self._sleep(max(delay, 0.0))

    async def _sleep(self, delay: float) -> None:
        """Sleep for ``delay`` seconds, interruptible by a stop request."""
        if delay <= 0:
            return
        stop_wait = asyncio.ensure_future(self._stop.wait())
        nap = asyncio.ensure_future(self._sleeper(delay))
        try:
            await asyncio.wait({stop_wait, nap}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for pending in (stop_wait, nap):
                if not pending.done():
                    pending.cancel()

    def _backoff(self, attempt: int, error: ProviderError) -> float:
        """Return the retry delay, honouring a rate-limit hint."""
        if isinstance(error, RateLimitError) and error.retry_after_seconds is not None:
            return min(
                float(error.retry_after_seconds), self._config.backoff_max_seconds
            )
        delay = self._config.backoff_base_seconds * float(2**attempt)
        return min(delay, self._config.backoff_max_seconds)

    # -- Metrics -----------------------------------------------------------

    def metrics(self) -> CollectorMetricsSnapshot:
        """Return a snapshot of the current metrics."""
        uptime = 0.0
        if self._metrics.started_at is not None:
            uptime = (self._clock.now() - self._metrics.started_at).total_seconds()
        return self._metrics.snapshot(max(uptime, 0.0))
