"""The historical import engine.

Imports historical candles from any :class:`MarketDataProvider` into the
:class:`HistoricalDataEngine`. It plans the missing time windows, fetches each
with bounded retries and exponential backoff, maps provider bars into validated
domain candles, and persists them through the data engine (which never bypasses
validation). It depends only on interfaces — never on a concrete provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import ValidationError

from app.core.clock import Clock
from app.core.logging import get_logger
from app.market.calendar.service import MarketCalendarService
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.importer.exceptions import OutOfWindowDataError
from app.market.historical.importer.models import (
    ImportConfig,
    ImportMode,
    ImportProgress,
    ImportSummary,
)
from app.market.historical.intervals import interval_delta
from app.market.historical.models import Candle, SeriesKey
from app.market.historical.storage.exceptions import StorageError
from app.market.models import HistoricalData
from app.providers.base import MarketDataProvider
from app.providers.exceptions import (
    AuthenticationError,
    NetworkError,
    ProviderError,
    RateLimitError,
)

logger = get_logger(__name__)

#: Progress callback signature; invoked after every successful batch.
ProgressCallback = Callable[[ImportProgress], None]

#: Async sleep signature, injectable so backoff is instant in tests.
Sleeper = Callable[[float], Awaitable[None]]

#: Guard against unbounded trading-day scans for a single window.
_MAX_WINDOW_DAY_SCAN = 4000


@dataclass
class _SummaryAccumulator:
    """Mutable running totals folded into an :class:`ImportSummary`."""

    symbols_processed: int = 0
    candles_downloaded: int = 0
    candles_imported: int = 0
    duplicates: int = 0
    gaps_detected: int = 0
    invalid_candles: int = 0
    retries: int = 0
    failed_requests: int = 0

    def build(self, elapsed_time: float) -> ImportSummary:
        """Return an immutable summary with the elapsed wall time."""
        return ImportSummary(
            symbols_processed=self.symbols_processed,
            candles_downloaded=self.candles_downloaded,
            candles_imported=self.candles_imported,
            duplicates=self.duplicates,
            gaps_detected=self.gaps_detected,
            invalid_candles=self.invalid_candles,
            retries=self.retries,
            failed_requests=self.failed_requests,
            elapsed_time=elapsed_time,
        )


class HistoricalImportEngine:
    """Imports historical data from a provider into the data engine."""

    def __init__(
        self,
        provider: MarketDataProvider,
        data_engine: HistoricalDataEngine,
        calendar: MarketCalendarService,
        clock: Clock,
        *,
        config: ImportConfig | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        """Initialise the import engine.

        Args:
            provider: Source of historical data (interface only).
            data_engine: Validating store for candles.
            calendar: Trading-calendar authority.
            clock: Time source for elapsed-time measurement.
            config: Batch/retry/backoff configuration.
            sleeper: Async sleep used for backoff (injectable for tests).
        """
        self._provider = provider
        self._data_engine = data_engine
        self._calendar = calendar
        self._clock = clock
        self._config = config or ImportConfig()
        self._sleeper = sleeper or asyncio.sleep

    async def import_history(
        self,
        symbols: Sequence[str],
        interval: Interval,
        start: datetime,
        end: datetime,
        exchange: Exchange = Exchange.NSE,
        *,
        mode: ImportMode = ImportMode.FULL,
        on_progress: ProgressCallback | None = None,
    ) -> ImportSummary:
        """Import historical candles for one or more symbols.

        Args:
            symbols: Symbols to import.
            interval: Candle interval.
            start: Inclusive range start (timezone-aware).
            end: Inclusive range end (timezone-aware).
            exchange: Listing exchange.
            mode: Full or incremental import.
            on_progress: Optional per-batch progress callback.

        Returns:
            An :class:`ImportSummary` of the run.

        Raises:
            ValueError: If ``end`` is not after ``start``.
            AuthenticationError: If the provider rejects credentials.
        """
        if end < start:
            raise ValueError("'end' must not be before 'start'.")

        totals = _SummaryAccumulator()
        started_at = self._clock.now()
        for symbol in self._unique_symbols(symbols):
            await self._import_symbol(
                symbol, interval, start, end, exchange, mode, on_progress, totals
            )
            totals.symbols_processed += 1
        elapsed = (self._clock.now() - started_at).total_seconds()
        return totals.build(max(elapsed, 0.0))

    # -- Per-symbol orchestration -----------------------------------------

    async def _import_symbol(
        self,
        symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
        exchange: Exchange,
        mode: ImportMode,
        on_progress: ProgressCallback | None,
        totals: _SummaryAccumulator,
    ) -> None:
        """Import a single symbol's range, window by window."""
        key = SeriesKey(symbol=symbol, exchange=exchange, interval=interval)
        latest = await self._data_engine.get_latest(key)
        latest_ts = latest.timestamp if latest is not None else None

        effective_start = self._effective_start(start, interval, mode, latest_ts)
        if effective_start > end:
            logger.info("Nothing to import for %s: already up to date.", symbol)
            return

        windows = self._plan_windows(effective_start, end, interval)
        for index, (window_start, window_end) in enumerate(windows, start=1):
            if await self._is_window_covered(
                key, window_start, window_end, mode, latest_ts
            ):
                continue
            if not self._has_trading_day(window_start, window_end, exchange):
                continue

            bars = await self._fetch_with_retry(
                symbol, interval, window_start, window_end, exchange, totals
            )
            if bars is None:
                continue

            try:
                self._verify_window(symbol, bars, window_start, window_end)
            except OutOfWindowDataError as exc:
                # REJECT the batch: never store or relabel out-of-window data.
                totals.failed_requests += 1
                logger.error("%s", exc)
                continue

            totals.candles_downloaded += len(bars.candles)
            await self._store_bars(bars, key, totals)

            if on_progress is not None:
                on_progress(
                    ImportProgress(
                        symbol=symbol,
                        exchange=exchange,
                        interval=interval,
                        window_start=window_start,
                        window_end=window_end,
                        completed_batches=index,
                        total_batches=len(windows),
                        candles_downloaded=totals.candles_downloaded,
                        candles_imported=totals.candles_imported,
                    )
                )

    @staticmethod
    def _verify_window(
        symbol: str,
        bars: HistoricalData,
        window_start: datetime,
        window_end: datetime,
    ) -> None:
        """Reject a batch whose candles fall outside the requested window.

        Raises:
            OutOfWindowDataError: If any returned candle is dated before
                ``window_start`` or after ``window_end``. A provider that serves
                the wrong date range is never trusted to relabel it as history.
        """
        if not bars.candles:
            return
        low, high = window_start.date(), window_end.date()
        outside = [c for c in bars.candles if not (low <= c.timestamp.date() <= high)]
        if outside:
            received_low = min(c.timestamp.date() for c in bars.candles)
            received_high = max(c.timestamp.date() for c in bars.candles)
            raise OutOfWindowDataError(
                f"Out-of-window data for {symbol}: requested {low}..{high} but the "
                f"provider returned {received_low}..{received_high} "
                f"({len(outside)}/{len(bars.candles)} candles outside the window). "
                "Rejecting the batch — recent data must never be relabelled as "
                "older history."
            )

    async def _store_bars(
        self, bars: HistoricalData, key: SeriesKey, totals: _SummaryAccumulator
    ) -> None:
        """Map provider bars to validated candles and persist them."""
        candles, invalid = self._to_candles(bars, key)
        totals.invalid_candles += invalid
        if not candles:
            return
        try:
            report = await self._data_engine.import_candles(candles)
        except StorageError as exc:
            totals.failed_requests += 1
            logger.error("Storage error importing %s: %s", key.symbol, exc)
            return
        totals.candles_imported += report.imported
        totals.duplicates += report.duplicates
        totals.gaps_detected += report.gaps_found
        totals.invalid_candles += report.invalid

    # -- Fetching with retry ----------------------------------------------

    async def _fetch_with_retry(
        self,
        symbol: str,
        interval: Interval,
        window_start: datetime,
        window_end: datetime,
        exchange: Exchange,
        totals: _SummaryAccumulator,
    ) -> HistoricalData | None:
        """Fetch one window, retrying only retryable failures.

        Returns:
            The provider payload, or ``None`` if the window ultimately failed.

        Raises:
            AuthenticationError: Propagated immediately (non-retryable, fatal).
        """
        attempt = 0
        while True:
            try:
                return await self._provider.get_historical_data(
                    symbol, interval, window_start, window_end, exchange
                )
            except AuthenticationError:
                totals.failed_requests += 1
                logger.error("Authentication failed while importing %s.", symbol)
                raise
            except (NetworkError, RateLimitError) as exc:
                if attempt >= self._config.max_retries:
                    totals.failed_requests += 1
                    logger.error(
                        "Giving up on %s window after %d retries: %s",
                        symbol,
                        attempt,
                        exc,
                    )
                    return None
                totals.retries += 1
                await self._sleeper(self._backoff_delay(attempt, exc))
                attempt += 1
            except ProviderError as exc:
                totals.failed_requests += 1
                logger.error(
                    "Provider error importing %s [%s..%s]: %s",
                    symbol,
                    window_start.date(),
                    window_end.date(),
                    exc.details or exc,
                )
                return None

    def _backoff_delay(self, attempt: int, error: ProviderError) -> float:
        """Return the delay before the next attempt.

        Honours a rate-limit ``retry_after`` hint when present, otherwise uses
        capped exponential backoff.
        """
        if isinstance(error, RateLimitError) and error.retry_after_seconds is not None:
            return min(
                float(error.retry_after_seconds), self._config.max_backoff_seconds
            )
        delay = self._config.base_backoff_seconds * float(2**attempt)
        return min(delay, self._config.max_backoff_seconds)

    # -- Planning helpers --------------------------------------------------

    @staticmethod
    def _effective_start(
        start: datetime,
        interval: Interval,
        mode: ImportMode,
        latest_ts: datetime | None,
    ) -> datetime:
        """Return the first timestamp to import, honouring incremental mode."""
        if mode is not ImportMode.INCREMENTAL or latest_ts is None:
            return start
        delta = interval_delta(interval) or timedelta(0)
        return max(start, latest_ts + delta)

    def _plan_windows(
        self, start: datetime, end: datetime, interval: Interval
    ) -> list[tuple[datetime, datetime]]:
        """Split ``[start, end]`` into inclusive, batch-sized windows."""
        delta = interval_delta(interval)
        if delta is None:
            return [(start, end)]
        span = delta * (self._config.batch_size - 1)
        windows: list[tuple[datetime, datetime]] = []
        cursor = start
        while cursor <= end:
            window_end = min(cursor + span, end)
            windows.append((cursor, window_end))
            cursor = window_end + delta
        # Fold a lone trailing point (e.g. today's not-yet-formed candle left by
        # an off-by-one batch split) into its predecessor, so it is not fetched
        # as a separate, redundant provider call — the second call that failed on
        # live runs after the first window had already stored the data.
        if len(windows) >= 2 and windows[-1][0] == windows[-1][1]:
            windows[-2:] = [(windows[-2][0], windows[-1][1])]
        return windows

    async def _is_window_covered(
        self,
        key: SeriesKey,
        window_start: datetime,
        window_end: datetime,
        mode: ImportMode,
        latest_ts: datetime | None,
    ) -> bool:
        """Return whether a window is already imported and can be skipped.

        A window is skippable only in full-import mode when it lies entirely at
        or before the latest stored candle and already holds data — which is
        exactly the resume case for a previously interrupted import.
        """
        if mode is not ImportMode.FULL or latest_ts is None:
            return False
        if window_end > latest_ts:
            return False
        existing = await self._data_engine.get_candles(key, window_start, window_end)
        return len(existing) > 0

    def _has_trading_day(
        self, window_start: datetime, window_end: datetime, exchange: Exchange
    ) -> bool:
        """Return whether the window spans at least one trading day."""
        day = window_start.date()
        last_day = window_end.date()
        for _ in range(_MAX_WINDOW_DAY_SCAN):
            if day > last_day:
                break
            if self._calendar.is_trading_day(day, exchange):
                return True
            day += timedelta(days=1)
        return False

    def _to_candles(
        self, bars: HistoricalData, key: SeriesKey
    ) -> tuple[list[Candle], int]:
        """Map provider bars into validated domain candles.

        Invalid bars are counted and skipped; validation is never bypassed.

        Returns:
            A tuple of the valid candles and the count of invalid bars.
        """
        candles: list[Candle] = []
        invalid = 0
        for bar in bars.candles:
            try:
                candles.append(
                    Candle(
                        symbol=key.symbol,
                        exchange=key.exchange,
                        interval=key.interval,
                        timestamp=bar.timestamp,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                    )
                )
            except ValidationError:
                invalid += 1
        return candles, invalid

    @staticmethod
    def _unique_symbols(symbols: Sequence[str]) -> list[str]:
        """Return normalised, de-duplicated symbols preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for raw in symbols:
            symbol = raw.strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                result.append(symbol)
        return result
