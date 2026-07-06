"""The scanner engine.

Runs scanners over many symbols concurrently, caches per-symbol results, and
merges/ranks/deduplicates them. It discovers candidates only — it never emits
trading decisions. It reads candles from the historical data engine and
computes indicators through the indicator engine; it never touches a provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime

from app.core.clock import Clock
from app.core.logging import get_logger
from app.indicators.engine import IndicatorEngine
from app.indicators.models import PriceSeries
from app.market.calendar.service import MarketCalendarService
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.models import Candle, SeriesKey
from app.scanner.base import Scanner
from app.scanner.models import ScannerContext, ScannerResult
from app.scanner.registry import ScannerRegistry, default_registry

logger = get_logger(__name__)

_Fingerprint = tuple[int, object, object]
_CacheKey = tuple[str, tuple[tuple[str, float], ...], SeriesKey, _Fingerprint]


class ScannerEngine:
    """Orchestrates scanners across a universe of symbols."""

    def __init__(
        self,
        data_engine: HistoricalDataEngine,
        indicator_engine: IndicatorEngine,
        calendar: MarketCalendarService,
        clock: Clock,
        registry: ScannerRegistry | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            data_engine: Source of historical candles (read-only).
            indicator_engine: Computes indicators for scanners.
            calendar: Trading-calendar authority.
            clock: Time source.
            registry: Scanner registry; a default one is built if omitted.
        """
        self._data = data_engine
        self._indicators = indicator_engine
        self._calendar = calendar
        self._clock = clock
        self._registry = registry or default_registry()
        self._cache: dict[_CacheKey, ScannerResult | None] = {}

    @property
    def registry(self) -> ScannerRegistry:
        """Return the scanner registry."""
        return self._registry

    # -- Public API --------------------------------------------------------

    async def run(
        self,
        scanner_name: str,
        keys: Sequence[SeriesKey],
        start: datetime,
        end: datetime,
        *,
        dedupe: bool = True,
        **params: float,
    ) -> list[ScannerResult]:
        """Run a single named scanner over many symbols."""
        scanner = self._registry.create(scanner_name, **params)
        return await self.scan([scanner], keys, start, end, dedupe=dedupe)

    async def run_all(
        self,
        keys: Sequence[SeriesKey],
        start: datetime,
        end: datetime,
        *,
        dedupe: bool = True,
    ) -> list[ScannerResult]:
        """Run every registered scanner (with defaults) over many symbols."""
        return await self.scan(
            self._registry.create_all(), keys, start, end, dedupe=dedupe
        )

    async def scan(
        self,
        scanners: Sequence[Scanner],
        keys: Sequence[SeriesKey],
        start: datetime,
        end: datetime,
        *,
        dedupe: bool = True,
    ) -> list[ScannerResult]:
        """Run scanners over symbols concurrently and rank the results.

        Args:
            scanners: The scanner instances to apply.
            keys: The series (symbol/exchange/interval) to scan.
            start: Inclusive candle range start.
            end: Inclusive candle range end.
            dedupe: Whether to keep only the best result per symbol.

        Returns:
            Results ranked by score (then confidence), highest first.
        """
        batches = await asyncio.gather(
            *(self._scan_key(scanners, key, start, end) for key in keys)
        )
        merged = [result for batch in batches for result in batch]
        return self.rank(merged, dedupe=dedupe)

    @staticmethod
    def rank(
        results: Sequence[ScannerResult], *, dedupe: bool = True
    ) -> list[ScannerResult]:
        """Rank results by score then confidence, optionally deduping symbols.

        Args:
            results: Results to rank.
            dedupe: If true, keep only the highest-scoring result per symbol.

        Returns:
            The ranked results.
        """
        ordered = sorted(results, key=lambda r: (r.score, r.confidence), reverse=True)
        if not dedupe:
            return ordered
        seen: set[str] = set()
        deduped: list[ScannerResult] = []
        for result in ordered:
            if result.symbol in seen:
                continue
            seen.add(result.symbol)
            deduped.append(result)
        return deduped

    # -- Internals ---------------------------------------------------------

    async def _scan_key(
        self,
        scanners: Sequence[Scanner],
        key: SeriesKey,
        start: datetime,
        end: datetime,
    ) -> list[ScannerResult]:
        """Fetch candles for one symbol and apply every scanner."""
        candles = await self._data.get_candles(key, start, end)
        if not candles:
            return []
        series = PriceSeries.from_candles(candles)
        context = ScannerContext(
            key=key,
            candles=tuple(candles),
            series=series,
            indicator_engine=self._indicators,
            clock=self._clock,
        )
        fingerprint = self._fingerprint(candles)
        results: list[ScannerResult] = []
        for scanner in scanners:
            result = self._cached_scan(scanner, context, key, fingerprint)
            if result is not None:
                results.append(result)
        return results

    def _cached_scan(
        self,
        scanner: Scanner,
        context: ScannerContext,
        key: SeriesKey,
        fingerprint: _Fingerprint,
    ) -> ScannerResult | None:
        """Return a scanner's result for a symbol, using the cache."""
        cache_key: _CacheKey = (
            scanner.name,
            scanner.config_key(),
            key,
            fingerprint,
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = scanner.scan(context)
        self._cache[cache_key] = result
        return result

    @staticmethod
    def _fingerprint(candles: Sequence[Candle]) -> _Fingerprint:
        """Build a fingerprint identifying a candle series."""
        return (len(candles), candles[0].timestamp, candles[-1].close)
