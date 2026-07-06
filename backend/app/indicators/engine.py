"""The indicator engine.

The single place in Titan where technical indicators are computed. It reads
candles from the :class:`HistoricalDataEngine`, delegates the maths to
registered calculators, caches identical computations, and supports incremental
extension for rolling-window indicators. It never fetches market data.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from app.core.clock import Clock
from app.core.logging import get_logger
from app.indicators.models import IndicatorResult, PriceSeries
from app.indicators.registry import IndicatorRegistry, default_registry
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.models import Candle, SeriesKey

logger = get_logger(__name__)

_Fingerprint = tuple[int, object, object, object, object]
_CacheKey = tuple[str, tuple[tuple[str, float], ...], _Fingerprint]


class IndicatorEngine:
    """Computes technical indicators from historical candles."""

    def __init__(
        self,
        data_engine: HistoricalDataEngine,
        clock: Clock,
        registry: IndicatorRegistry | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            data_engine: Source of historical candles (read-only).
            clock: Time source.
            registry: Indicator registry; a default one is built if omitted.
        """
        self._data = data_engine
        self._clock = clock
        self._registry = registry or default_registry()
        self._cache: dict[_CacheKey, list[IndicatorResult]] = {}

    @property
    def registry(self) -> IndicatorRegistry:
        """Return the indicator registry."""
        return self._registry

    async def compute(
        self,
        key: SeriesKey,
        indicator: str,
        start: datetime,
        end: datetime,
        **params: float,
    ) -> list[IndicatorResult]:
        """Compute an indicator over a stored candle range.

        Args:
            key: The series to read candles for.
            indicator: The registered indicator name.
            start: Inclusive range start.
            end: Inclusive range end.
            **params: Indicator parameters.

        Returns:
            The indicator series.
        """
        candles = await self._data.get_candles(key, start, end)
        return self.compute_from_candles(candles, indicator, **params)

    def compute_from_candles(
        self, candles: Sequence[Candle], indicator: str, **params: float
    ) -> list[IndicatorResult]:
        """Compute an indicator from a candle sequence, using the cache.

        Args:
            candles: Chronologically ordered candles.
            indicator: The registered indicator name.
            **params: Indicator parameters.

        Returns:
            The indicator series (a cached list on repeat calls).
        """
        cache_key = self._cache_key(indicator, params, candles)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(
                "Indicator cache hit for %s at %s", indicator, self._clock.now()
            )
            return cached
        calculator = self._registry.create(indicator, **params)
        results = calculator.calculate(PriceSeries.from_candles(candles))
        self._cache[cache_key] = results
        return results

    def compute_incremental(
        self,
        candles: Sequence[Candle],
        indicator: str,
        previous: list[IndicatorResult],
        **params: float,
    ) -> list[IndicatorResult]:
        """Extend a previous result with only the new candles' values.

        For rolling-window indicators only the newly appended points are
        computed; recursive indicators fall back to a full (cached) compute.

        Args:
            candles: The full, extended candle sequence.
            indicator: The registered indicator name.
            previous: Results previously computed for a prefix of ``candles``.
            **params: Indicator parameters.

        Returns:
            The full indicator series.
        """
        calculator = self._registry.create(indicator, **params)
        if not calculator.supports_incremental or not previous:
            return self.compute_from_candles(candles, indicator, **params)

        expected_total = len(candles) - calculator.lookback + 1
        if expected_total <= len(previous):
            return previous

        tail_candles = candles[len(previous) :]
        tail = calculator.calculate(PriceSeries.from_candles(tail_candles))
        results = [*previous, *tail]
        self._cache[self._cache_key(indicator, params, candles)] = results
        return results

    @staticmethod
    def _cache_key(
        indicator: str, params: dict[str, float], candles: Sequence[Candle]
    ) -> _CacheKey:
        """Build a cache key from the indicator, params and a data fingerprint."""
        param_key = tuple(sorted(params.items()))
        fingerprint: _Fingerprint = (
            len(candles),
            candles[0].timestamp if candles else None,
            candles[-1].timestamp if candles else None,
            candles[0].close if candles else None,
            candles[-1].close if candles else None,
        )
        return indicator, param_key, fingerprint
