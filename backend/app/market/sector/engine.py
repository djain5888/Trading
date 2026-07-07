"""The sector-strength engine.

Ranks sectors by strength so every stock gets sector context. It maps each
watchlist symbol to a sector, aggregates member signals (price versus EMA,
momentum and breadth) into a 0-100 score, and derives a trend from the sector's
aggregate-EMA slope. All maths is delegated to the :class:`IndicatorEngine`.

Sectors with too few valid members are excluded rather than failed, and when no
sector metadata is configured the report degrades to ``available=False``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import fmean

from app.core.clock import Clock
from app.core.logging import get_logger
from app.indicators.engine import IndicatorEngine
from app.indicators.exceptions import IndicatorError
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.models import Candle, SeriesKey
from app.market.sector.models import (
    SectorConfig,
    SectorReport,
    SectorStrength,
    SectorTrend,
)

logger = get_logger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


@dataclass(frozen=True)
class _Member:
    """A single member stock's computed signals within a sector."""

    ema_series: list[float]
    above: bool
    price_gap: float
    momentum: float


class SectorStrengthEngine:
    """Ranks sectors by an aggregate of their member stocks."""

    def __init__(
        self,
        data_engine: HistoricalDataEngine,
        indicator_engine: IndicatorEngine,
        clock: Clock,
        config: SectorConfig | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            data_engine: Source of stored candles (read-only).
            indicator_engine: The single place indicator maths lives.
            clock: Time source for the report timestamp and default range.
            config: Metadata and tuning; an empty map degrades gracefully.
        """
        self._data = data_engine
        self._indicators = indicator_engine
        self._clock = clock
        self._config = config or SectorConfig()

    @property
    def config(self) -> SectorConfig:
        """Return the engine's configuration."""
        return self._config

    async def analyze(
        self,
        symbols: Sequence[str],
        *,
        exchange: Exchange = Exchange.NSE,
        interval: Interval = Interval.ONE_DAY,
        end: datetime | None = None,
    ) -> SectorReport:
        """Rank the sectors represented in ``symbols``.

        Args:
            symbols: Watchlist symbols to bucket into sectors.
            exchange: Exchange the symbols trade on.
            interval: Candle interval to read.
            end: Inclusive range end; defaults to the clock's now.

        Returns:
            A :class:`SectorReport`; ``available`` is ``False`` when no sector
            metadata is configured.
        """
        now = end or self._clock.now()
        if not self._config.sector_map:
            return SectorReport.unavailable(
                generated_at=now,
                detail="No sector metadata configured — sector strength unavailable.",
            )

        start = now - timedelta(days=self._config.history_days)
        buckets: dict[str, list[_Member]] = defaultdict(list)
        for symbol in symbols:
            sector = self._config.sector_map.get(symbol.strip().upper())
            if sector is None:
                continue
            member = await self._member(symbol, exchange, interval, start, now)
            if member is not None:
                buckets[sector].append(member)

        strengths = [
            self._score_sector(sector, members)
            for sector, members in buckets.items()
            if len(members) >= self._config.min_members
        ]
        ranked = tuple(sorted(strengths, key=lambda s: s.score, reverse=True))
        top = self._config.top_n
        return SectorReport(
            available=True,
            ranked=ranked,
            strongest=ranked[:top],
            weakest=tuple(reversed(ranked))[:top],
            detail=(
                f"{len(ranked)} sector(s) ranked."
                if ranked
                else "No sector had enough valid members to rank."
            ),
            generated_at=now,
        )

    # -- Per-member and per-sector aggregation -----------------------------

    async def _member(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> _Member | None:
        """Compute a member's signals, or ``None`` if its data is too thin."""
        key = SeriesKey(symbol=symbol, exchange=exchange, interval=interval)
        candles = await self._data.get_candles(key, start, end)
        if len(candles) < self._config.required_candles:
            return None
        try:
            ema = self._ema(candles, self._config.ema_period)
            momentum = self._roc(candles, self._config.momentum_period)
        except IndicatorError:
            return None
        if not ema or not momentum:
            return None
        close = float(candles[-1].close)
        ema_last = ema[-1]
        gap = (close - ema_last) / ema_last * 100 if ema_last else 0.0
        return _Member(
            ema_series=ema,
            above=close > ema_last,
            price_gap=gap,
            momentum=momentum[-1],
        )

    def _score_sector(self, sector: str, members: list[_Member]) -> SectorStrength:
        """Aggregate member signals into a 0-100 score and a trend."""
        above_pct = 100.0 * sum(1 for m in members if m.above) / len(members)
        breadth_score = above_pct
        price_score = _clamp(50.0 + fmean(m.price_gap for m in members), 0.0, 100.0)
        momentum_score = _clamp(50.0 + fmean(m.momentum for m in members), 0.0, 100.0)
        score = (
            self._config.breadth_weight * breadth_score
            + self._config.price_weight * price_score
            + self._config.momentum_weight * momentum_score
        )
        return SectorStrength(
            sector=sector,
            score=round(_clamp(score, 0.0, 100.0), 1),
            trend=self._sector_trend(members),
            members=len(members),
            above_pct=round(above_pct, 1),
        )

    def _sector_trend(self, members: list[_Member]) -> SectorTrend:
        """Classify the sector's trend from its aggregate-EMA slope.

        EMA is linear, so the mean of the members' EMA series is exactly the
        EMA of the mean price series — the sector's aggregate EMA.
        """
        series = [m.ema_series for m in members]
        length = min(len(s) for s in series)
        if length <= self._config.slope_lookback:
            return SectorTrend.FLAT
        aggregate = [
            fmean(s[len(s) - length + index] for s in series) for index in range(length)
        ]
        slope = aggregate[-1] - aggregate[-1 - self._config.slope_lookback]
        reference = aggregate[-1] or 1.0
        percent = slope / reference * 100
        if percent > self._config.trend_flat_pct:
            return SectorTrend.UP
        if percent < -self._config.trend_flat_pct:
            return SectorTrend.DOWN
        return SectorTrend.FLAT

    # -- Indicator helpers -------------------------------------------------

    def _ema(self, candles: Sequence[Candle], period: int) -> list[float]:
        """Return the EMA series values via the indicator engine."""
        return [
            result.value
            for result in self._indicators.compute_from_candles(
                candles, "ema", period=period
            )
        ]

    def _roc(self, candles: Sequence[Candle], period: int) -> list[float]:
        """Return the rate-of-change series values via the indicator engine."""
        return [
            result.value
            for result in self._indicators.compute_from_candles(
                candles, "roc", period=period
            )
        ]
