"""The market-regime engine.

Classifies the overall market so every scan result gets context. It reads the
benchmark index and the stored watchlist through the :class:`HistoricalDataEngine`
and delegates every calculation to the :class:`IndicatorEngine` — it never
reimplements indicator maths. If index data is missing it degrades gracefully to
a "regime unavailable" report instead of raising.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from app.core.clock import Clock
from app.core.logging import get_logger
from app.indicators.engine import IndicatorEngine
from app.indicators.exceptions import IndicatorError
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.models import Candle, SeriesKey
from app.market.regime.models import Regime, RegimeConfig, RegimeReport, Volatility

logger = get_logger(__name__)


def _sign(value: float) -> int:
    """Return the sign of ``value`` as ``-1``, ``0`` or ``1``."""
    return (value > 0) - (value < 0)


class MarketRegimeEngine:
    """Classifies the market into a trend, volatility band and breadth."""

    def __init__(
        self,
        data_engine: HistoricalDataEngine,
        indicator_engine: IndicatorEngine,
        clock: Clock,
        config: RegimeConfig | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            data_engine: Source of stored candles (read-only).
            indicator_engine: The single place indicator maths lives.
            clock: Time source for the report timestamp and default range.
            config: Tuning parameters; sensible NIFTY/daily defaults if omitted.
        """
        self._data = data_engine
        self._indicators = indicator_engine
        self._clock = clock
        self._config = config or RegimeConfig()

    @property
    def config(self) -> RegimeConfig:
        """Return the engine's configuration."""
        return self._config

    async def analyze(
        self,
        symbols: Sequence[str],
        *,
        exchange: Exchange = Exchange.NSE,
        interval: Interval = Interval.ONE_DAY,
        end: datetime | None = None,
    ) -> RegimeReport:
        """Classify the market from the index and the watchlist.

        Args:
            symbols: Watchlist symbols used to compute breadth.
            exchange: Exchange the watchlist trades on.
            interval: Candle interval to read for the index and members.
            end: Inclusive range end; defaults to the clock's now.

        Returns:
            A :class:`RegimeReport`; ``available`` is ``False`` when the index
            has too little data to classify.
        """
        now = end or self._clock.now()
        start = now - timedelta(days=self._config.history_days)
        index_key = SeriesKey(
            symbol=self._config.index_symbol,
            exchange=self._config.index_exchange,
            interval=interval,
        )
        index_candles = await self._data.get_candles(index_key, start, now)
        if len(index_candles) < self._config.min_index_candles:
            return self._unavailable(now)

        try:
            trend_vote = self._trend_vote(index_candles)
            slope_vote = self._slope_vote(index_candles)
            volatility = self._volatility(index_candles)
        except IndicatorError as exc:
            logger.warning("Regime classification failed on index maths: %s", exc)
            return self._unavailable(now)

        breadth = await self._breadth(symbols, exchange, interval, start, now)
        breadth_vote = self._breadth_vote(breadth)
        regime, confidence = self._classify(trend_vote, slope_vote, breadth_vote)
        return RegimeReport(
            available=True,
            regime=regime,
            confidence=confidence,
            volatility=volatility,
            breadth=breadth,
            index_symbol=self._config.index_symbol,
            detail=(
                f"{self._config.index_symbol} {regime.value.lower()} "
                f"(confidence {confidence:.0f}), "
                f"volatility {volatility.value.lower()}, "
                f"breadth {breadth:.0f}%."
            ),
            generated_at=now,
        )

    # -- Signal votes ------------------------------------------------------

    def _trend_vote(self, candles: Sequence[Candle]) -> int:
        """Vote on trend structure: price versus the fast and slow EMAs."""
        price = float(candles[-1].close)
        fast = self._ema(candles, self._config.trend_fast_period)[-1]
        slow = self._ema(candles, self._config.trend_slow_period)[-1]
        if price > fast > slow:
            return 1
        if price < fast < slow:
            return -1
        return 0

    def _slope_vote(self, candles: Sequence[Candle]) -> int:
        """Vote on the fast EMA's slope over the configured lookback."""
        fast = self._ema(candles, self._config.trend_fast_period)
        if len(fast) <= self._config.slope_lookback:
            return 0
        change = fast[-1] - fast[-1 - self._config.slope_lookback]
        if change > 0:
            return 1
        if change < 0:
            return -1
        return 0

    def _breadth_vote(self, breadth: float) -> int:
        """Vote on market breadth against the bull/bear thresholds."""
        if breadth > self._config.breadth_bull_pct:
            return 1
        if breadth < self._config.breadth_bear_pct:
            return -1
        return 0

    # -- Aggregation -------------------------------------------------------

    def _classify(
        self, trend_vote: int, slope_vote: int, breadth_vote: int
    ) -> tuple[Regime, float]:
        """Combine weighted votes into a regime and its confidence.

        The regime is the sign of the weighted score once it clears the
        threshold; confidence is the summed weight of the signals that agree
        with the chosen regime (for SIDEWAYS, the weight of the neutral ones).
        """
        weights = (
            (trend_vote, self._config.trend_weight),
            (slope_vote, self._config.slope_weight),
            (breadth_vote, self._config.breadth_weight),
        )
        score = sum(vote * weight for vote, weight in weights)
        if score >= self._config.regime_threshold:
            regime, sign = Regime.BULL, 1
        elif score <= -self._config.regime_threshold:
            regime, sign = Regime.BEAR, -1
        else:
            regime, sign = Regime.SIDEWAYS, 0
        agreement = sum(weight for vote, weight in weights if _sign(vote) == sign)
        return regime, round(min(max(agreement, 0.0), 1.0) * 100, 1)

    # -- Volatility --------------------------------------------------------

    def _volatility(self, candles: Sequence[Candle]) -> Volatility:
        """Band the latest ATR by its percentile in the trailing window."""
        atr = [
            result.value
            for result in self._indicators.compute_from_candles(
                candles, "atr", period=self._config.atr_period
            )
        ]
        window = atr[-self._config.volatility_lookback :]
        current = window[-1]
        rank = sum(1 for value in window if value <= current)
        percentile = 100.0 * rank / len(window)
        if percentile < self._config.volatility_low_pct:
            return Volatility.LOW
        if percentile > self._config.volatility_high_pct:
            return Volatility.HIGH
        return Volatility.NORMAL

    # -- Breadth -----------------------------------------------------------

    async def _breadth(
        self,
        symbols: Sequence[str],
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> float:
        """Return the percent of members trading above their breadth EMA."""
        above = 0
        counted = 0
        for symbol in symbols:
            key = SeriesKey(symbol=symbol, exchange=exchange, interval=interval)
            candles = await self._data.get_candles(key, start, end)
            if len(candles) < self._config.breadth_period:
                continue
            try:
                ema = self._ema(candles, self._config.breadth_period)
            except IndicatorError:
                continue
            counted += 1
            if float(candles[-1].close) > ema[-1]:
                above += 1
        if counted == 0:
            return 0.0
        return round(100.0 * above / counted, 1)

    # -- Helpers -----------------------------------------------------------

    def _ema(self, candles: Sequence[Candle], period: int) -> list[float]:
        """Return the EMA series values for ``period`` via the indicator engine."""
        return [
            result.value
            for result in self._indicators.compute_from_candles(
                candles, "ema", period=period
            )
        ]

    def _unavailable(self, now: datetime) -> RegimeReport:
        """Build the graceful "regime unavailable" report."""
        return RegimeReport.unavailable(
            index_symbol=self._config.index_symbol,
            generated_at=now,
            detail=(
                f"{self._config.index_symbol} index data unavailable — "
                "regime unavailable."
            ),
        )
