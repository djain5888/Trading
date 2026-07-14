"""The strategy engine.

Turns raw scanner hits and indicator structure into named setups. It consumes
the :class:`IndicatorEngine` (never reimplementing maths), the scanner hits, and
the regime/sector/RS reports as context for a confidence blend. It derives
entry/stop/target from ATR. It emits no buy/sell decision.

Missing context degrades to neutral confidence weights; a symbol without enough
history is excluded from the report rather than failing the run.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta

from app.core.clock import Clock
from app.core.logging import get_logger
from app.indicators.engine import IndicatorEngine
from app.indicators.exceptions import IndicatorError
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.models import Candle, SeriesKey
from app.market.regime.models import Regime, RegimeReport
from app.market.relative.models import RSReport
from app.market.sector.models import SectorReport
from app.scanner.models import ScannerResult
from app.strategy.base import Strategy, StrategyContext, StrategySignal, clamp
from app.strategy.models import StrategyConfig, StrategyReport, StrategySetup
from app.strategy.registry import StrategyRegistry, default_registry

logger = get_logger(__name__)

#: Neutral score used when a context component is unavailable.
_NEUTRAL = 50.0


def _regime_alignment(regime: RegimeReport | None) -> float:
    """Score how a bullish setup aligns with the regime, in ``[0, 100]``."""
    if regime is None or not regime.available or regime.regime is None:
        return _NEUTRAL
    if regime.regime is Regime.BULL:
        return _NEUTRAL + regime.confidence / 2.0
    if regime.regime is Regime.BEAR:
        return _NEUTRAL - regime.confidence / 2.0
    return _NEUTRAL


class StrategyEngine:
    """Names setups from scanner hits and market context."""

    def __init__(
        self,
        data_engine: HistoricalDataEngine,
        indicator_engine: IndicatorEngine,
        clock: Clock,
        config: StrategyConfig | None = None,
        registry: StrategyRegistry | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            data_engine: Source of stored candles (read-only).
            indicator_engine: The single place indicator maths lives.
            clock: Time source for the report timestamp and default range.
            config: Tuning, risk and blend weights.
            registry: Strategy registry; a default one is built if omitted.
        """
        self._data = data_engine
        self._indicators = indicator_engine
        self._clock = clock
        self._config = config or StrategyConfig()
        self._registry = registry or default_registry()
        self._strategies: list[Strategy] = self._registry.create_all()

    @property
    def config(self) -> StrategyConfig:
        """Return the engine's configuration."""
        return self._config

    @property
    def registry(self) -> StrategyRegistry:
        """Return the strategy registry."""
        return self._registry

    async def analyze(
        self,
        symbols: Sequence[str],
        *,
        exchange: Exchange = Exchange.NSE,
        interval: Interval = Interval.ONE_DAY,
        end: datetime | None = None,
        scan_results: Sequence[ScannerResult] = (),
        regime: RegimeReport | None = None,
        sectors: SectorReport | None = None,
        relative: RSReport | None = None,
    ) -> StrategyReport:
        """Name setups for ``symbols`` and rank them by blended confidence.

        Args:
            symbols: Symbols to evaluate.
            exchange: Exchange the symbols trade on.
            interval: Candle interval to read.
            end: Inclusive range end; defaults to the clock's now.
            scan_results: Scanner hits used to reinforce strategy signals.
            regime: Market-regime context (neutral when omitted).
            sectors: Sector-strength context (neutral when omitted).
            relative: Relative-strength context (neutral when omitted).

        Returns:
            A :class:`StrategyReport`; ``available`` is ``False`` only when no
            symbol had enough history to evaluate.
        """
        now = end or self._clock.now()
        start = now - timedelta(days=self._config.history_days)
        hits = self._hits_by_symbol(scan_results)
        sector_scores = self._sector_scores(sectors)
        by_symbol = relative.by_symbol if relative and relative.available else {}

        setups: list[StrategySetup] = []
        evaluated = 0
        for symbol in symbols:
            name = symbol.strip().upper()
            candles = await self._data.get_candles(
                SeriesKey(symbol=name, exchange=exchange, interval=interval), start, now
            )
            if len(candles) < self._config.min_candles:
                continue
            atr = self._latest_atr(candles)
            if atr is None or atr <= 0:
                continue
            evaluated += 1
            context = StrategyContext(
                symbol=name,
                candles=tuple(candles),
                indicator_engine=self._indicators,
                scanner_hits=frozenset(hits.get(name, set())),
                regime=regime,
                sector_strength=self._sector_strength(name, sector_scores),
                rs=by_symbol.get(name),
                config=self._config,
            )
            for strategy in self._strategies:
                signal = strategy.evaluate(context)
                if signal is not None:
                    setups.append(self._finalize(signal, atr, context))

        setups.sort(key=lambda setup: setup.confidence, reverse=True)
        if evaluated == 0:
            return StrategyReport.unavailable(
                generated_at=now,
                detail="No symbol had enough history — strategies unavailable.",
            )
        return StrategyReport(
            available=True,
            setups=tuple(setups),
            detail=f"{len(setups)} setup(s) across {evaluated} evaluated symbol(s).",
            generated_at=now,
        )

    # -- Finalisation ------------------------------------------------------

    def _finalize(
        self, signal: StrategySignal, atr: float, context: StrategyContext
    ) -> StrategySetup:
        """Derive stop/target from ATR and blend the confidence."""
        risk = self._config.stop_atr_mult * atr
        entry = signal.entry
        stop = entry - risk
        target = entry + self._config.reward_risk * risk
        reward_risk = (target - entry) / (entry - stop) if entry > stop else 0.0
        return StrategySetup(
            symbol=context.symbol,
            strategy=signal.strategy,
            confidence=self._blend(signal.signal_strength, context),
            signal_strength=round(signal.signal_strength, 1),
            entry=round(entry, 2),
            stop=round(stop, 2),
            target=round(target, 2),
            reward_risk=round(reward_risk, 2),
        )

    def _blend(self, signal_strength: float, context: StrategyContext) -> float:
        """Blend signal strength with regime, sector and RS context."""
        regime_score = _regime_alignment(context.regime)
        sector = (
            context.sector_strength if context.sector_strength is not None else _NEUTRAL
        )
        rs = context.rs.composite if context.rs is not None else _NEUTRAL
        blended = (
            self._config.signal_weight * signal_strength
            + self._config.regime_weight * regime_score
            + self._config.sector_weight * sector
            + self._config.rs_weight * rs
        )
        return round(clamp(blended), 1)

    # -- Context helpers ---------------------------------------------------

    @staticmethod
    def _hits_by_symbol(
        scan_results: Sequence[ScannerResult],
    ) -> dict[str, set[str]]:
        """Group scanner names by the symbol they fired on."""
        hits: dict[str, set[str]] = defaultdict(set)
        for result in scan_results:
            hits[result.symbol.strip().upper()].add(result.scanner_name)
        return hits

    @staticmethod
    def _sector_scores(sectors: SectorReport | None) -> dict[str, float]:
        """Map sector name -> strength score from a sector report."""
        if sectors is None or not sectors.available:
            return {}
        return {strength.sector: strength.score for strength in sectors.ranked}

    def _sector_strength(
        self, symbol: str, sector_scores: dict[str, float]
    ) -> float | None:
        """Return the symbol's sector strength score, or ``None`` if unknown."""
        sector = self._config.sector_map.get(symbol)
        if sector is None:
            return None
        return sector_scores.get(sector)

    def _latest_atr(self, candles: Sequence[Candle]) -> float | None:
        """Return the latest ATR value for a candle series, or ``None``."""
        try:
            results = self._indicators.compute_from_candles(
                candles, "atr", period=self._config.atr_period
            )
        except IndicatorError:
            return None
        return results[-1].value if results else None
