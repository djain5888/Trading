"""The strategy interface, per-symbol context and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from app.indicators.engine import IndicatorEngine
from app.indicators.exceptions import IndicatorError
from app.market.historical.models import Candle
from app.market.regime.models import RegimeReport
from app.market.relative.models import RelativeStrength
from app.strategy.models import StrategyConfig


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    """Clamp a value into ``[lower, upper]``."""
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class StrategySignal:
    """A strategy's raw finding: its name, signal strength and entry price.

    The engine turns this into a :class:`StrategySetup` by deriving the stop and
    target from ATR and blending the confidence with the market context.
    """

    strategy: str
    signal_strength: float
    entry: float


@dataclass(frozen=True)
class StrategyContext:
    """Everything a strategy needs to evaluate one symbol.

    Strategies read prices and compute indicators through this context; the
    market-context fields (regime, sector strength, RS) feed the confidence
    blend and are ``None`` when unavailable (treated as neutral).
    """

    symbol: str
    candles: tuple[Candle, ...]
    indicator_engine: IndicatorEngine
    scanner_hits: frozenset[str]
    regime: RegimeReport | None
    sector_strength: float | None
    rs: RelativeStrength | None
    config: StrategyConfig

    # -- Price access ------------------------------------------------------

    @property
    def opens(self) -> tuple[float, ...]:
        """Open prices as floats."""
        return tuple(float(candle.open) for candle in self.candles)

    @property
    def highs(self) -> tuple[float, ...]:
        """High prices as floats."""
        return tuple(float(candle.high) for candle in self.candles)

    @property
    def lows(self) -> tuple[float, ...]:
        """Low prices as floats."""
        return tuple(float(candle.low) for candle in self.candles)

    @property
    def closes(self) -> tuple[float, ...]:
        """Close prices as floats."""
        return tuple(float(candle.close) for candle in self.candles)

    @property
    def volumes(self) -> tuple[float, ...]:
        """Volumes as floats."""
        return tuple(float(candle.volume) for candle in self.candles)

    @property
    def last_close(self) -> float:
        """The latest close price."""
        return float(self.candles[-1].close)

    # -- Indicator access --------------------------------------------------

    def values(self, indicator: str, **params: float) -> list[float]:
        """Return an indicator's series values, or ``[]`` on short history."""
        try:
            return [
                result.value
                for result in self.indicator_engine.compute_from_candles(
                    self.candles, indicator, **params
                )
            ]
        except IndicatorError:
            return []

    def last(self, indicator: str, **params: float) -> float | None:
        """Return the latest value of an indicator, or ``None``."""
        series = self.values(indicator, **params)
        return series[-1] if series else None

    def macd(self, fast: int, slow: int, signal: int) -> tuple[float, float] | None:
        """Return the latest ``(macd_line, signal_line)``, or ``None``."""
        try:
            results = self.indicator_engine.compute_from_candles(
                self.candles, "macd", fast=fast, slow=slow, signal=signal
            )
        except IndicatorError:
            return None
        if not results:
            return None
        last = results[-1]
        return last.value, last.metadata.get("signal", 0.0)


class Strategy(ABC):
    """Names one kind of setup from a symbol's context.

    Strategies are independently pluggable: adding one means registering a new
    subclass. They describe *what* a setup is and never emit a buy/sell call.
    """

    name: ClassVar[str]
    #: Scanner names that reinforce this strategy's signal, if they fired.
    reinforcing_scanners: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def evaluate(self, context: StrategyContext) -> StrategySignal | None:
        """Return a signal when the setup is present, else ``None``."""

    def _scanner_bonus(self, context: StrategyContext, bonus: float = 10.0) -> float:
        """Return a small signal bonus when a reinforcing scanner fired."""
        return bonus if context.scanner_hits & self.reinforcing_scanners else 0.0
