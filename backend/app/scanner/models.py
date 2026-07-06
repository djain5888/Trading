"""Value objects for the scanner engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.clock import Clock
from app.indicators.engine import IndicatorEngine
from app.indicators.exceptions import IndicatorError
from app.indicators.models import IndicatorResult, PriceSeries
from app.market.historical.models import Candle, SeriesKey


class ScannerResult(BaseModel):
    """A single scanner's finding for one symbol.

    A result is a ranked, filtered *candidate* — never a buy/sell signal,
    target or stop.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="The symbol scanned.")
    scanner_name: str = Field(description="The scanner that produced this result.")
    score: float = Field(ge=0, le=100, description="Candidate strength, 0-100.")
    confidence: float = Field(ge=0, le=1, description="Confidence, 0-1.")
    timestamp: datetime = Field(description="Timestamp of the evaluated candle.")
    metadata: dict[str, float] = Field(
        default_factory=dict, description="Supporting numeric details."
    )


@dataclass(frozen=True)
class ScannerContext:
    """Everything a scanner needs to evaluate a single symbol.

    Scanners read candles and compute indicators through this context; they
    never fetch market data themselves.
    """

    key: SeriesKey
    candles: tuple[Candle, ...]
    series: PriceSeries
    indicator_engine: IndicatorEngine
    clock: Clock

    def compute(self, indicator: str, **params: float) -> list[IndicatorResult]:
        """Compute an indicator over this context's candles."""
        return self.indicator_engine.compute_from_candles(
            self.candles, indicator, **params
        )

    def safe_compute(self, indicator: str, **params: float) -> list[IndicatorResult]:
        """Compute an indicator, returning ``[]`` on insufficient history."""
        try:
            return self.compute(indicator, **params)
        except IndicatorError:
            return []

    @property
    def timestamp(self) -> datetime:
        """Return the last candle's timestamp."""
        return self.candles[-1].timestamp
