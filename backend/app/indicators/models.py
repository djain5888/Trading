"""Value objects for the indicator engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.indicators.exceptions import IndicatorError
from app.market.historical.models import Candle


class IndicatorResult(BaseModel):
    """A single indicator value at a point in time."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Indicator name, e.g. 'sma'.")
    timestamp: datetime = Field(description="Candle timestamp the value is for.")
    lookback: int = Field(ge=1, description="Candles required to produce the value.")
    value: float = Field(description="Primary indicator value.")
    metadata: dict[str, float] = Field(
        default_factory=dict,
        description="Auxiliary outputs/params (e.g. signal, upper, period).",
    )


@dataclass(frozen=True)
class PriceSeries:
    """Column-oriented, float view of a candle series for fast computation."""

    timestamps: tuple[datetime, ...]
    open: tuple[float, ...]
    high: tuple[float, ...]
    low: tuple[float, ...]
    close: tuple[float, ...]
    volume: tuple[float, ...]

    def __len__(self) -> int:
        """Return the number of candles."""
        return len(self.timestamps)

    @classmethod
    def from_candles(cls, candles: Sequence[Candle]) -> PriceSeries:
        """Build a price series from candles, validating ordering.

        Args:
            candles: Chronologically ordered candles.

        Returns:
            The column-oriented series.

        Raises:
            IndicatorError: If the series is empty or not strictly increasing.
        """
        if not candles:
            raise IndicatorError("Cannot build a price series from no candles.")
        previous: datetime | None = None
        for candle in candles:
            if previous is not None and candle.timestamp <= previous:
                raise IndicatorError(
                    "Candles must be strictly ordered with no duplicate timestamps."
                )
            previous = candle.timestamp
        return cls(
            timestamps=tuple(c.timestamp for c in candles),
            open=tuple(float(c.open) for c in candles),
            high=tuple(float(c.high) for c in candles),
            low=tuple(float(c.low) for c in candles),
            close=tuple(float(c.close) for c in candles),
            volume=tuple(float(c.volume) for c in candles),
        )
