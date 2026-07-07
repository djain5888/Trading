"""Value objects for the market-regime engine."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.market.enums import Exchange


class Regime(StrEnum):
    """The directional state of the overall market."""

    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"


class Volatility(StrEnum):
    """A coarse band for market volatility."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class RegimeConfig(BaseModel):
    """Tunable parameters for the regime engine.

    Defaults describe a daily-timeframe classification of the NIFTY index.
    Periods are configurable so tests can exercise the logic on short series.
    """

    model_config = ConfigDict(frozen=True)

    index_symbol: str = Field(
        default="NIFTY", min_length=1, description="Benchmark index symbol."
    )
    index_exchange: Exchange = Field(
        default=Exchange.NSE, description="Exchange the index trades on."
    )
    trend_fast_period: int = Field(
        default=50, ge=1, description="Fast EMA period for the trend signal."
    )
    trend_slow_period: int = Field(
        default=200, ge=1, description="Slow EMA period for the trend signal."
    )
    slope_lookback: int = Field(
        default=20, ge=1, description="Bars back used to measure the fast-EMA slope."
    )
    breadth_period: int = Field(
        default=50, ge=1, description="EMA period each breadth member is compared to."
    )
    atr_period: int = Field(default=14, ge=1, description="ATR period for volatility.")
    volatility_lookback: int = Field(
        default=252, ge=2, description="Trailing ATR window for the percentile."
    )
    volatility_low_pct: float = Field(
        default=33.0, ge=0, le=100, description="Below this percentile is LOW."
    )
    volatility_high_pct: float = Field(
        default=66.0, ge=0, le=100, description="Above this percentile is HIGH."
    )
    breadth_bull_pct: float = Field(
        default=55.0, ge=0, le=100, description="Breadth above this votes bullish."
    )
    breadth_bear_pct: float = Field(
        default=45.0, ge=0, le=100, description="Breadth below this votes bearish."
    )
    trend_weight: float = Field(
        default=0.5, ge=0, le=1, description="Weight of the trend-structure signal."
    )
    slope_weight: float = Field(
        default=0.3, ge=0, le=1, description="Weight of the fast-EMA slope signal."
    )
    breadth_weight: float = Field(
        default=0.2, ge=0, le=1, description="Weight of the breadth signal."
    )
    regime_threshold: float = Field(
        default=0.3, gt=0, le=1, description="Score magnitude needed to leave SIDEWAYS."
    )
    history_days: int = Field(
        default=500, ge=1, description="Calendar days of history to read."
    )

    @model_validator(mode="after")
    def _validate(self) -> RegimeConfig:
        """Enforce coherent periods, thresholds and unit signal weights."""
        if self.trend_slow_period <= self.trend_fast_period:
            raise ValueError("trend_slow_period must exceed trend_fast_period.")
        if self.volatility_high_pct <= self.volatility_low_pct:
            raise ValueError("volatility_high_pct must exceed volatility_low_pct.")
        if self.breadth_bull_pct <= self.breadth_bear_pct:
            raise ValueError("breadth_bull_pct must exceed breadth_bear_pct.")
        total = self.trend_weight + self.slope_weight + self.breadth_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Signal weights must sum to 1.0.")
        return self

    @property
    def min_index_candles(self) -> int:
        """Minimum index candles needed to classify the trend and volatility."""
        return max(self.trend_slow_period, self.atr_period + 1)


class RegimeReport(BaseModel):
    """The market-regime engine's output.

    When ``available`` is ``False`` the index data was missing or too short and
    the remaining fields carry their neutral defaults; consumers should render
    "regime unavailable" rather than treat this as a signal.
    """

    model_config = ConfigDict(frozen=True)

    available: bool = Field(description="Whether a classification was produced.")
    regime: Regime | None = Field(
        default=None, description="Directional market state, if available."
    )
    confidence: float = Field(
        default=0.0, ge=0, le=100, description="Weighted agreement of the signals."
    )
    volatility: Volatility | None = Field(
        default=None, description="Volatility band, if available."
    )
    breadth: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Percent of watchlist above its breadth EMA.",
    )
    index_symbol: str = Field(description="Benchmark index the regime is based on.")
    detail: str = Field(description="Human-readable summary of the classification.")
    generated_at: datetime = Field(description="When the report was produced.")

    @classmethod
    def unavailable(
        cls, *, index_symbol: str, generated_at: datetime, detail: str
    ) -> RegimeReport:
        """Build a graceful "regime unavailable" report."""
        return cls(
            available=False,
            index_symbol=index_symbol,
            detail=detail,
            generated_at=generated_at,
        )
