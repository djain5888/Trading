"""Historical-data entities: the series key and the candle."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.market.enums import Exchange, Interval
from app.market.historical import invariants


class HistoricalModel(BaseModel):
    """Base class for immutable historical-data value objects."""

    model_config = ConfigDict(frozen=True)


class SeriesKey(HistoricalModel):
    """Identifies a single time-series: one symbol, exchange and interval."""

    symbol: str = Field(min_length=1, description="Trading symbol.")
    exchange: Exchange = Field(description="Listing exchange.")
    interval: Interval = Field(description="Candle interval.")

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        """Trim and upper-case the symbol for consistent identity."""
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("Symbol must not be blank.")
        return normalized


class Candle(HistoricalModel):
    """An immutable OHLCV candle for a single instrument and interval."""

    symbol: str = Field(min_length=1, description="Trading symbol.")
    exchange: Exchange = Field(description="Listing exchange.")
    interval: Interval = Field(description="Candle interval.")
    timestamp: datetime = Field(description="Bar open time (timezone-aware).")
    open: Decimal = Field(description="Open price.")
    high: Decimal = Field(description="High price.")
    low: Decimal = Field(description="Low price.")
    close: Decimal = Field(description="Close price.")
    volume: int = Field(ge=0, description="Traded volume.")
    open_interest: int | None = Field(
        default=None, ge=0, description="Open interest (derivatives only)."
    )

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        """Trim and upper-case the symbol for consistent identity."""
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("Symbol must not be blank.")
        return normalized

    @field_validator("timestamp")
    @classmethod
    def _require_aware(cls, value: datetime) -> datetime:
        """Reject naive timestamps."""
        error = invariants.check_timestamp_aware(value)
        if error is not None:
            raise ValueError(error)
        return value

    @model_validator(mode="after")
    def _validate_prices(self) -> Candle:
        """Validate price positivity and OHLC consistency.

        Raises:
            ValueError: If a price is non-positive or OHLC is inconsistent.
        """
        for check in (
            invariants.check_positive_prices,
            invariants.check_ohlc_consistency,
        ):
            error = check(self.open, self.high, self.low, self.close)
            if error is not None:
                raise ValueError(error)
        return self

    @property
    def series_key(self) -> SeriesKey:
        """Return the series this candle belongs to."""
        return SeriesKey(
            symbol=self.symbol, exchange=self.exchange, interval=self.interval
        )
