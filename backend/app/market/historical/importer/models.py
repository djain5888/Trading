"""Value objects for the historical import engine."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.market.enums import Exchange, Interval


class ImportMode(StrEnum):
    """How much of the requested range to (re)import."""

    FULL = "full"
    """Import the whole requested range, skipping already-covered windows."""
    INCREMENTAL = "incremental"
    """Import only candles newer than the latest stored candle."""


class ImportConfig(BaseModel):
    """Tunable import behaviour."""

    model_config = ConfigDict(frozen=True)

    batch_size: int = Field(
        default=500, ge=1, description="Candles requested per provider call."
    )
    max_retries: int = Field(
        default=3, ge=0, description="Retries after the first attempt."
    )
    base_backoff_seconds: float = Field(
        default=0.5, gt=0, description="Initial exponential-backoff delay."
    )
    max_backoff_seconds: float = Field(
        default=30.0, gt=0, description="Maximum backoff delay."
    )


class ImportProgress(BaseModel):
    """Progress reported to the callback after each successful batch."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Symbol being imported.")
    exchange: Exchange = Field(description="Listing exchange.")
    interval: Interval = Field(description="Candle interval.")
    window_start: datetime = Field(description="Start of the completed window.")
    window_end: datetime = Field(description="End of the completed window.")
    completed_batches: int = Field(ge=0, description="Batches completed so far.")
    total_batches: int = Field(ge=0, description="Planned batches for this symbol.")
    candles_downloaded: int = Field(ge=0, description="Running downloaded count.")
    candles_imported: int = Field(ge=0, description="Running imported count.")

    @property
    def fraction(self) -> float:
        """Return completion in ``[0, 1]`` for this symbol."""
        if self.total_batches == 0:
            return 1.0
        return self.completed_batches / self.total_batches


class ImportSummary(BaseModel):
    """Aggregate outcome of an import run."""

    model_config = ConfigDict(frozen=True)

    symbols_processed: int = Field(default=0, ge=0)
    candles_downloaded: int = Field(default=0, ge=0)
    candles_imported: int = Field(default=0, ge=0)
    duplicates: int = Field(default=0, ge=0)
    gaps_detected: int = Field(default=0, ge=0)
    invalid_candles: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)
    elapsed_time: float = Field(default=0.0, ge=0, description="Wall time in seconds.")
