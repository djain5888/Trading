"""Configuration and metrics value objects for the live collector."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.market.enums import Exchange, Interval
from app.market.live.enums import CollectionMode


class CollectorConfig(BaseModel):
    """Immutable configuration for a :class:`LiveMarketCollector`."""

    model_config = ConfigDict(frozen=True)

    watchlist: tuple[str, ...] = Field(
        min_length=1, description="Symbols to poll each cycle."
    )
    exchange: Exchange = Field(default=Exchange.NSE, description="Listing exchange.")
    interval: Interval = Field(
        default=Interval.ONE_MINUTE, description="Interval tag for stored snapshots."
    )
    mode: CollectionMode = Field(
        default=CollectionMode.LIVE, description="Collection mode."
    )
    poll_interval_seconds: float = Field(
        default=5.0, gt=0, description="Seconds between poll cycles."
    )
    stale_after_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Flag a price as stale after this long unchanged.",
    )
    max_retries: int = Field(
        default=3, ge=0, description="Retries after the first attempt per cycle."
    )
    backoff_base_seconds: float = Field(
        default=0.5, gt=0, description="Initial exponential-backoff delay."
    )
    backoff_max_seconds: float = Field(
        default=30.0, gt=0, description="Maximum backoff delay."
    )

    @field_validator("watchlist")
    @classmethod
    def _normalize(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Normalise, de-duplicate and validate the watchlist."""
        seen: set[str] = set()
        result: list[str] = []
        for raw in value:
            symbol = raw.strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                result.append(symbol)
        if not result:
            raise ValueError("Watchlist must contain at least one symbol.")
        return tuple(result)


class CollectorMetricsSnapshot(BaseModel):
    """An immutable snapshot of collector metrics."""

    model_config = ConfigDict(frozen=True)

    requests_made: int = Field(default=0, ge=0)
    successful_updates: int = Field(default=0, ge=0)
    failed_updates: int = Field(default=0, ge=0)
    missing_updates: int = Field(default=0, ge=0)
    duplicate_ticks: int = Field(default=0, ge=0)
    out_of_order: int = Field(default=0, ge=0)
    stale_detections: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    average_latency_seconds: float = Field(default=0.0, ge=0)
    last_successful_update: datetime | None = Field(default=None)
    uptime_seconds: float = Field(default=0.0, ge=0)
