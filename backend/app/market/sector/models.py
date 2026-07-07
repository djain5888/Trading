"""Value objects for the sector-strength engine."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SectorTrend(StrEnum):
    """The direction of a sector's aggregate EMA."""

    UP = "UP"
    FLAT = "FLAT"
    DOWN = "DOWN"


class SectorConfig(BaseModel):
    """Tunable parameters and the symbol-to-sector metadata.

    ``sector_map`` maps each watchlist symbol to its sector. When it is empty
    the engine has no metadata and degrades to an "unavailable" report.
    """

    model_config = ConfigDict(frozen=True)

    sector_map: dict[str, str] = Field(
        default_factory=dict, description="Symbol -> sector name."
    )
    ema_period: int = Field(
        default=50, ge=1, description="EMA period each member is compared to."
    )
    momentum_period: int = Field(
        default=10, ge=1, description="Rate-of-change period for member momentum."
    )
    slope_lookback: int = Field(
        default=10, ge=1, description="Bars back for the aggregate-EMA slope."
    )
    min_members: int = Field(
        default=2, ge=1, description="Valid members a sector needs to be ranked."
    )
    history_days: int = Field(
        default=500, ge=1, description="Calendar days of history to read."
    )
    breadth_weight: float = Field(
        default=0.4, ge=0, le=1, description="Weight of the % -above-EMA signal."
    )
    price_weight: float = Field(
        default=0.3, ge=0, le=1, description="Weight of the price-versus-EMA signal."
    )
    momentum_weight: float = Field(
        default=0.3, ge=0, le=1, description="Weight of the momentum signal."
    )
    trend_flat_pct: float = Field(
        default=0.1, ge=0, description="Abs. slope percent below which trend is FLAT."
    )
    top_n: int = Field(
        default=3, ge=1, description="Size of the strongest/weakest convenience lists."
    )

    @field_validator("sector_map")
    @classmethod
    def _normalise_map(cls, value: dict[str, str]) -> dict[str, str]:
        """Upper-case and trim symbols so lookups match stored series keys."""
        return {
            symbol.strip().upper(): sector.strip()
            for symbol, sector in value.items()
            if symbol.strip() and sector.strip()
        }

    @model_validator(mode="after")
    def _validate(self) -> SectorConfig:
        """Enforce unit scoring weights."""
        total = self.breadth_weight + self.price_weight + self.momentum_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Scoring weights must sum to 1.0.")
        return self

    @property
    def required_candles(self) -> int:
        """Minimum candles a member needs for the EMA and momentum."""
        return max(self.ema_period, self.momentum_period + 1)


class SectorStrength(BaseModel):
    """One sector's strength score and trend."""

    model_config = ConfigDict(frozen=True)

    sector: str = Field(description="Sector name.")
    score: float = Field(ge=0, le=100, description="Aggregate strength score.")
    trend: SectorTrend = Field(description="Aggregate-EMA trend direction.")
    members: int = Field(ge=0, description="Valid members contributing to the score.")
    above_pct: float = Field(
        ge=0, le=100, description="Percent of members above their EMA."
    )


class SectorReport(BaseModel):
    """The sector-strength engine's output.

    When ``available`` is ``False`` no sector metadata was configured and the
    lists are empty; consumers should render "sectors unavailable".
    """

    model_config = ConfigDict(frozen=True)

    available: bool = Field(description="Whether a ranking was produced.")
    ranked: tuple[SectorStrength, ...] = Field(
        default_factory=tuple, description="Sectors ranked strongest-first."
    )
    strongest: tuple[SectorStrength, ...] = Field(
        default_factory=tuple, description="The strongest sectors (convenience)."
    )
    weakest: tuple[SectorStrength, ...] = Field(
        default_factory=tuple, description="The weakest sectors (convenience)."
    )
    detail: str = Field(description="Human-readable summary of the ranking.")
    generated_at: datetime = Field(description="When the report was produced.")

    @classmethod
    def unavailable(cls, *, generated_at: datetime, detail: str) -> SectorReport:
        """Build a graceful "sectors unavailable" report."""
        return cls(available=False, detail=detail, generated_at=generated_at)
