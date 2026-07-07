"""Value objects for the relative-strength engine."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.market.enums import Exchange

#: The neutral relative-strength value used when a component cannot be computed.
NEUTRAL_RS = 50.0


class RSConfig(BaseModel):
    """Tuning and metadata for the relative-strength engine.

    ``sector_map`` mirrors the sector metadata; when a symbol has no sector its
    sector component degrades to neutral rather than failing.
    """

    model_config = ConfigDict(frozen=True)

    index_symbol: str = Field(
        default="NIFTY", min_length=1, description="Benchmark index symbol."
    )
    index_exchange: Exchange = Field(
        default=Exchange.NSE, description="Exchange the index trades on."
    )
    lookback_periods: int = Field(
        default=20, ge=1, description="Bars over which the return is measured."
    )
    leader_percentile: float = Field(
        default=80.0,
        ge=0,
        le=100,
        description="Minimum percentile on both axes to be a leader.",
    )
    sector_map: dict[str, str] = Field(
        default_factory=dict, description="Symbol -> sector name."
    )
    history_days: int = Field(
        default=500, ge=1, description="Calendar days of history to read."
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

    @property
    def required_candles(self) -> int:
        """Minimum candles a symbol needs to measure its return."""
        return self.lookback_periods + 1


class RelativeStrength(BaseModel):
    """One symbol's relative strength versus the market and its sector."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="The symbol scored.")
    rs_vs_market: float = Field(
        ge=0, le=100, description="Percentile strength versus the index."
    )
    rs_vs_sector: float = Field(
        ge=0, le=100, description="Percentile strength versus the sector."
    )
    composite: float = Field(
        ge=0, le=100, description="Mean of the two relative-strength percentiles."
    )
    leader: bool = Field(description="Whether the symbol leads on both axes.")


class RSReport(BaseModel):
    """The relative-strength engine's output.

    ``available`` is ``False`` only when no symbol had enough history to score;
    a missing index or sector merely neutralises that component.
    """

    model_config = ConfigDict(frozen=True)

    available: bool = Field(description="Whether any symbol could be scored.")
    market_available: bool = Field(
        default=False, description="Whether the benchmark index was present."
    )
    entries: tuple[RelativeStrength, ...] = Field(
        default_factory=tuple, description="Scored symbols, strongest composite first."
    )
    leaders: tuple[RelativeStrength, ...] = Field(
        default_factory=tuple, description="Symbols leading on both axes."
    )
    detail: str = Field(description="Human-readable summary of the scoring.")
    generated_at: datetime = Field(description="When the report was produced.")

    @property
    def by_symbol(self) -> dict[str, RelativeStrength]:
        """Return the entries keyed by symbol for quick lookup."""
        return {entry.symbol: entry for entry in self.entries}

    @classmethod
    def unavailable(cls, *, generated_at: datetime, detail: str) -> RSReport:
        """Build a graceful "relative strength unavailable" report."""
        return cls(available=False, detail=detail, generated_at=generated_at)
