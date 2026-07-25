"""Portfolio report value objects produced by the analysis engine."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.portfolio.models import AssetType, Sleeve
from app.portfolio.prices import HoldingMismatch


class BandStatus(StrEnum):
    """Where the portfolio's XIRR sits relative to the target band."""

    BELOW = "BELOW"
    WITHIN = "WITHIN"
    ABOVE = "ABOVE"


class HoldingReport(BaseModel):
    """Per-holding measurement."""

    model_config = ConfigDict(frozen=True)

    identifier: str = Field(description="Symbol / scheme code.")
    name: str = Field(description="Instrument name.")
    asset_type: AssetType = Field(description="EQUITY or MF.")
    sleeve: Sleeve = Field(description="Sleeve.")
    units: Decimal = Field(description="Open units.")
    price: Decimal | None = Field(description="Valuation price/NAV (None=unpriced).")
    market_value: Decimal = Field(description="Units * price (0 if unpriced).")
    cost_basis: Decimal = Field(description="Open cost basis.")
    invested: Decimal = Field(description="Gross buys incl. charges.")
    realised_proceeds: Decimal = Field(description="Net proceeds from sells.")
    realised_gain: Decimal = Field(description="Realised gain (proceeds minus cost).")
    dividends: Decimal = Field(description="Dividends received.")
    unrealised_pnl: Decimal = Field(description="Market value minus cost basis.")
    absolute_return_pct: float = Field(description="Simple return on invested (%).")
    xirr_pct: float | None = Field(description="Annualised XIRR (%), None if n/a.")
    ltcg_ready_value: Decimal = Field(description="Value already long-term.")
    ltcg_value_90d: Decimal = Field(description="Extra value long-term within 90 days.")
    exit_tax: Decimal = Field(description="Estimated tax on a full exit now.")
    priced: bool = Field(description="Whether a valuation price was available.")


class SleeveReport(BaseModel):
    """Per-sleeve allocation and return."""

    model_config = ConfigDict(frozen=True)

    sleeve: Sleeve = Field(description="Sleeve.")
    market_value: Decimal = Field(description="Sleeve market value.")
    actual_pct: float = Field(description="Share of portfolio value (%).")
    target_pct: float = Field(description="Target weight (%).")
    xirr_pct: float | None = Field(description="Sleeve XIRR (%), None if n/a.")

    @property
    def drift_pct(self) -> float:
        """Return actual minus target weight (percentage points)."""
        return round(self.actual_pct - self.target_pct, 2)


class AssetClassReport(BaseModel):
    """Per-asset-class (EQUITY/MF) roll-up."""

    model_config = ConfigDict(frozen=True)

    asset_type: AssetType = Field(description="Asset class.")
    market_value: Decimal = Field(description="Market value.")
    xirr_pct: float | None = Field(description="Asset-class XIRR (%).")


class PortfolioReport(BaseModel):
    """The full portfolio measurement (no signals — analysis only)."""

    model_config = ConfigDict(frozen=True)

    valuation_date: date = Field(description="Date the book was valued on.")
    market_value: Decimal = Field(description="Total market value.")
    cost_basis: Decimal = Field(description="Total open cost basis.")
    invested: Decimal = Field(description="Total gross buys.")
    unrealised_pnl: Decimal = Field(description="Total unrealised P&L.")
    realised_gain: Decimal = Field(description="Total realised gain to date.")
    absolute_return_pct: float = Field(description="Simple absolute return (%).")
    xirr_pct: float | None = Field(description="Portfolio XIRR (%).")
    band_low: float = Field(description="Lower XIRR band (%).")
    band_high: float = Field(description="Upper XIRR band (%).")
    band_status: BandStatus = Field(description="XIRR vs band.")
    drags: tuple[str, ...] = Field(description="Biggest return drags (identifiers).")
    concentration_id: str = Field(description="Largest holding identifier.")
    concentration_pct: float = Field(description="Largest holding share (%).")
    ltcg_ready_value: Decimal = Field(description="Value already long-term.")
    exit_tax: Decimal = Field(description="Estimated tax on a full exit now.")
    sleeves: tuple[SleeveReport, ...] = Field(description="Per-sleeve reports.")
    asset_classes: tuple[AssetClassReport, ...] = Field(description="Per-class rollup.")
    holdings: tuple[HoldingReport, ...] = Field(description="Per-holding reports.")
    unpriced: tuple[str, ...] = Field(
        default=(), description="Holdings with no valuation price."
    )
    mismatches: tuple[HoldingMismatch, ...] = Field(
        default=(), description="Broker cross-check discrepancies."
    )
