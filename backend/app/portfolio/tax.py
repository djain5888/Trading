"""Holding-period and tax analysis: FIFO lots, STCG/LTCG, exit-tax estimate.

Indian capital-gains rules are lot-based and FIFO: the oldest units are sold
first, and each lot's holding period at the sale (or valuation) date decides
short- vs long-term treatment.

Default rates (India, FY 2024-25 onward, post 23-Jul-2024 Budget):
    * Listed equity / equity mutual funds held > 12 months -> LONG term.
    * STCG (Sec 111A): 20%.
    * LTCG (Sec 112A): 12.5%, with a 1,25,000 aggregate annual exemption.
All rates and thresholds are configurable via :class:`TaxConfig`; the defaults
above are documented so a user in a different regime can override them.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.portfolio.models import AssetType, Transaction, TxnType


class Term(StrEnum):
    """Capital-gains term."""

    SHORT = "STCG"
    LONG = "LTCG"


class TaxConfig(BaseModel):
    """Configurable capital-gains thresholds and rates (percent)."""

    model_config = ConfigDict(frozen=True)

    equity_ltcg_days: int = Field(default=365, gt=0, description="Equity LT threshold.")
    mf_ltcg_days: int = Field(default=365, gt=0, description="Equity-MF LT threshold.")
    stcg_rate_pct: Decimal = Field(default=Decimal("20"), ge=0, le=100)
    ltcg_rate_pct: Decimal = Field(default=Decimal("12.5"), ge=0, le=100)
    ltcg_exemption: Decimal = Field(
        default=Decimal("125000"), ge=0, description="Annual LTCG exemption amount."
    )

    def ltcg_days(self, asset_type: AssetType) -> int:
        """Return the long-term threshold in days for an asset type."""
        return (
            self.equity_ltcg_days
            if asset_type is AssetType.EQUITY
            else self.mf_ltcg_days
        )


class Lot(BaseModel):
    """One open acquisition lot (FIFO unit)."""

    model_config = ConfigDict(frozen=True)

    acquired: date = Field(description="Acquisition date.")
    units: Decimal = Field(gt=0, description="Units still open in this lot.")
    cost_per_unit: Decimal = Field(ge=0, description="Cost per unit incl. buy charges.")

    def days_held(self, on: date) -> int:
        """Return calendar days held as of ``on``."""
        return (on - self.acquired).days

    def is_long_term(self, on: date, threshold_days: int) -> bool:
        """Return whether the lot is long-term as of ``on`` (strictly beyond)."""
        return self.days_held(on) > threshold_days


class RealisedGain(BaseModel):
    """A realised gain from one FIFO match at a sell."""

    model_config = ConfigDict(frozen=True)

    sell_date: date = Field(description="Date of the sell.")
    units: Decimal = Field(gt=0, description="Units matched from a lot.")
    proceeds: Decimal = Field(description="Net proceeds for these units.")
    cost: Decimal = Field(description="Cost basis for these units.")
    term: Term = Field(description="Short or long term.")

    @property
    def gain(self) -> Decimal:
        """Return the realised gain (proceeds minus cost)."""
        return self.proceeds - self.cost


class ExitTaxEstimate(BaseModel):
    """Estimated tax on a hypothetical full exit at the valuation price."""

    model_config = ConfigDict(frozen=True)

    stcg_gain: Decimal = Field(description="Aggregate short-term gain.")
    ltcg_gain: Decimal = Field(description="Aggregate long-term gain (pre-exemption).")
    ltcg_taxable: Decimal = Field(description="Long-term gain after the exemption.")
    stcg_tax: Decimal = Field(description="Estimated short-term tax.")
    ltcg_tax: Decimal = Field(description="Estimated long-term tax.")

    @property
    def total_tax(self) -> Decimal:
        """Return the total estimated exit tax."""
        return self.stcg_tax + self.ltcg_tax


class HoldingTax(BaseModel):
    """Per-holding holding-period and LTCG-readiness analysis."""

    model_config = ConfigDict(frozen=True)

    units_held: Decimal = Field(description="Open units.")
    cost_basis: Decimal = Field(description="Open cost basis.")
    ltcg_units_now: Decimal = Field(description="Units already long-term today.")
    ltcg_units_30d: Decimal = Field(description="Extra units long-term within 30 days.")
    ltcg_units_60d: Decimal = Field(description="Extra units long-term within 60 days.")
    ltcg_units_90d: Decimal = Field(description="Extra units long-term within 90 days.")


def build_lots(
    transactions: Sequence[Transaction],
    config: TaxConfig | None = None,
) -> tuple[list[Lot], list[RealisedGain]]:
    """Replay BUY/SELL transactions FIFO into open lots and realised gains.

    Dividends and charges do not touch units, so they are ignored here (they
    affect cashflow/XIRR, not lots). Transactions are processed in date order.
    The long-term threshold that classifies each realised gain comes from the
    holding's own asset type via ``config``.

    Raises:
        ValueError: If a SELL exceeds the units currently held (over-sell).
    """
    resolved = config or TaxConfig()
    ordered = sorted(transactions, key=lambda t: (t.date, t.txn_type))
    open_lots: list[Lot] = []
    realised: list[RealisedGain] = []
    asset_type = _single_asset_type(transactions)
    threshold_days = resolved.ltcg_days(asset_type)

    for txn in ordered:
        if txn.txn_type is TxnType.BUY:
            cost_per_unit = (txn.amount + txn.charges) / txn.units
            open_lots.append(
                Lot(acquired=txn.date, units=txn.units, cost_per_unit=cost_per_unit)
            )
        elif txn.txn_type is TxnType.SELL:
            _apply_sell(txn, open_lots, realised, threshold_days)
    return open_lots, realised


def _single_asset_type(transactions: Sequence[Transaction]) -> AssetType:
    """Return the asset type of a single-holding transaction set (default EQUITY)."""
    for txn in transactions:
        return txn.asset_type
    return AssetType.EQUITY


def _apply_sell(
    txn: Transaction,
    open_lots: list[Lot],
    realised: list[RealisedGain],
    threshold_days: int,
) -> None:
    """Consume ``txn.units`` from the oldest lots, recording realised gains."""
    remaining = txn.units
    net_proceeds_per_unit = (txn.amount - txn.charges) / txn.units
    while remaining > 0:
        if not open_lots:
            raise ValueError(
                f"Over-sell: SELL of {txn.units} {txn.identifier} on {txn.date} "
                "exceeds units held."
            )
        lot = open_lots[0]
        matched = min(lot.units, remaining)
        held_days = (txn.date - lot.acquired).days
        term = Term.LONG if held_days > threshold_days else Term.SHORT
        realised.append(
            RealisedGain(
                sell_date=txn.date,
                units=matched,
                proceeds=net_proceeds_per_unit * matched,
                cost=lot.cost_per_unit * matched,
                term=term,
            )
        )
        if matched == lot.units:
            open_lots.pop(0)
        else:
            open_lots[0] = lot.model_copy(update={"units": lot.units - matched})
        remaining -= matched


def holding_tax(
    open_lots: Sequence[Lot],
    asset_type: AssetType,
    valuation_date: date,
    config: TaxConfig,
) -> HoldingTax:
    """Summarise open lots' units, cost and LTCG readiness at 30/60/90 days."""
    threshold = config.ltcg_days(asset_type)
    units_held = sum((lot.units for lot in open_lots), Decimal(0))
    cost_basis = sum((lot.units * lot.cost_per_unit for lot in open_lots), Decimal(0))

    def eligible_by(target: date) -> Decimal:
        return sum(
            (lot.units for lot in open_lots if lot.is_long_term(target, threshold)),
            Decimal(0),
        )

    now = eligible_by(valuation_date)
    by_30 = eligible_by(valuation_date + timedelta(days=30))
    by_60 = eligible_by(valuation_date + timedelta(days=60))
    by_90 = eligible_by(valuation_date + timedelta(days=90))
    # Non-overlapping buckets (each counts only units crossing in that window),
    # so the report can sum them without double-counting.
    return HoldingTax(
        units_held=units_held,
        cost_basis=cost_basis,
        ltcg_units_now=now,
        ltcg_units_30d=by_30 - now,
        ltcg_units_60d=by_60 - by_30,
        ltcg_units_90d=by_90 - by_60,
    )


def exit_tax(
    open_lots: Sequence[Lot],
    asset_type: AssetType,
    valuation_date: date,
    price: Decimal,
    config: TaxConfig,
) -> ExitTaxEstimate:
    """Estimate tax if every open lot were sold now at ``price``.

    Gains are split short/long by each lot's holding period at ``valuation_date``.
    The LTCG exemption is applied to the aggregate long-term gain (a portfolio
    caller may instead net exemption across holdings; here it is per call).
    """
    threshold = config.ltcg_days(asset_type)
    stcg_gain = Decimal(0)
    ltcg_gain = Decimal(0)
    for lot in open_lots:
        gain = (price - lot.cost_per_unit) * lot.units
        if lot.is_long_term(valuation_date, threshold):
            ltcg_gain += gain
        else:
            stcg_gain += gain
    ltcg_taxable = max(Decimal(0), ltcg_gain - config.ltcg_exemption)
    stcg_tax = max(Decimal(0), stcg_gain) * config.stcg_rate_pct / Decimal(100)
    ltcg_tax = ltcg_taxable * config.ltcg_rate_pct / Decimal(100)
    return ExitTaxEstimate(
        stcg_gain=stcg_gain,
        ltcg_gain=ltcg_gain,
        ltcg_taxable=ltcg_taxable,
        stcg_tax=stcg_tax,
        ltcg_tax=ltcg_tax,
    )
