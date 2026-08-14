"""Canonical portfolio domain models: transactions, holdings and config.

Titan v2 manages an existing portfolio rather than generating signals. Every
importer (Groww equity CSV, CAMS/KFintech CAS, manual JSON) normalises its
source into the single :class:`Transaction` model here, so the XIRR, tax and
reporting engines never see a source-specific shape.
"""

from __future__ import annotations

import datetime
import json
from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: The oldest transaction date we treat as plausible (Indian demat era).
_MIN_TXN_DATE = date(1990, 1, 1)


class AssetType(StrEnum):
    """The two asset classes Titan v2 tracks."""

    EQUITY = "EQUITY"
    MF = "MF"


class Sleeve(StrEnum):
    """Portfolio sleeves, in decreasing stability/target-weight order."""

    CORE = "CORE"
    SATELLITE = "SATELLITE"
    TACTICAL = "TACTICAL"


class TxnType(StrEnum):
    """The transaction kinds that affect units, cost basis or cashflow."""

    BUY = "BUY"  # includes SIP instalments (each instalment is one BUY)
    SELL = "SELL"  # includes partial sells
    DIVIDEND = "DIVIDEND"  # cash payout / dividend (inflow, no units)
    CHARGE = "CHARGE"  # standalone fee/charge (outflow, no units)


class Transaction(BaseModel):
    """One canonical transaction, whatever statement it was imported from.

    Amounts are gross (``units * price`` for buys/sells); ``charges`` are the
    brokerage/STT/stamp levied on top and are always a cost. Signed cashflows
    for XIRR are derived by :meth:`cashflow`, never stored, so the sign
    convention lives in exactly one place.
    """

    model_config = ConfigDict(frozen=True)

    asset_type: AssetType = Field(description="EQUITY or MF.")
    identifier: str = Field(min_length=1, description="Ticker (equity) / scheme code.")
    name: str = Field(default="", description="Human-readable instrument name.")
    txn_type: TxnType = Field(description="BUY / SELL / DIVIDEND / CHARGE.")
    date: datetime.date = Field(description="Trade / allotment date.")
    units: Decimal = Field(default=Decimal(0), ge=0, description="Units transacted.")
    price: Decimal = Field(default=Decimal(0), ge=0, description="Price/NAV per unit.")
    amount: Decimal = Field(ge=0, description="Gross value of the transaction.")
    charges: Decimal = Field(default=Decimal(0), ge=0, description="Fees/taxes levied.")
    source: str = Field(default="", description="Which importer produced this row.")

    @model_validator(mode="after")
    def _validate(self) -> Transaction:
        """Enforce per-kind invariants and a sane date."""
        if self.date < _MIN_TXN_DATE:
            raise ValueError(f"Transaction date {self.date} is implausibly old.")
        if self.txn_type in (TxnType.BUY, TxnType.SELL) and self.units <= 0:
            raise ValueError(f"{self.txn_type} must have positive units.")
        if self.txn_type in (TxnType.DIVIDEND, TxnType.CHARGE) and self.units != 0:
            raise ValueError(f"{self.txn_type} must not carry units.")
        if self.txn_type in (TxnType.DIVIDEND, TxnType.CHARGE) and self.amount <= 0:
            raise ValueError(f"{self.txn_type} must have a positive amount.")
        return self

    @property
    def key(self) -> tuple[AssetType, str]:
        """Return the holding key this transaction belongs to."""
        return (self.asset_type, self.identifier)

    def cashflow(self) -> Decimal:
        """Return the signed investor cashflow (outflow negative, inflow positive).

        Buys and standalone charges are money leaving the investor; sells and
        dividends are money returning. Charges reduce the proceeds of a sell and
        add to the cost of a buy.
        """
        if self.txn_type is TxnType.BUY:
            return -(self.amount + self.charges)
        if self.txn_type is TxnType.SELL:
            return self.amount - self.charges
        if self.txn_type is TxnType.DIVIDEND:
            return self.amount - self.charges
        return -self.amount  # CHARGE


class Holding(BaseModel):
    """A tracked instrument and the sleeve it belongs to."""

    model_config = ConfigDict(frozen=True)

    identifier: str = Field(min_length=1, description="Ticker / scheme code / name.")
    name: str = Field(min_length=1, description="Human-readable name.")
    asset_type: AssetType = Field(description="EQUITY or MF.")
    sleeve: Sleeve = Field(description="CORE / SATELLITE / TACTICAL.")
    scheme_code: str | None = Field(
        default=None,
        description="Explicit AMFI scheme code override (MF only, skips name match).",
    )

    @property
    def key(self) -> tuple[AssetType, str]:
        """Return the (asset_type, identifier) key."""
        return (self.asset_type, self.identifier)


class SleeveTargets(BaseModel):
    """Target sleeve weights (percent) and the XIRR band (percent)."""

    model_config = ConfigDict(frozen=True)

    core_pct: Decimal = Field(default=Decimal(70), ge=0, le=100)
    satellite_pct: Decimal = Field(default=Decimal(20), ge=0, le=100)
    tactical_pct: Decimal = Field(default=Decimal(10), ge=0, le=100)
    xirr_low: Decimal = Field(default=Decimal(15), description="Lower XIRR band (%).")
    xirr_high: Decimal = Field(default=Decimal(20), description="Upper XIRR band (%).")

    @model_validator(mode="after")
    def _validate(self) -> SleeveTargets:
        """Sleeve targets must sum to 100 and the band must be ordered."""
        total = self.core_pct + self.satellite_pct + self.tactical_pct
        if abs(total - Decimal(100)) > Decimal("0.01"):
            raise ValueError(f"Sleeve targets must sum to 100 (got {total}).")
        if self.xirr_low >= self.xirr_high:
            raise ValueError("xirr_low must be below xirr_high.")
        return self

    def target_for(self, sleeve: Sleeve) -> Decimal:
        """Return the target weight (%) for one sleeve."""
        return {
            Sleeve.CORE: self.core_pct,
            Sleeve.SATELLITE: self.satellite_pct,
            Sleeve.TACTICAL: self.tactical_pct,
        }[sleeve]


class PortfolioConfig(BaseModel):
    """The portfolio: its holdings and its sleeve/XIRR targets."""

    model_config = ConfigDict(frozen=True)

    holdings: tuple[Holding, ...] = Field(description="Tracked instruments.")
    targets: SleeveTargets = Field(
        default_factory=SleeveTargets, description="Sleeve weights and XIRR band."
    )

    @model_validator(mode="after")
    def _validate(self) -> PortfolioConfig:
        """Holding keys must be unique."""
        seen: set[tuple[AssetType, str]] = set()
        for holding in self.holdings:
            if holding.key in seen:
                raise ValueError(f"Duplicate holding {holding.key}.")
            seen.add(holding.key)
        return self

    def by_key(self) -> dict[tuple[AssetType, str], Holding]:
        """Return holdings indexed by (asset_type, identifier)."""
        return {h.key: h for h in self.holdings}


def load_portfolio_config(path: Path) -> PortfolioConfig:
    """Load and validate a ``portfolio.json`` file.

    Raises:
        ValueError: If the file is missing, malformed, or fails validation.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read portfolio config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Portfolio config {path} must be a JSON object.")
    return PortfolioConfig(**raw)
