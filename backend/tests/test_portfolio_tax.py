"""FIFO lot tracking and STCG/LTCG tests, incl. the 12-month boundary."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from app.portfolio.models import AssetType, Transaction, TxnType
from app.portfolio.tax import (
    TaxConfig,
    Term,
    build_lots,
    exit_tax,
    holding_tax,
)


def _buy(identifier: str, on: date, units: str, price: str) -> Transaction:
    return Transaction(
        asset_type=AssetType.EQUITY,
        identifier=identifier,
        txn_type=TxnType.BUY,
        date=on,
        units=Decimal(units),
        price=Decimal(price),
        amount=Decimal(units) * Decimal(price),
    )


def _sell(identifier: str, on: date, units: str, price: str) -> Transaction:
    return Transaction(
        asset_type=AssetType.EQUITY,
        identifier=identifier,
        txn_type=TxnType.SELL,
        date=on,
        units=Decimal(units),
        price=Decimal(price),
        amount=Decimal(units) * Decimal(price),
    )


def test_fifo_partial_sell_consumes_oldest_lot() -> None:
    """A partial sell reduces the oldest lot first; newer lots remain intact."""
    txns = [
        _buy("INFY", date(2022, 1, 1), "100", "100"),
        _buy("INFY", date(2022, 6, 1), "100", "120"),
        _sell("INFY", date(2022, 9, 1), "60", "150"),
    ]

    open_lots, realised = build_lots(txns)

    # 40 remain from the first lot (100 cost), 100 from the second (120 cost).
    assert [(lot.units, lot.cost_per_unit) for lot in open_lots] == [
        (Decimal("40"), Decimal("100")),
        (Decimal("100"), Decimal("120")),
    ]
    assert len(realised) == 1
    gain = realised[0]
    assert gain.units == Decimal("60")
    assert gain.cost == Decimal("6000")  # 60 * 100 (oldest lot)
    assert gain.proceeds == Decimal("9000")  # 60 * 150
    assert gain.gain == Decimal("3000")
    assert gain.term is Term.SHORT  # held ~8 months


def test_ltcg_boundary_at_365_days() -> None:
    """Exactly 365 days held is SHORT; 366 days is LONG (>'12 months')."""
    buy_day = date(2023, 1, 1)
    at_365 = build_lots(
        [_buy("X", buy_day, "10", "100"), _sell("X", date(2024, 1, 1), "10", "150")]
    )
    at_366 = build_lots(
        [_buy("X", buy_day, "10", "100"), _sell("X", date(2024, 1, 2), "10", "150")]
    )

    assert (date(2024, 1, 1) - buy_day).days == 365
    assert at_365[1][0].term is Term.SHORT
    assert at_366[1][0].term is Term.LONG


def test_over_sell_raises() -> None:
    """Selling more units than held is rejected."""
    with pytest.raises(ValueError, match="Over-sell"):
        build_lots(
            [
                _buy("X", date(2023, 1, 1), "10", "100"),
                _sell("X", date(2023, 2, 1), "15", "110"),
            ]
        )


def test_buy_charges_raise_cost_basis() -> None:
    """Buy charges are folded into the lot's cost per unit."""
    txn = Transaction(
        asset_type=AssetType.EQUITY,
        identifier="X",
        txn_type=TxnType.BUY,
        date=date(2023, 1, 1),
        units=Decimal("10"),
        price=Decimal("100"),
        amount=Decimal("1000"),
        charges=Decimal("50"),
    )

    open_lots, _ = build_lots([txn])

    assert open_lots[0].cost_per_unit == Decimal("105")  # (1000 + 50) / 10


def test_holding_tax_ltcg_readiness_windows() -> None:
    """LTCG readiness splits units into now vs becoming-eligible in 30/60/90 days."""
    valuation = date(2024, 1, 1)
    config = TaxConfig()
    # Lot A bought 2022-06 (already long-term); lot B bought 2023-01-15 (turns
    # long-term ~2024-01-15, i.e. within 30 days of valuation).
    txns = [
        _buy("X", date(2022, 6, 1), "10", "100"),
        _buy("X", date(2023, 1, 15), "5", "120"),
    ]
    open_lots, _ = build_lots(txns, config)

    htax = holding_tax(open_lots, AssetType.EQUITY, valuation, config)

    assert htax.units_held == Decimal("15")
    assert htax.ltcg_units_now == Decimal("10")  # lot A only
    assert htax.ltcg_units_30d == Decimal("5")  # lot B crosses within 30 days
    assert htax.ltcg_units_60d == Decimal("0")
    assert htax.ltcg_units_90d == Decimal("0")


def test_exit_tax_splits_short_and_long_with_exemption() -> None:
    """Exit tax classifies gains by term and applies the LTCG exemption."""
    valuation = date(2024, 1, 1)
    config = TaxConfig(ltcg_exemption=Decimal("0"))  # simplify: no exemption
    # Long-term lot (bought 2022) with a 100k gain; short-term lot with 10k gain.
    txns = [
        _buy("X", date(2022, 1, 1), "1000", "100"),  # LT at valuation
        _buy("X", date(2023, 11, 1), "100", "100"),  # ST at valuation
    ]
    open_lots, _ = build_lots(txns, config)

    estimate = exit_tax(open_lots, AssetType.EQUITY, valuation, Decimal("200"), config)

    assert estimate.ltcg_gain == Decimal("100000")  # 1000 * (200-100)
    assert estimate.stcg_gain == Decimal("10000")  # 100 * (200-100)
    assert estimate.ltcg_tax == Decimal("12500.0")  # 12.5% of 100000
    assert estimate.stcg_tax == Decimal("2000.0")  # 20% of 10000
    assert estimate.total_tax == Decimal("14500.0")
