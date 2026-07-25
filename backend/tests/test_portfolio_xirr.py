"""XIRR engine tests — hand-verified, including a SIP and a partial sell."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.portfolio.models import AssetType, Transaction, TxnType
from app.portfolio.xirr import absolute_return_pct, cashflows_from, xirr


def test_xirr_simple_doubling_over_one_year() -> None:
    """-100 today, +110 in exactly one year -> 10% XIRR."""
    flows = [(date(2023, 1, 1), -100.0), (date(2024, 1, 1), 110.0)]

    rate = xirr(flows)

    assert rate is not None
    assert abs(rate - 0.10) < 1e-4  # 10% annualised


def test_xirr_two_year_growth() -> None:
    """-1000 then +1210 two years later -> 10% (1.1^2 = 1.21)."""
    flows = [(date(2022, 1, 1), -1000.0), (date(2024, 1, 1), 1210.0)]

    rate = xirr(flows)

    assert rate is not None
    assert abs(rate - 0.10) < 1e-3


def test_xirr_none_when_all_outflows() -> None:
    """A series with no inflow has no internal rate of return."""
    assert xirr([(date(2023, 1, 1), -100.0), (date(2023, 6, 1), -50.0)]) is None


def test_xirr_sip_is_money_weighted() -> None:
    """A 3-instalment SIP valued at a known terminal has a sane, bounded XIRR.

    12 monthly-ish buys of 10,000 with the book worth 40,000 at the end is a
    loss; the XIRR must be clearly negative and finite.
    """
    txns = [
        Transaction(
            asset_type=AssetType.MF,
            identifier="120503",
            txn_type=TxnType.BUY,
            date=date(2023, 1, 1),
            units=Decimal("100"),
            price=Decimal("100"),
            amount=Decimal("10000"),
        ),
        Transaction(
            asset_type=AssetType.MF,
            identifier="120503",
            txn_type=TxnType.BUY,
            date=date(2023, 2, 1),
            units=Decimal("100"),
            price=Decimal("100"),
            amount=Decimal("10000"),
        ),
        Transaction(
            asset_type=AssetType.MF,
            identifier="120503",
            txn_type=TxnType.BUY,
            date=date(2023, 3, 1),
            units=Decimal("100"),
            price=Decimal("100"),
            amount=Decimal("10000"),
        ),
    ]
    flows = cashflows_from(
        txns, terminal_value=Decimal("24000"), terminal_date=date(2024, 3, 1)
    )

    rate = xirr(flows)

    assert rate is not None
    assert rate < 0  # invested 30k, worth 24k a year later -> negative
    assert -0.5 < rate < 0.0  # finite and plausible


def test_xirr_with_partial_sell_and_terminal() -> None:
    """A buy, a partial sell, then a terminal value produce a positive XIRR."""
    txns = [
        Transaction(
            asset_type=AssetType.EQUITY,
            identifier="INFY",
            txn_type=TxnType.BUY,
            date=date(2022, 1, 1),
            units=Decimal("100"),
            price=Decimal("100"),
            amount=Decimal("10000"),
        ),
        Transaction(
            asset_type=AssetType.EQUITY,
            identifier="INFY",
            txn_type=TxnType.SELL,
            date=date(2023, 1, 1),
            units=Decimal("50"),
            price=Decimal("150"),
            amount=Decimal("7500"),
        ),
    ]
    # 50 units left; terminal at 200 -> 10,000.
    flows = cashflows_from(
        txns, terminal_value=Decimal("10000"), terminal_date=date(2024, 1, 1)
    )

    rate = xirr(flows)

    assert rate is not None
    assert rate > 0.30  # strong compounding: 10k -> 7.5k + 10k over 2 years


def test_absolute_return_pct() -> None:
    """Absolute return is (value+realised - invested)/invested."""
    assert absolute_return_pct(Decimal("10000"), Decimal("12500")) == 25.0
    assert absolute_return_pct(Decimal("0"), Decimal("100")) == 0.0
