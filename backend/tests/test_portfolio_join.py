"""Transaction<->holding join, zero-match loud failure, clamp-never-rendered."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from app.portfolio.engine import (
    analyse_portfolio,
    raise_for_join_errors,
)
from app.portfolio.models import (
    AssetType,
    Holding,
    PortfolioConfig,
    Sleeve,
    Transaction,
    TxnType,
)
from app.portfolio.nav import parse_navall
from app.portfolio.render import render_portfolio
from app.portfolio.xirr import xirr

_NAVALL = (
    "Scheme Code;A;B;Scheme Name;Net Asset Value;Date\n"
    "122639;X;Y;Parag Parikh Flexi Cap Fund - Direct Plan - Growth;65.4321;"
    "05-Jul-2024\n"
)


def _mf_holding(identifier: str, name: str, sleeve: Sleeve = Sleeve.CORE) -> Holding:
    return Holding(
        identifier=identifier, name=name, asset_type=AssetType.MF, sleeve=sleeve
    )


def _mf_txn(
    scheme: str, kind: TxnType, on: date, units: str, price: str
) -> Transaction:
    return Transaction(
        asset_type=AssetType.MF,
        identifier=scheme.upper(),
        name=scheme,
        txn_type=kind,
        date=on,
        units=Decimal(units),
        price=Decimal(price),
        amount=Decimal(units) * Decimal(price),
    )


# -- Name-normalised join (BUG A) ------------------------------------------


def test_mf_transactions_join_by_normalised_name() -> None:
    """CAS txns (name-keyed, differing suffix/case) join to the holding by name."""
    config = PortfolioConfig(
        holdings=(_mf_holding("PPFCF", "Parag Parikh Flexi Cap Fund"),)
    )
    # The statement's scheme label differs in case and lacks "Fund".
    txns = [
        _mf_txn("parag parikh flexi cap", TxnType.BUY, date(2022, 1, 1), "100", "40"),
        _mf_txn("parag parikh flexi cap", TxnType.BUY, date(2022, 6, 1), "50", "48"),
    ]
    nav = parse_navall(_NAVALL)

    report = analyse_portfolio(
        config, txns, valuation_date=date(2024, 7, 5), equity_prices={}, nav=nav
    )

    holding = report.holdings[0]
    assert holding.matched_txns == 2  # both joined despite the name differences
    assert holding.units == Decimal("150")
    assert holding.market_value == Decimal("9814.815")  # 150 * 65.4321
    assert report.zero_txn_holdings == ()
    assert report.held_unconfigured == ()
    raise_for_join_errors(report)  # clean join -> no raise


# -- Zero-match loud failure (BUG A) ---------------------------------------


def test_zero_match_holding_and_orphan_txn_raise() -> None:
    """An MF holding with no matching txns, and an orphan txn, both fail loudly."""
    config = PortfolioConfig(
        holdings=(_mf_holding("PPFCF", "Parag Parikh Flexi Cap Fund"),)
    )
    # Transaction is for a totally different scheme -> matches no holding, and
    # the holding therefore matches zero transactions.
    txns = [
        _mf_txn("Quantum Long Term Equity", TxnType.BUY, date(2022, 1, 1), "10", "50")
    ]
    nav = parse_navall(_NAVALL)

    report = analyse_portfolio(
        config, txns, valuation_date=date(2024, 7, 5), equity_prices={}, nav=nav
    )

    assert "PPFCF" in report.zero_txn_holdings
    # A non-zero orphan position (still holding something not in config) is an
    # error listed under held_unconfigured.
    assert any("Quantum" in label for label in report.held_unconfigured)
    with pytest.raises(ValueError, match="Transaction join failed"):
        raise_for_join_errors(report)


# -- Closed positions (fully exited) are legitimate, not errors ------------


def test_fully_exited_orphan_is_a_closed_position_not_an_error() -> None:
    """An exited symbol absent from config is a closed position, not a failure."""
    config = PortfolioConfig(
        holdings=(_mf_holding("PPFCF", "Parag Parikh Flexi Cap Fund"),)
    )
    # A holding fully bought then sold, and it is NOT in portfolio.json.
    orphan = [
        _mf_txn("Quantum Long Term Equity", TxnType.BUY, date(2021, 1, 1), "100", "20"),
        _mf_txn(
            "Quantum Long Term Equity", TxnType.SELL, date(2023, 1, 1), "100", "26"
        ),
    ]
    # Plus a genuinely held, configured MF so the run has an open position too.
    held = [
        _mf_txn(
            "Parag Parikh Flexi Cap Fund", TxnType.BUY, date(2022, 1, 1), "10", "40"
        )
    ]
    nav = parse_navall(_NAVALL)

    report = analyse_portfolio(
        config,
        orphan + held,
        valuation_date=date(2024, 7, 5),
        equity_prices={},
        nav=nav,
    )

    # The exited orphan is a closed position, not a join error.
    assert report.held_unconfigured == ()
    assert len(report.closed_positions) == 1
    closed = report.closed_positions[0]
    assert "Quantum" in closed.identifier
    assert closed.trades == 2
    assert closed.realised_gain == Decimal("600")  # 100*(26-20)
    assert closed.xirr_pct is not None and closed.xirr_pct > 0  # +30% over 2y
    # It does not block the run.
    raise_for_join_errors(report)
    # Its cashflows count toward a real portfolio XIRR (not withheld).
    assert report.xirr_incomplete is False
    assert report.xirr_pct is not None


def test_closed_config_holding_keeps_real_xirr() -> None:
    """A configured holding fully exited keeps a real XIRR and does not block."""
    config = PortfolioConfig(
        holdings=(_mf_holding("PPFCF", "Parag Parikh Flexi Cap Fund"),)
    )
    txns = [
        _mf_txn(
            "Parag Parikh Flexi Cap Fund", TxnType.BUY, date(2022, 1, 1), "100", "40"
        ),
        _mf_txn(
            "Parag Parikh Flexi Cap Fund", TxnType.SELL, date(2023, 1, 1), "100", "60"
        ),
    ]
    nav = parse_navall(_NAVALL)

    report = analyse_portfolio(
        config, txns, valuation_date=date(2024, 7, 5), equity_prices={}, nav=nav
    )

    holding = report.holdings[0]
    assert holding.matched_txns == 2 and holding.units == Decimal("0")
    assert report.xirr_incomplete is False  # fully-exited cashflows are complete
    assert report.xirr_pct is not None  # a real, computable rate
    raise_for_join_errors(report)  # not an error


# -- Clamped bound never surfaced (BUG B) ----------------------------------


def test_xirr_clamp_bound_returns_none() -> None:
    """A return so extreme it pins the solver to the domain bound yields None."""
    # -1 then +1e9 the next day -> annualised rate far beyond the +100.0 clamp.
    flows = [(date(2022, 1, 1), -1.0), (date(2022, 1, 2), 1_000_000_000.0)]

    assert xirr(flows) is None  # never the clamp ceiling


def test_incomplete_portfolio_never_renders_clamped_bound() -> None:
    """The +10000% bug: an unpriced MF with cashflows renders n/a, not a bound."""
    config = PortfolioConfig(
        holdings=(
            Holding(
                identifier="INFY",
                name="Infosys",
                asset_type=AssetType.EQUITY,
                sleeve=Sleeve.CORE,
            ),
            _mf_holding("PPFCF", "Parag Parikh Flexi Cap Fund"),
        )
    )
    txns = [
        Transaction(
            asset_type=AssetType.EQUITY,
            identifier="INFY",
            txn_type=TxnType.BUY,
            date=date(2022, 1, 1),
            units=Decimal("10"),
            price=Decimal("100"),
            amount=Decimal("1000"),
        ),
        _mf_txn(
            "Parag Parikh Flexi Cap Fund", TxnType.BUY, date(2022, 1, 1), "100", "40"
        ),
    ]
    # No NAV supplied -> the MF is held (units 150... here 100) but unpriced.
    report = analyse_portfolio(
        config,
        txns,
        valuation_date=date(2024, 1, 1),
        equity_prices={"INFY": Decimal("200")},
        nav=None,
    )

    assert report.holdings  # both holdings present
    ppfcf = next(h for h in report.holdings if h.identifier == "PPFCF")
    assert ppfcf.matched_txns == 1 and ppfcf.units == Decimal("100")  # joined
    assert ppfcf.priced is False  # no NAV -> unpriced (not "not held")
    assert report.xirr_incomplete is True
    assert report.xirr_pct is None
    rendered = render_portfolio(report)
    assert "10000" not in rendered  # a clamped bound is never printed
    assert "n/a (incomplete pricing)" in rendered
