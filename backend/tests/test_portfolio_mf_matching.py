"""MF name->AMFI-code matching and incomplete-pricing XIRR tests (BUG 1 & 2)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from app.portfolio.engine import analyse_portfolio, raise_for_unmatched
from app.portfolio.models import (
    AssetType,
    Holding,
    PortfolioConfig,
    Sleeve,
    Transaction,
    TxnType,
)
from app.portfolio.nav import parse_navall

# A NAV file carrying the usual Direct/Regular x Growth/IDCW variants.
_NAVALL = (
    "Scheme Code;ISIN1;ISIN2;Scheme Name;Net Asset Value;Date\n"
    "\n"
    "Open Ended Schemes ( Equity Scheme )\n"
    "122639;INFA;INFB;Parag Parikh Flexi Cap Fund - Direct Plan - Growth;65.4321;"
    "05-Jul-2024\n"
    "122640;INFC;INFD;Parag Parikh Flexi Cap Fund - Regular Plan - Growth;60.0;"
    "05-Jul-2024\n"
    "122641;INFE;INFF;Parag Parikh Flexi Cap Fund - Direct Plan - IDCW;40.0;"
    "05-Jul-2024\n"
    "119551;INFG;INFH;UTI Nifty 50 Index Fund - Direct Plan - Growth;120.5;"
    "05-Jul-2024\n"
)


def _mf(identifier: str, name: str, scheme_code: str | None = None) -> Holding:
    return Holding(
        identifier=identifier,
        name=name,
        asset_type=AssetType.MF,
        sleeve=Sleeve.CORE,
        scheme_code=scheme_code,
    )


def _buy_mf(identifier: str, on: date, units: str, price: str) -> Transaction:
    return Transaction(
        asset_type=AssetType.MF,
        identifier=identifier,
        txn_type=TxnType.BUY,
        date=on,
        units=Decimal(units),
        price=Decimal(price),
        amount=Decimal(units) * Decimal(price),
    )


# -- Fuzzy name matching ---------------------------------------------------


def test_match_by_name_resolves_plain_name_to_direct_growth() -> None:
    """A plain scheme name matches, preferring the Direct Growth variant."""
    nav = parse_navall(_NAVALL)

    entry = nav.match_by_name("Parag Parikh Flexi Cap Fund")

    assert entry is not None
    assert entry.scheme_code == "122639"  # Direct Growth chosen over Regular/IDCW
    assert entry.nav == Decimal("65.4321")


def test_match_by_name_ignores_case_and_punctuation() -> None:
    """Case, punctuation and plan suffixes do not block a match."""
    nav = parse_navall(_NAVALL)

    entry = nav.match_by_name("parag-parikh flexi cap fund (direct growth)")

    assert entry is not None and entry.scheme_code == "122639"


def test_match_by_name_returns_none_below_threshold() -> None:
    """An unrelated name matches nothing rather than a wrong scheme."""
    nav = parse_navall(_NAVALL)

    assert nav.match_by_name("Some Random Bluechip Bond Fund") is None


# -- Engine resolution -----------------------------------------------------


def test_mf_holding_priced_by_name_match() -> None:
    """An MF holding keyed by NAME is valued via the fuzzy AMFI match (BUG 1)."""
    config = PortfolioConfig(holdings=(_mf("PPFCF", "Parag Parikh Flexi Cap Fund"),))
    txns = [_buy_mf("PPFCF", date(2022, 1, 1), "100", "50")]
    nav = parse_navall(_NAVALL)

    report = analyse_portfolio(
        config, txns, valuation_date=date(2024, 7, 5), equity_prices={}, nav=nav
    )

    holding = report.holdings[0]
    assert holding.priced is True
    assert holding.matched_code == "122639"
    assert holding.market_value == Decimal("6543.21")  # 100 * 65.4321
    assert report.unmatched_schemes == ()


def test_explicit_scheme_code_override_is_used() -> None:
    """An explicit scheme_code override skips name matching."""
    config = PortfolioConfig(
        holdings=(_mf("PP", "Totally Different Label", scheme_code="122640"),)
    )
    txns = [_buy_mf("PP", date(2022, 1, 1), "10", "60")]
    nav = parse_navall(_NAVALL)

    report = analyse_portfolio(
        config, txns, valuation_date=date(2024, 7, 5), equity_prices={}, nav=nav
    )

    assert report.holdings[0].matched_code == "122640"  # Regular Growth, as forced
    assert report.holdings[0].price == Decimal("60.0")


# -- Unmatched scheme loud failure (BUG 1) ---------------------------------


def test_unmatched_scheme_is_flagged_and_raises() -> None:
    """An unmatchable MF scheme is flagged and raise_for_unmatched fails loudly."""
    config = PortfolioConfig(holdings=(_mf("MYSTERY", "Mystery Unlisted Fund XYZ"),))
    txns = [_buy_mf("MYSTERY", date(2022, 1, 1), "100", "50")]
    nav = parse_navall(_NAVALL)

    report = analyse_portfolio(
        config, txns, valuation_date=date(2024, 7, 5), equity_prices={}, nav=nav
    )

    assert "MYSTERY" in report.unmatched_schemes
    assert report.holdings[0].priced is False
    with pytest.raises(ValueError, match="Could not match"):
        raise_for_unmatched(report)


def test_matched_portfolio_does_not_raise() -> None:
    """A fully matched portfolio passes the loud-failure guard cleanly."""
    config = PortfolioConfig(holdings=(_mf("PPFCF", "Parag Parikh Flexi Cap Fund"),))
    txns = [_buy_mf("PPFCF", date(2022, 1, 1), "100", "50")]
    nav = parse_navall(_NAVALL)

    report = analyse_portfolio(
        config, txns, valuation_date=date(2024, 7, 5), equity_prices={}, nav=nav
    )

    raise_for_unmatched(report)  # must not raise


# -- Incomplete pricing -> n/a XIRR (BUG 2) --------------------------------


def test_incomplete_pricing_yields_na_not_clamped_bound() -> None:
    """An unpriced holding makes aggregate XIRR n/a, never a clamped +10000%."""
    config = PortfolioConfig(
        holdings=(
            Holding(
                identifier="INFY",
                name="Infosys",
                asset_type=AssetType.EQUITY,
                sleeve=Sleeve.CORE,
            ),
            Holding(
                identifier="RELIANCE",
                name="Reliance",
                asset_type=AssetType.EQUITY,
                sleeve=Sleeve.CORE,
            ),
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
        Transaction(
            asset_type=AssetType.EQUITY,
            identifier="RELIANCE",
            txn_type=TxnType.BUY,
            date=date(2022, 1, 1),
            units=Decimal("10"),
            price=Decimal("100"),
            amount=Decimal("1000"),
        ),
    ]
    # Only INFY is priced; RELIANCE has no quote -> incomplete pricing.
    report = analyse_portfolio(
        config,
        txns,
        valuation_date=date(2024, 1, 1),
        equity_prices={"INFY": Decimal("200")},
    )

    assert report.xirr_incomplete is True
    assert report.xirr_pct is None  # never the clamp ceiling (+10000%)
    core = next(s for s in report.sleeves if s.sleeve is Sleeve.CORE)
    assert core.xirr_incomplete is True and core.xirr_pct is None
    equity = next(a for a in report.asset_classes if a.asset_type is AssetType.EQUITY)
    assert equity.xirr_incomplete is True and equity.xirr_pct is None


def test_complete_pricing_still_computes_xirr() -> None:
    """With every holding priced, aggregate XIRR is a real number again."""
    config = PortfolioConfig(
        holdings=(
            Holding(
                identifier="INFY",
                name="Infosys",
                asset_type=AssetType.EQUITY,
                sleeve=Sleeve.CORE,
            ),
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
        )
    ]

    report = analyse_portfolio(
        config,
        txns,
        valuation_date=date(2024, 1, 1),
        equity_prices={"INFY": Decimal("200")},
    )

    assert report.xirr_incomplete is False
    assert report.xirr_pct is not None and 0.0 < report.xirr_pct < 100.0
