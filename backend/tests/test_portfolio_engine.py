"""Engine, config, allocation-drift, cross-check and NAV-staleness tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from app.portfolio.engine import analyse_portfolio
from app.portfolio.models import (
    AssetType,
    Holding,
    PortfolioConfig,
    Sleeve,
    SleeveTargets,
    Transaction,
    TxnType,
)
from app.portfolio.nav import load_navs, parse_navall
from app.portfolio.prices import cross_check_holdings
from app.portfolio.report import BandStatus


def _buy(
    identifier: str, asset: AssetType, on: date, units: str, price: str
) -> Transaction:
    return Transaction(
        asset_type=asset,
        identifier=identifier,
        txn_type=TxnType.BUY,
        date=on,
        units=Decimal(units),
        price=Decimal(price),
        amount=Decimal(units) * Decimal(price),
    )


# -- Config validation -----------------------------------------------------


def test_sleeve_targets_must_sum_to_100() -> None:
    """Targets that do not sum to 100 are rejected."""
    with pytest.raises(ValueError, match="sum to 100"):
        SleeveTargets(
            core_pct=Decimal("70"),
            satellite_pct=Decimal("20"),
            tactical_pct=Decimal("5"),
        )


def test_portfolio_rejects_duplicate_holdings() -> None:
    """Two holdings with the same key are rejected."""
    holding = Holding(
        identifier="INFY",
        name="Infosys",
        asset_type=AssetType.EQUITY,
        sleeve=Sleeve.CORE,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        PortfolioConfig(holdings=(holding, holding))


def test_transaction_rejects_oversell_shape_and_bad_dates() -> None:
    """Zero-unit buys and implausibly old dates are rejected at construction."""
    with pytest.raises(ValueError):
        Transaction(
            asset_type=AssetType.EQUITY,
            identifier="X",
            txn_type=TxnType.BUY,
            date=date(2023, 1, 1),
            units=Decimal("0"),
            amount=Decimal("0"),
        )
    with pytest.raises(ValueError, match="implausibly old"):
        Transaction(
            asset_type=AssetType.EQUITY,
            identifier="X",
            txn_type=TxnType.BUY,
            date=date(1980, 1, 1),
            units=Decimal("1"),
            price=Decimal("1"),
            amount=Decimal("1"),
        )


# -- Cross-check -----------------------------------------------------------


def test_cross_check_flags_quantity_mismatches() -> None:
    """Imported vs broker quantity differences beyond tolerance are flagged."""
    mismatches = cross_check_holdings(
        {"INFY": Decimal("10"), "TCS": Decimal("5")},
        {"INFY": Decimal("10"), "TCS": Decimal("6"), "WIPRO": Decimal("3")},
    )

    flagged = {m.identifier for m in mismatches}
    assert flagged == {"TCS", "WIPRO"}  # INFY matches; TCS off by 1; WIPRO extra
    assert all(m.identifier != "INFY" for m in mismatches)


# -- NAV staleness ---------------------------------------------------------


def test_load_navs_fails_loudly_on_stale_cache(tmp_path: Path) -> None:
    """A cache older than the freshness bound raises rather than mis-valuing."""
    cache = tmp_path / "navall.txt"
    cache.write_text(
        "Scheme Code;A;B;Scheme Name;Net Asset Value;Date\n"
        "120503;;;Fund;50;01-Jan-2024\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stale"):
        load_navs(
            cache, on=date(2024, 3, 1), fetcher=None, offline=True, max_stale_days=5
        )

    # Fresh enough -> loads fine.
    snapshot = load_navs(cache, on=date(2024, 1, 3), fetcher=None, offline=True)
    assert snapshot.get("120503") is not None


def test_load_navs_missing_cache_raises(tmp_path: Path) -> None:
    """No cache and no fetch -> loud failure, never a silent zero valuation."""
    with pytest.raises(FileNotFoundError):
        load_navs(
            tmp_path / "absent.txt", on=date(2024, 1, 1), fetcher=None, offline=True
        )


# -- End-to-end measurement ------------------------------------------------


def _config() -> PortfolioConfig:
    return PortfolioConfig(
        holdings=(
            Holding(
                identifier="INFY",
                name="Infosys",
                asset_type=AssetType.EQUITY,
                sleeve=Sleeve.CORE,
            ),
            Holding(
                identifier="TATAMOTORS",
                name="Tata Motors",
                asset_type=AssetType.EQUITY,
                sleeve=Sleeve.SATELLITE,
            ),
            Holding(
                identifier="120503",
                name="PPFCF",
                asset_type=AssetType.MF,
                sleeve=Sleeve.CORE,
            ),
        ),
        targets=SleeveTargets(),
    )


def test_analyse_portfolio_is_deterministic_and_measures_allocation() -> None:
    """The engine values the book, computes sleeve drift and names concentration."""
    config = _config()
    txns = [
        _buy("INFY", AssetType.EQUITY, date(2022, 1, 1), "100", "100"),  # cost 10k
        _buy(
            "TATAMOTORS", AssetType.EQUITY, date(2022, 1, 1), "100", "100"
        ),  # cost 10k
        _buy("120503", AssetType.MF, date(2022, 1, 1), "100", "100"),  # cost 10k
    ]
    navs = parse_navall(
        "Scheme Code;A;B;Scheme Name;Net Asset Value;Date\n"
        "120503;;;PPFCF;150;05-Jul-2024\n"
    )
    equity_prices = {"INFY": Decimal("200"), "TATAMOTORS": Decimal("100")}

    first = analyse_portfolio(
        config,
        txns,
        valuation_date=date(2024, 7, 5),
        equity_prices=equity_prices,
        nav=navs,
    )
    second = analyse_portfolio(
        config,
        txns,
        valuation_date=date(2024, 7, 5),
        equity_prices=equity_prices,
        nav=navs,
    )

    # Deterministic.
    assert first.model_dump() == second.model_dump()
    # Values: INFY 20k, TATAMOTORS 10k, MF 15k -> total 45k.
    assert first.market_value == Decimal("45000")
    assert first.concentration_id == "INFY"
    # CORE = INFY(20k)+PPFCF(15k)=35k of 45k ~ 77.8%; target 70 -> +7.8pp drift.
    core = next(s for s in first.sleeves if s.sleeve is Sleeve.CORE)
    assert round(core.actual_pct, 1) == 77.8
    assert core.drift_pct > 0
    # XIRR of a book that grew 30k -> 45k over ~2.5 years is comfortably positive.
    assert first.xirr_pct is not None and first.xirr_pct > 0
    assert first.band_status in (BandStatus.WITHIN, BandStatus.ABOVE, BandStatus.BELOW)


def test_unpriced_equity_is_flagged_not_valued_at_zero_silently() -> None:
    """A holding with no price is reported as unpriced, not silently zero-valued."""
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
    txns = [_buy("INFY", AssetType.EQUITY, date(2022, 1, 1), "10", "100")]

    report = analyse_portfolio(
        config, txns, valuation_date=date(2024, 1, 1), equity_prices={}
    )

    assert "INFY" in report.unpriced
    assert report.market_value == Decimal("0")
