"""Importer tests: Groww equity CSV, CAS CSV, manual JSON, and AMFI NAV.

Includes malformed rows (skipped with reasons) and header-order tolerance.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.portfolio.importers.cas_mf import parse_cas_csv
from app.portfolio.importers.groww_equity import parse_groww_equity_csv
from app.portfolio.importers.manual_json import parse_manual_json
from app.portfolio.models import AssetType, TxnType
from app.portfolio.nav import parse_navall

# -- Groww equity CSV ------------------------------------------------------


def test_groww_equity_parses_and_skips_bad_rows() -> None:
    """Good rows parse; malformed rows are skipped with reasons."""
    csv = (
        "Stock Name,Trade Date,Trade Type,Quantity,Price,Amount,Charges\n"
        "INFY,2022-04-01,BUY,10,1500,15000,20\n"
        "TCS,01-05-2022,SELL,5,3400,17000,15\n"
        ",2022-04-01,BUY,10,1500,15000,0\n"  # missing symbol -> skip
        "WIPRO,not-a-date,BUY,10,400,4000,0\n"  # bad date -> skip
        "HDFCBANK,2022-04-01,GIFT,10,1500,15000,0\n"  # unknown type -> skip
    )

    result = parse_groww_equity_csv(csv)

    assert result.parsed == 2
    assert len(result.skipped) == 3
    infy = result.transactions[0]
    assert infy.identifier == "INFY"
    assert infy.txn_type is TxnType.BUY
    assert infy.units == Decimal("10")
    assert infy.amount == Decimal("15000")
    assert infy.charges == Decimal("20")
    reasons = {row.reason for row in result.skipped}
    assert any("symbol" in r for r in reasons)
    assert any("date" in r for r in reasons)


def test_groww_equity_tolerates_header_order_and_derives_amount() -> None:
    """Columns may be reordered/renamed; amount derives from price*qty if absent."""
    csv = "Type,Symbol,Qty,Rate,Date\n" "buy,RELIANCE,4,2500,2023-01-10\n"

    result = parse_groww_equity_csv(csv)

    assert result.parsed == 1
    txn = result.transactions[0]
    assert txn.identifier == "RELIANCE"
    assert txn.amount == Decimal("10000.00")  # 4 * 2500 derived


# -- CAS mutual-fund CSV ---------------------------------------------------


def test_cas_parses_purchase_and_redemption() -> None:
    """CAS purchases and redemptions map to BUY/SELL with NAV and units."""
    csv = (
        "Scheme Code,Scheme Name,Date,Transaction Type,Amount,Units,NAV\n"
        "120503,Parag Parikh Flexi Cap,2023-01-05,Purchase,10000,200,50\n"
        "120503,Parag Parikh Flexi Cap,2023-06-05,Redemption,6000,100,60\n"
        "120503,Parag Parikh Flexi Cap,2023-07-05,Dividend,500,,\n"
    )

    result = parse_cas_csv(csv)

    assert result.parsed == 3
    buy, sell, div = result.transactions
    assert buy.asset_type is AssetType.MF and buy.txn_type is TxnType.BUY
    assert buy.units == Decimal("200") and buy.price == Decimal("50")
    assert sell.txn_type is TxnType.SELL and sell.amount == Decimal("6000")
    assert div.txn_type is TxnType.DIVIDEND and div.amount == Decimal("500")


def test_cas_skips_rows_missing_units() -> None:
    """A purchase with no units and no amount cannot be valued -> skipped."""
    csv = (
        "Scheme Code,Scheme Name,Date,Transaction Type,Amount,Units,NAV\n"
        "120503,X,2023-01-05,Purchase,,,\n"
    )

    result = parse_cas_csv(csv)

    assert result.parsed == 0
    assert len(result.skipped) == 1


# -- Manual JSON -----------------------------------------------------------


def test_manual_json_parses_and_flags_bad_entries() -> None:
    """Valid objects parse; a malformed object is skipped with its index."""
    text = (
        '[{"asset_type": "EQUITY", "identifier": "INFY", "txn_type": "BUY", '
        '"date": "2022-04-01", "units": "10", "price": "1500", "amount": "15000"}, '
        '{"asset_type": "EQUITY", "identifier": "BAD", "txn_type": "BUY", '
        '"date": "2022-04-01", "units": "0", "amount": "0"}]'
    )

    result = parse_manual_json(text)

    assert result.parsed == 1
    assert len(result.skipped) == 1  # units<=0 fails validation
    assert result.skipped[0].row == 2


# -- AMFI NAV --------------------------------------------------------------


def test_parse_navall_extracts_entries_and_as_of() -> None:
    """The AMFI file parses into scheme-code-keyed entries with a latest date."""
    text = (
        "Scheme Code;ISIN Div Payout;ISIN Div Reinvestment;Scheme Name;"
        "Net Asset Value;Date\n"
        "\n"
        "Open Ended Schemes ( Equity Scheme )\n"
        "\n"
        "Parag Parikh Mutual Fund\n"
        "120503;INF879O01027;INF879O01035;Parag Parikh Flexi Cap Fund;65.4321;"
        "05-Jul-2024\n"
        "119551;INF209K01157;INF209K01165;UTI Nifty 50 Index Fund;120.5;"
        "05-Jul-2024\n"
        "999999;;;Dud Scheme;N.A.;05-Jul-2024\n"  # no NAV -> skipped
    )

    snapshot = parse_navall(text)

    assert snapshot.as_of == date(2024, 7, 5)
    assert snapshot.get("120503") is not None
    assert snapshot.get("120503").nav == Decimal("65.4321")
    assert snapshot.get("999999") is None  # unparseable NAV dropped
    assert snapshot.is_stale(date(2024, 7, 20)) is True
    assert snapshot.is_stale(date(2024, 7, 7)) is False
