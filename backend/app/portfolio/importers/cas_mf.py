"""CAMS / KFintech Consolidated Account Statement (CAS) mutual-fund importer.

INVESTIGATED FORMAT — which is practical? The CAS is distributed as a
*password-protected PDF*. Parsing PDF text positionally is brittle and the
layout differs between CAMS and KFintech, so the practical, deterministic path
is a **transaction-level CSV**: either the CAMS "Statement of transactions"
CSV, the CSV that mfcentral/CAS-parser tools emit, or a CSV the user exports
from the PDF. This importer therefore parses that CSV shape (and callers may
run a PDF->CSV step first). Documented / observed columns:

    Scheme Code (AMFI) | Scheme Name | Date | Transaction Type |
    Amount | Units | NAV | Folio

Scheme code is preferred as the identifier (stable, joins to AMFI NAV); if
absent, the scheme name is used. Header matching is tolerant of spelling and
order; unusable rows are skipped with a reason.
"""

from __future__ import annotations

import csv
from decimal import Decimal

from app.core.logging import get_logger
from app.portfolio.importers.base import (
    ImportResult,
    SkippedRow,
    match_columns,
    parse_amount,
    parse_date,
    parse_txn_type,
)
from app.portfolio.models import AssetType, Transaction, TxnType

logger = get_logger(__name__)

_SOURCE = "cas_mf"

_FIELDS: dict[str, tuple[str, ...]] = {
    "scheme_code": ("scheme code", "amfi", "amfi code", "code"),
    "scheme_name": ("scheme name", "scheme", "fund name", "fund"),
    "date": ("date", "transaction date", "posted date"),
    "txn_type": ("transaction type", "type", "description", "txn type"),
    "amount": ("amount", "value"),
    "units": ("units", "unit"),
    "nav": ("nav", "price", "purchase price"),
}


def parse_cas_csv(text: str) -> ImportResult:
    """Parse a CAMS/KFintech transaction CSV into canonical MF transactions."""
    rows = list(csv.reader(text.splitlines()))
    header_index = _find_header(rows)
    if header_index is None:
        return ImportResult(
            source=_SOURCE,
            transactions=(),
            skipped=(SkippedRow(row=1, reason="No recognisable CAS header row."),),
        )

    header = rows[header_index]
    columns = match_columns(header, _FIELDS)
    mapping = {header[idx]: field for field, idx in columns.items()}
    if "scheme_code" not in columns and "scheme_name" not in columns:
        return ImportResult(
            source=_SOURCE,
            transactions=(),
            skipped=(
                SkippedRow(
                    row=header_index + 1, reason="No scheme code or name column."
                ),
            ),
            mapping=mapping,
        )
    if "date" not in columns or "txn_type" not in columns:
        return ImportResult(
            source=_SOURCE,
            transactions=(),
            skipped=(
                SkippedRow(row=header_index + 1, reason="Missing date/type column."),
            ),
            mapping=mapping,
        )

    transactions, skipped = _parse_rows(rows, header_index, columns)
    logger.info("CAS import: %d parsed, %d skipped.", len(transactions), len(skipped))
    return ImportResult(
        source=_SOURCE,
        transactions=tuple(transactions),
        skipped=tuple(skipped),
        mapping=mapping,
    )


def _find_header(rows: list[list[str]]) -> int | None:
    """Return the index of the first row that looks like a CAS transaction header."""
    for index, row in enumerate(rows):
        joined = " ".join(cell.strip().lower() for cell in row)
        if ("scheme" in joined or "amfi" in joined or "fund" in joined) and (
            "date" in joined
        ):
            return index
    return None


def _cell(row: list[str], columns: dict[str, int], field: str) -> str | None:
    """Return the trimmed value for ``field`` in ``row``, or ``None``."""
    index = columns.get(field)
    if index is None or index >= len(row):
        return None
    return row[index].strip()


def _parse_rows(
    rows: list[list[str]], header_index: int, columns: dict[str, int]
) -> tuple[list[Transaction], list[SkippedRow]]:
    """Parse the data rows following the header."""
    transactions: list[Transaction] = []
    skipped: list[SkippedRow] = []
    for offset, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        if not any(cell.strip() for cell in row):
            continue
        result = _parse_one(row, columns)
        if isinstance(result, Transaction):
            transactions.append(result)
        else:
            skipped.append(
                SkippedRow(row=offset, reason=result, detail=",".join(row)[:80])
            )
    return transactions, skipped


def _parse_one(row: list[str], columns: dict[str, int]) -> Transaction | str:
    """Parse one CAS data row into a Transaction, or return a skip reason."""
    code = _cell(row, columns, "scheme_code")
    scheme = _cell(row, columns, "scheme_name")
    identifier = code or scheme
    when = parse_date(_cell(row, columns, "date"))
    txn_type = parse_txn_type(_cell(row, columns, "txn_type"))
    amount = parse_amount(_cell(row, columns, "amount"))
    units = parse_amount(_cell(row, columns, "units"))
    nav = parse_amount(_cell(row, columns, "nav"))

    if not identifier:
        return "missing scheme code and name"
    if when is None:
        return "unparseable date"
    if txn_type is None:
        return "unknown transaction type"

    if txn_type in (TxnType.BUY, TxnType.SELL):
        if units is None or units <= 0:
            return "missing/invalid units"
        if nav is None and amount is None:
            return "missing both NAV and amount"
        if amount is None and nav is not None:
            amount = (nav * units).quantize(Decimal("0.01"))
        if nav is None and amount is not None:
            nav = (amount / units).quantize(Decimal("0.0001"))
    else:  # DIVIDEND / CHARGE — cash only
        if amount is None or amount <= 0:
            return "missing amount for cash transaction"
        units = Decimal(0)
        nav = Decimal(0)

    try:
        return Transaction(
            asset_type=AssetType.MF,
            identifier=str(identifier).strip().upper(),
            name=scheme or str(identifier),
            txn_type=txn_type,
            date=when,
            units=units or Decimal(0),
            price=nav or Decimal(0),
            amount=amount or Decimal(0),
            charges=Decimal(0),
            source=_SOURCE,
        )
    except ValueError as exc:
        return f"validation error: {exc}"
