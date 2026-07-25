"""Groww equity transaction CSV importer.

INVESTIGATED FORMAT (Groww "Stocks -> Reports -> Transaction/P&L statement"
CSV export). The export is a flat CSV with a single header row. Observed /
documented columns (spelling varies slightly across export versions):

    Stock Name, ISIN, Trade Date, Exchange, Segment, Trade Type,
    Quantity, Price, Amount, Brokerage/Charges, Order ID

Only a handful are needed for portfolio maths, so parsing is header-driven and
tolerant of column order and minor spelling differences (``Symbol`` vs
``Stock Name``, ``Trade Type`` vs ``Type``, ``Trade Date`` vs ``Date``). Rows
that lack the essentials (date, symbol, type, quantity, price/amount) are
skipped with a reason rather than guessed.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
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

_SOURCE = "groww_equity"

#: canonical field -> accepted header spellings (lower-cased substrings).
_FIELDS: dict[str, tuple[str, ...]] = {
    "symbol": ("symbol", "stock name", "stock", "scrip", "company"),
    "date": ("trade date", "date"),
    "txn_type": ("trade type", "transaction type", "type", "buy/sell"),
    "quantity": ("quantity", "qty", "units"),
    "price": ("price", "avg price", "rate"),
    "amount": ("amount", "trade value", "value", "net amount"),
    "charges": ("charges", "brokerage", "taxes", "total charges"),
}


def parse_groww_equity_csv(text: str) -> ImportResult:
    """Parse a Groww equity transaction CSV into canonical transactions."""
    rows = list(csv.reader(text.splitlines()))
    header_index = _find_header(rows)
    if header_index is None:
        return ImportResult(
            source=_SOURCE,
            transactions=(),
            skipped=(SkippedRow(row=1, reason="No recognisable header row."),),
        )

    header = rows[header_index]
    columns = match_columns(header, _FIELDS)
    mapping = {header[idx]: field for field, idx in columns.items()}
    required = ("symbol", "date", "txn_type", "quantity")
    missing = [field for field in required if field not in columns]
    if missing:
        return ImportResult(
            source=_SOURCE,
            transactions=(),
            skipped=(
                SkippedRow(row=header_index + 1, reason=f"Missing columns: {missing}"),
            ),
            mapping=mapping,
        )

    transactions, skipped = _parse_rows(rows, header_index, columns)
    logger.info(
        "Groww equity import: %d parsed, %d skipped.", len(transactions), len(skipped)
    )
    return ImportResult(
        source=_SOURCE,
        transactions=tuple(transactions),
        skipped=tuple(skipped),
        mapping=mapping,
    )


def _find_header(rows: list[list[str]]) -> int | None:
    """Return the index of the first row that looks like a Groww header."""
    for index, row in enumerate(rows):
        lowered = [cell.strip().lower() for cell in row]
        joined = " ".join(lowered)
        if ("stock" in joined or "symbol" in joined or "scrip" in joined) and (
            "trade type" in joined or "type" in joined or "buy/sell" in joined
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
            continue  # blank line
        result = _parse_one(row, columns)
        if isinstance(result, Transaction):
            transactions.append(result)
        else:
            skipped.append(
                SkippedRow(row=offset, reason=result, detail=",".join(row)[:80])
            )
    return transactions, skipped


def _parse_one(row: list[str], columns: dict[str, int]) -> Transaction | str:
    """Parse one data row into a Transaction, or return a skip reason string."""
    symbol = _cell(row, columns, "symbol")
    when = parse_date(_cell(row, columns, "date"))
    txn_type = parse_txn_type(_cell(row, columns, "txn_type"))
    quantity = parse_amount(_cell(row, columns, "quantity"))
    price = parse_amount(_cell(row, columns, "price"))
    amount = parse_amount(_cell(row, columns, "amount"))
    charges = parse_amount(_cell(row, columns, "charges")) or Decimal(0)

    if not symbol:
        return "missing symbol"
    if when is None:
        return "unparseable date"
    if txn_type is None:
        return "unknown transaction type"
    if txn_type not in (TxnType.BUY, TxnType.SELL):
        return f"unsupported equity transaction type {txn_type}"
    if quantity is None or quantity <= 0:
        return "missing/invalid quantity"
    if price is None and amount is None:
        return "missing both price and amount"
    if amount is None and price is not None:
        amount = (price * quantity).quantize(Decimal("0.01"))
    if price is None and amount is not None:
        price = (amount / quantity).quantize(Decimal("0.0001"))

    try:
        return Transaction(
            asset_type=AssetType.EQUITY,
            identifier=symbol.upper(),
            name=symbol,
            txn_type=txn_type,
            date=when,
            units=quantity,
            price=price or Decimal(0),
            amount=amount or Decimal(0),
            charges=charges,
            source=_SOURCE,
        )
    except ValueError as exc:
        return f"validation error: {exc}"


def parse_groww_equity_lines(lines: Iterable[str]) -> ImportResult:
    """Convenience wrapper: parse from an iterable of raw CSV lines."""
    return parse_groww_equity_csv("\n".join(lines))
