"""Shared importer scaffolding: canonical result types and tolerant parsers.

Every importer normalises a statement into :class:`~app.portfolio.models.Transaction`
rows and returns an :class:`ImportResult` that reports exactly what parsed and
what was skipped (and why), so a user can trust — and audit — the ingestion.

The real-world statements (Groww equity CSV, CAMS/KFintech CAS) vary in header
spelling and column order, so parsing is header-driven: each canonical field
names several accepted header spellings and the column is matched
case-insensitively. This keeps the parsers working against the actual export
rather than a guessed fixed layout.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, Field

from app.portfolio.models import Transaction, TxnType

#: Date formats seen across Groww / CAMS / KFintech exports.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d-%b-%Y",
    "%d-%b-%y",
    "%d %b %Y",
    "%d-%B-%Y",
    "%Y/%m/%d",
)

#: Transaction-type synonyms -> canonical kind (matched on a lower-cased token).
_TXN_SYNONYMS: dict[str, TxnType] = {
    "buy": TxnType.BUY,
    "b": TxnType.BUY,
    "purchase": TxnType.BUY,
    "purchase-sip": TxnType.BUY,
    "sip": TxnType.BUY,
    "credit": TxnType.BUY,
    "systematic": TxnType.BUY,
    "sell": TxnType.SELL,
    "s": TxnType.SELL,
    "redemption": TxnType.SELL,
    "redeem": TxnType.SELL,
    "debit": TxnType.SELL,
    "switch-out": TxnType.SELL,
    "dividend": TxnType.DIVIDEND,
    "idcw": TxnType.DIVIDEND,
    "payout": TxnType.DIVIDEND,
    "charge": TxnType.CHARGE,
    "fee": TxnType.CHARGE,
    "charges": TxnType.CHARGE,
}


class SkippedRow(BaseModel):
    """One row that could not be parsed, with a human-readable reason."""

    model_config = ConfigDict(frozen=True)

    row: int = Field(description="1-based source row number.")
    reason: str = Field(description="Why the row was skipped.")
    detail: str = Field(default="", description="A short excerpt of the raw row.")


class ImportResult(BaseModel):
    """The outcome of one import: the canonical rows plus a skip audit."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(description="Importer name.")
    transactions: tuple[Transaction, ...] = Field(description="Parsed transactions.")
    skipped: tuple[SkippedRow, ...] = Field(
        default=(), description="Rows skipped, with reasons."
    )
    mapping: dict[str, str] = Field(
        default_factory=dict, description="Detected header -> canonical field."
    )

    @property
    def parsed(self) -> int:
        """Return the number of transactions parsed."""
        return len(self.transactions)


def parse_amount(raw: str | None) -> Decimal | None:
    """Parse a money/number cell, tolerating commas, currency signs and blanks."""
    if raw is None:
        return None
    text = raw.strip().replace(",", "").replace("₹", "").replace("Rs.", "")
    text = text.replace("Rs", "").replace("(", "-").replace(")", "").strip()
    if not text or text in {"-", "--"}:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def parse_date(raw: str | None) -> date | None:
    """Parse a date cell across the known statement formats."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_txn_type(raw: str | None) -> TxnType | None:
    """Map a free-text transaction-type cell to a canonical kind."""
    if raw is None:
        return None
    token = raw.strip().lower()
    if not token:
        return None
    if token in _TXN_SYNONYMS:
        return _TXN_SYNONYMS[token]
    # Fall back to a substring scan (e.g. "Purchase - SIP", "Sell / Redemption").
    for needle, kind in _TXN_SYNONYMS.items():
        if needle in token:
            return kind
    return None


def match_columns(
    header: list[str], fields: dict[str, tuple[str, ...]]
) -> dict[str, int]:
    """Map each canonical field to a column index by accepted header spellings.

    Args:
        header: The raw header cells.
        fields: canonical field -> accepted header spellings (lower-cased).

    Returns:
        canonical field -> column index for every field that matched a header.
    """
    normalised = [cell.strip().lower() for cell in header]
    resolved: dict[str, int] = {}
    for field, spellings in fields.items():
        for index, cell in enumerate(normalised):
            if cell in spellings or any(s in cell for s in spellings):
                resolved[field] = index
                break
    return resolved
