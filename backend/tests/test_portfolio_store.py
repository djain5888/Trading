"""Transaction store round-trip and merge de-duplication tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from app.portfolio.models import AssetType, Transaction, TxnType
from app.portfolio.store import load_transactions, merge_transactions, save_transactions


def _txn(identifier: str, on: date, amount: str) -> Transaction:
    return Transaction(
        asset_type=AssetType.EQUITY,
        identifier=identifier,
        txn_type=TxnType.BUY,
        date=on,
        units=Decimal("1"),
        price=Decimal(amount),
        amount=Decimal(amount),
    )


def test_store_round_trip_is_deterministic(tmp_path: Path) -> None:
    """Saving then loading yields the same transactions in a stable order."""
    path = tmp_path / "transactions.json"
    txns = [
        _txn("TCS", date(2023, 2, 1), "3400"),
        _txn("INFY", date(2023, 1, 1), "1500"),
    ]

    save_transactions(path, txns)
    loaded = load_transactions(path)

    assert len(loaded) == 2
    # Deterministic ordering by date.
    assert [t.identifier for t in loaded] == ["INFY", "TCS"]


def test_merge_dedupes_exact_repeats(tmp_path: Path) -> None:
    """Re-importing the same statement does not duplicate transactions."""
    existing = [_txn("INFY", date(2023, 1, 1), "1500")]
    incoming = [
        _txn("INFY", date(2023, 1, 1), "1500"),  # exact repeat -> dropped
        _txn("TCS", date(2023, 2, 1), "3400"),  # new -> kept
    ]

    merged = merge_transactions(existing, incoming)

    assert len(merged) == 2
    assert {t.identifier for t in merged} == {"INFY", "TCS"}


def test_load_missing_store_is_empty(tmp_path: Path) -> None:
    """A missing store loads as an empty list, not an error."""
    assert load_transactions(tmp_path / "absent.json") == []
