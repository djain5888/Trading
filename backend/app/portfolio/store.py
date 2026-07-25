"""On-disk persistence for canonical transactions and default paths.

Imported transactions are merged and saved as a canonical JSON array so that
``titan portfolio`` runs offline against the same deterministic data. The
storage root is ``TITAN_PORTFOLIO_DIR`` (default ``./data/portfolio``), mirroring
the ``WATCHLIST_FILE`` override pattern used elsewhere.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from app.portfolio.models import Transaction


def portfolio_dir() -> Path:
    """Return the portfolio data directory (``TITAN_PORTFOLIO_DIR`` override)."""
    return Path(os.environ.get("TITAN_PORTFOLIO_DIR", "./data/portfolio"))


def transactions_path() -> Path:
    """Return the canonical transactions store path."""
    return portfolio_dir() / "transactions.json"


def config_path() -> Path:
    """Return the portfolio config path."""
    return portfolio_dir() / "portfolio.json"


def save_transactions(path: Path, transactions: Sequence[Transaction]) -> None:
    """Persist transactions as a canonical, deterministically-ordered JSON array."""
    ordered = sorted(
        transactions,
        key=lambda t: (t.date, t.asset_type, t.identifier, t.txn_type, str(t.amount)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [t.model_dump(mode="json") for t in ordered]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_transactions(path: Path) -> list[Transaction]:
    """Load canonical transactions, returning an empty list if absent."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Transactions file {path} must be a JSON array.")
    return [Transaction(**entry) for entry in raw]


def merge_transactions(
    existing: Sequence[Transaction], incoming: Sequence[Transaction]
) -> list[Transaction]:
    """Merge transaction sets, de-duplicating exact repeats deterministically."""
    seen: set[tuple[str, ...]] = set()
    merged: list[Transaction] = []
    for txn in [*existing, *incoming]:
        signature = (
            txn.asset_type.value,
            txn.identifier,
            txn.txn_type.value,
            txn.date.isoformat(),
            str(txn.units),
            str(txn.amount),
            str(txn.charges),
        )
        if signature in seen:
            continue
        seen.add(signature)
        merged.append(txn)
    return merged
