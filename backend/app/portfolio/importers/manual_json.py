"""Manual JSON transaction importer — the fallback the CSV parsers can't cover.

Use it for odd corporate actions, off-market transfers or gifted holdings.
Format: a JSON array of transaction objects using the canonical
:class:`~app.portfolio.models.Transaction` field names, e.g.::

    [
      {"asset_type": "EQUITY", "identifier": "INFY", "txn_type": "BUY",
       "date": "2022-04-01", "units": "10", "price": "1500", "amount": "15000"}
    ]

Each object is validated individually; a malformed object is skipped with its
index and the validation error, never aborting the whole import.
"""

from __future__ import annotations

import json

from app.core.logging import get_logger
from app.portfolio.importers.base import ImportResult, SkippedRow
from app.portfolio.models import Transaction

logger = get_logger(__name__)

_SOURCE = "manual_json"


def parse_manual_json(text: str) -> ImportResult:
    """Parse a manual JSON transaction array into canonical transactions."""
    try:
        raw = json.loads(text)
    except ValueError as exc:
        return ImportResult(
            source=_SOURCE,
            transactions=(),
            skipped=(SkippedRow(row=1, reason=f"Invalid JSON: {exc}"),),
        )
    if not isinstance(raw, list):
        return ImportResult(
            source=_SOURCE,
            transactions=(),
            skipped=(SkippedRow(row=1, reason="Top level must be a JSON array."),),
        )

    transactions: list[Transaction] = []
    skipped: list[SkippedRow] = []
    for index, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            skipped.append(SkippedRow(row=index, reason="Entry is not an object."))
            continue
        try:
            transactions.append(
                Transaction(**{**entry, "source": entry.get("source", _SOURCE)})
            )
        except (ValueError, TypeError) as exc:
            skipped.append(
                SkippedRow(
                    row=index, reason=f"validation error: {exc}", detail=str(entry)[:80]
                )
            )
    logger.info(
        "Manual JSON import: %d parsed, %d skipped.", len(transactions), len(skipped)
    )
    return ImportResult(
        source=_SOURCE,
        transactions=tuple(transactions),
        skipped=tuple(skipped),
    )
