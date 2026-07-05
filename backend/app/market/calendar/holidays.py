"""Holiday providers and file loaders.

:class:`HolidayProvider` is the replaceable seam: any source (file, database,
vendor API) that can answer "which dates are holidays for this exchange" can
satisfy it. :class:`StaticHolidayProvider` is an in-memory implementation with
JSON and CSV loaders.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.market.enums import Exchange

#: CSV column headers recognised by :meth:`StaticHolidayProvider.from_csv`.
_CSV_DATE_COLUMN = "date"
_CSV_EXCHANGE_COLUMN = "exchange"


@runtime_checkable
class HolidayProvider(Protocol):
    """Supplies the set of trading holidays for an exchange."""

    def holidays(self, exchange: Exchange) -> frozenset[date]:
        """Return the holiday dates for ``exchange`` (empty if none)."""
        ...


class StaticHolidayProvider:
    """A holiday provider backed by in-memory data."""

    def __init__(self, holidays_by_exchange: Mapping[Exchange, Iterable[date]]) -> None:
        """Initialise the provider.

        Args:
            holidays_by_exchange: Mapping of exchange to its holiday dates.
        """
        self._holidays: dict[Exchange, frozenset[date]] = {
            exchange: frozenset(dates)
            for exchange, dates in holidays_by_exchange.items()
        }

    def holidays(self, exchange: Exchange) -> frozenset[date]:
        """Return the holiday dates for ``exchange``.

        Args:
            exchange: The exchange to look up.

        Returns:
            The configured holidays, or an empty set if none are registered.
        """
        return self._holidays.get(exchange, frozenset())

    @classmethod
    def from_json(cls, path: str | Path) -> StaticHolidayProvider:
        """Load holidays from a JSON file.

        The file maps exchange codes to ISO date strings, e.g.::

            {"NSE": ["2025-01-26", "2025-08-15"]}

        Args:
            path: Path to the JSON file.

        Returns:
            A provider populated from the file.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        parsed: dict[Exchange, list[date]] = {}
        for code, values in raw.items():
            exchange = Exchange(code.upper())
            parsed.setdefault(exchange, []).extend(
                date.fromisoformat(value) for value in values
            )
        return cls(parsed)

    @classmethod
    def from_csv(cls, path: str | Path) -> StaticHolidayProvider:
        """Load holidays from a CSV file.

        The file must have a header with ``date`` and ``exchange`` columns::

            date,exchange
            2025-01-26,NSE

        Args:
            path: Path to the CSV file.

        Returns:
            A provider populated from the file.

        Raises:
            ValueError: If required columns are missing.
        """
        parsed: dict[Exchange, list[date]] = {}
        with Path(path).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if (
                reader.fieldnames is None
                or _CSV_DATE_COLUMN not in reader.fieldnames
                or _CSV_EXCHANGE_COLUMN not in reader.fieldnames
            ):
                raise ValueError(
                    f"CSV must contain '{_CSV_DATE_COLUMN}' and "
                    f"'{_CSV_EXCHANGE_COLUMN}' columns."
                )
            for row in reader:
                exchange = Exchange(row[_CSV_EXCHANGE_COLUMN].strip().upper())
                holiday = date.fromisoformat(row[_CSV_DATE_COLUMN].strip())
                parsed.setdefault(exchange, []).append(holiday)
        return cls(parsed)
