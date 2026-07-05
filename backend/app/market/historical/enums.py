"""Enumerations for the historical-data domain."""

from __future__ import annotations

from enum import StrEnum


class IssueType(StrEnum):
    """Categories of data-quality issue detected during import/validation."""

    DUPLICATE = "duplicate"
    GAP = "gap"
    INVALID_OHLC = "invalid_ohlc"
    NON_POSITIVE_PRICE = "non_positive_price"
    NEGATIVE_VOLUME = "negative_volume"
    INVALID_OPEN_INTEREST = "invalid_open_interest"
    UNORDERED = "unordered"
    OVERLAP = "overlap"
    NAIVE_TIMESTAMP = "naive_timestamp"


class DuplicatePolicy(StrEnum):
    """How to treat candles that already exist in the repository."""

    SKIP = "skip"
    """Leave the stored candle untouched and count it as a duplicate."""
    UPDATE = "update"
    """Overwrite the stored candle and count it as repaired."""
