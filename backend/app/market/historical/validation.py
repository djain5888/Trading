"""The validation engine: detects data-quality issues in candles and series.

Per-candle checks reuse the shared invariants; series checks operate on the
fixed grid implied by the interval. The engine is pure and stateless.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from app.market.enums import Interval
from app.market.historical import invariants
from app.market.historical.enums import IssueType
from app.market.historical.intervals import interval_delta
from app.market.historical.models import Candle
from app.market.historical.report import ValidationIssue


class ValidationEngine:
    """Detects duplicate, gapped, unordered, overlapping and invalid candles."""

    def validate_candle(self, candle: Candle) -> list[ValidationIssue]:
        """Return issues for a single candle's intrinsic invariants.

        Args:
            candle: The candle to check.

        Returns:
            A list of issues (empty when the candle is valid).
        """
        issues: list[ValidationIssue] = []
        checks = (
            (
                IssueType.NAIVE_TIMESTAMP,
                invariants.check_timestamp_aware(candle.timestamp),
            ),
            (
                IssueType.NON_POSITIVE_PRICE,
                invariants.check_positive_prices(
                    candle.open, candle.high, candle.low, candle.close
                ),
            ),
            (
                IssueType.INVALID_OHLC,
                invariants.check_ohlc_consistency(
                    candle.open, candle.high, candle.low, candle.close
                ),
            ),
            (IssueType.NEGATIVE_VOLUME, invariants.check_volume(candle.volume)),
            (
                IssueType.INVALID_OPEN_INTEREST,
                invariants.check_open_interest(candle.open_interest),
            ),
        )
        for issue_type, error in checks:
            if error is not None:
                issues.append(
                    ValidationIssue(
                        issue_type=issue_type,
                        message=error,
                        timestamp=candle.timestamp,
                    )
                )
        return issues

    def detect_duplicates(self, candles: Sequence[Candle]) -> list[ValidationIssue]:
        """Return an issue for each candle repeating an earlier timestamp."""
        seen: set[object] = set()
        issues: list[ValidationIssue] = []
        for candle in candles:
            if candle.timestamp in seen:
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.DUPLICATE,
                        message=f"Duplicate candle at {candle.timestamp.isoformat()}.",
                        timestamp=candle.timestamp,
                    )
                )
            else:
                seen.add(candle.timestamp)
        return issues

    def detect_unordered(self, candles: Sequence[Candle]) -> list[ValidationIssue]:
        """Return an issue for each candle that breaks chronological ordering."""
        issues: list[ValidationIssue] = []
        for previous, current in zip(candles, candles[1:], strict=False):
            if current.timestamp < previous.timestamp:
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.UNORDERED,
                        message=(
                            f"Candle at {current.timestamp.isoformat()} precedes "
                            f"{previous.timestamp.isoformat()}."
                        ),
                        timestamp=current.timestamp,
                    )
                )
        return issues

    def detect_overlaps(
        self, ordered: Sequence[Candle], interval: Interval
    ) -> list[ValidationIssue]:
        """Return an issue for consecutive candles closer than one interval.

        Args:
            ordered: Chronologically ordered candles.
            interval: The series interval.

        Returns:
            Overlap issues (empty for calendar-variable intervals).
        """
        delta = interval_delta(interval)
        if delta is None:
            return []
        issues: list[ValidationIssue] = []
        for previous, current in zip(ordered, ordered[1:], strict=False):
            distance = current.timestamp - previous.timestamp
            if not timedelta(0) < distance < delta:
                continue
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.OVERLAP,
                    message=(
                        f"Candle at {current.timestamp.isoformat()} overlaps the "
                        f"previous candle for interval {interval.value}."
                    ),
                    timestamp=current.timestamp,
                )
            )
        return issues

    def detect_gaps(
        self, ordered: Sequence[Candle], interval: Interval
    ) -> tuple[list[ValidationIssue], int]:
        """Detect missing candles on the interval grid.

        Args:
            ordered: Chronologically ordered, de-duplicated candles.
            interval: The series interval.

        Returns:
            A tuple of the gap issues and the total number of missing slots.
        """
        delta = interval_delta(interval)
        if delta is None:
            return [], 0
        issues: list[ValidationIssue] = []
        total_missing = 0
        for previous, current in zip(ordered, ordered[1:], strict=False):
            distance = current.timestamp - previous.timestamp
            if distance <= delta:
                continue
            missing = int(distance / delta) - 1
            if missing <= 0:
                continue
            total_missing += missing
            issues.append(
                ValidationIssue(
                    issue_type=IssueType.GAP,
                    message=(
                        f"{missing} missing candle(s) between "
                        f"{previous.timestamp.isoformat()} and "
                        f"{current.timestamp.isoformat()}."
                    ),
                    timestamp=current.timestamp,
                )
            )
        return issues, total_missing
