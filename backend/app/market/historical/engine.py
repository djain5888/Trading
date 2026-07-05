"""The historical-data engine use case.

Orchestrates validation and persistence of OHLCV candles: it validates each
candle, removes in-batch duplicates, detects series-level issues, applies the
requested duplicate policy against the store, and returns a data-quality report.
It contains no storage or trading logic — only coordination.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from app.core.logging import get_logger
from app.market.historical.enums import DuplicatePolicy, IssueType
from app.market.historical.models import Candle, SeriesKey
from app.market.historical.report import DataQualityReport, ValidationIssue
from app.market.historical.repository import CandleRepository
from app.market.historical.validation import ValidationEngine

logger = get_logger(__name__)


class HistoricalDataEngine:
    """Stores, updates, validates and retrieves historical candles."""

    def __init__(
        self, repository: CandleRepository, validator: ValidationEngine
    ) -> None:
        """Initialise the engine.

        Args:
            repository: The persistence port for candles.
            validator: The validation engine.
        """
        self._repository = repository
        self._validator = validator

    async def import_candles(
        self,
        candles: Sequence[Candle],
        *,
        on_duplicate: DuplicatePolicy = DuplicatePolicy.SKIP,
    ) -> DataQualityReport:
        """Bulk-import a single series of candles.

        Args:
            candles: Candles belonging to one ``(symbol, exchange, interval)``.
            on_duplicate: Whether to skip or update candles that already exist.

        Returns:
            A :class:`DataQualityReport` describing the outcome.

        Raises:
            ValueError: If the candles do not all share one series key.
        """
        if not candles:
            return DataQualityReport.empty()
        self._require_single_series(candles)

        issues: list[ValidationIssue] = []
        valid: list[Candle] = []
        seen: set[datetime] = set()
        invalid = 0
        batch_duplicates = 0

        for candle in candles:
            candle_issues = self._validator.validate_candle(candle)
            if candle_issues:
                invalid += 1
                issues.extend(candle_issues)
                continue
            if candle.timestamp in seen:
                batch_duplicates += 1
                issues.append(
                    ValidationIssue(
                        issue_type=IssueType.DUPLICATE,
                        message=f"Duplicate candle at {candle.timestamp.isoformat()}.",
                        timestamp=candle.timestamp,
                    )
                )
                continue
            seen.add(candle.timestamp)
            valid.append(candle)

        interval = candles[0].interval
        issues.extend(self._validator.detect_unordered(valid))
        ordered = sorted(valid, key=lambda candle: candle.timestamp)
        issues.extend(self._validator.detect_overlaps(ordered, interval))
        gap_issues, gaps_found = self._validator.detect_gaps(ordered, interval)
        issues.extend(gap_issues)

        imported, repaired, store_duplicates = await self._persist(
            ordered, on_duplicate
        )

        skipped = (
            invalid
            + batch_duplicates
            + (store_duplicates if on_duplicate is DuplicatePolicy.SKIP else 0)
        )
        report = DataQualityReport(
            imported=imported,
            skipped=skipped,
            duplicates=batch_duplicates + store_duplicates,
            repaired=repaired,
            invalid=invalid,
            gaps_found=gaps_found,
            issues=tuple(issues),
        )
        logger.info(
            "Imported series %s/%s/%s: %s",
            candles[0].symbol,
            candles[0].exchange,
            interval,
            report.model_dump(exclude={"issues"}),
        )
        return report

    async def _persist(
        self, ordered: Sequence[Candle], on_duplicate: DuplicatePolicy
    ) -> tuple[int, int, int]:
        """Persist candles according to the duplicate policy.

        Returns:
            A tuple of ``(imported, repaired, store_duplicates)``.
        """
        to_insert: list[Candle] = []
        repaired = 0
        store_duplicates = 0
        for candle in ordered:
            if await self._repository.exists(candle.series_key, candle.timestamp):
                if on_duplicate is DuplicatePolicy.UPDATE:
                    await self._repository.save(candle)
                    repaired += 1
                else:
                    store_duplicates += 1
                continue
            to_insert.append(candle)
        if to_insert:
            await self._repository.save_many(to_insert)
        return len(to_insert), repaired, store_duplicates

    @staticmethod
    def _require_single_series(candles: Sequence[Candle]) -> None:
        """Ensure all candles share one series key.

        Raises:
            ValueError: If more than one series is present.
        """
        keys = {candle.series_key for candle in candles}
        if len(keys) > 1:
            raise ValueError("All candles must belong to a single series.")

    # -- Retrieval ---------------------------------------------------------

    async def get_candle(self, key: SeriesKey, timestamp: datetime) -> Candle | None:
        """Return the candle at ``timestamp`` for ``key``."""
        return await self._repository.get(key, timestamp)

    async def get_candles(
        self, key: SeriesKey, start: datetime, end: datetime
    ) -> list[Candle]:
        """Return the ordered candles for ``key`` within ``[start, end]``."""
        return await self._repository.get_range(key, start, end)

    async def get_latest(self, key: SeriesKey) -> Candle | None:
        """Return the most recent candle for ``key``."""
        return await self._repository.latest(key)

    async def count(self, key: SeriesKey) -> int:
        """Return the number of stored candles for ``key``."""
        return await self._repository.count(key)

    async def delete_candle(self, key: SeriesKey, timestamp: datetime) -> bool:
        """Delete the candle at ``timestamp``; return whether one was removed."""
        return await self._repository.delete(key, timestamp)
