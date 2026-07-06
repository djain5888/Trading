"""The indicator calculator interface and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

from app.indicators.calculations import Aligned, is_finite
from app.indicators.exceptions import (
    ComputationError,
    InsufficientHistoryError,
    InvalidPeriodError,
)
from app.indicators.models import IndicatorResult, PriceSeries


class IndicatorCalculator(ABC):
    """Computes one indicator over a price series.

    Calculators are configured with their parameters at construction and are
    independently pluggable: adding an indicator means registering a new
    subclass, with no changes to the engine.
    """

    name: ClassVar[str]
    #: Whether the engine may compute this indicator incrementally. Only pure
    #: rolling-window indicators (no recursion/cumulation) qualify.
    supports_incremental: ClassVar[bool] = False
    #: Minimum candles required to produce one value. Set by subclasses.
    _lookback: int = 1

    @property
    def lookback(self) -> int:
        """Minimum number of candles required to produce one value."""
        return self._lookback

    @property
    def warmup(self) -> int:
        """Candles consumed before the first value (``lookback - 1``)."""
        return self.lookback - 1

    def calculate(self, series: PriceSeries) -> list[IndicatorResult]:
        """Validate history and compute the indicator series.

        Args:
            series: The price series.

        Returns:
            One result per index where the indicator is defined.

        Raises:
            InsufficientHistoryError: If there are too few candles.
            ComputationError: If a computed value is non-finite.
        """
        if len(series) < self.lookback:
            raise InsufficientHistoryError(
                f"{self.name} needs >= {self.lookback} candles, " f"got {len(series)}."
            )
        return self._compute(series)

    @abstractmethod
    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        """Compute the indicator (history already validated)."""

    def _results(
        self,
        series: PriceSeries,
        primary: Aligned,
        *,
        metadata: Mapping[str, Aligned] | None = None,
        params: Mapping[str, float] | None = None,
    ) -> list[IndicatorResult]:
        """Assemble :class:`IndicatorResult` objects from aligned values.

        Args:
            series: The source price series (for timestamps).
            primary: The primary values aligned to the series.
            metadata: Optional aligned auxiliary outputs.
            params: Optional static parameters to attach to every result.

        Returns:
            Results for every index where the primary value is defined.

        Raises:
            ComputationError: If a value is non-finite.
        """
        results: list[IndicatorResult] = []
        for index, value in enumerate(primary):
            if value is None:
                continue
            meta: dict[str, float] = dict(params or {})
            if metadata is not None:
                for key, aligned in metadata.items():
                    component = aligned[index]
                    if component is not None:
                        meta[key] = _finite(component, self.name, key)
            results.append(
                IndicatorResult(
                    name=self.name,
                    timestamp=series.timestamps[index],
                    lookback=self.lookback,
                    value=_finite(value, self.name, "value"),
                    metadata=meta,
                )
            )
        return results


def positive_int(value: int, field: str) -> int:
    """Validate that a period parameter is a positive integer.

    Args:
        value: The provided value.
        field: The parameter name (for the error message).

    Returns:
        The validated value.

    Raises:
        InvalidPeriodError: If the value is not a positive integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise InvalidPeriodError(
            f"'{field}' must be a positive integer, got {value!r}."
        )
    return value


def _finite(value: float, indicator: str, field: str) -> float:
    """Return ``value`` if finite, else raise :class:`ComputationError`."""
    if not is_finite(value):
        raise ComputationError(f"{indicator} produced a non-finite {field}: {value!r}.")
    return value
