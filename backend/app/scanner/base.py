"""The scanner interface and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from app.scanner.exceptions import ScannerConfigError
from app.scanner.models import ScannerContext, ScannerResult


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    """Clamp a value into ``[lower, upper]``."""
    return max(lower, min(upper, value))


def require_positive(value: float, field: str) -> float:
    """Return ``value`` if strictly positive, else raise a config error."""
    if value <= 0:
        raise ScannerConfigError(f"'{field}' must be positive, got {value!r}.")
    return value


def require_positive_int(value: int, field: str) -> int:
    """Return ``value`` if a positive integer, else raise a config error."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ScannerConfigError(
            f"'{field}' must be a positive integer, got {value!r}."
        )
    return value


class Scanner(ABC):
    """Discovers whether a symbol is a candidate under one heuristic.

    Scanners are independently pluggable: adding one means registering a new
    subclass. A scanner returns a scored :class:`ScannerResult` for a candidate
    or ``None`` to filter the symbol out. It never emits buy/sell decisions.
    """

    name: ClassVar[str]
    #: Numeric configuration, set by subclasses in ``__init__``.
    _config: dict[str, float] = {}

    @abstractmethod
    def scan(self, context: ScannerContext) -> ScannerResult | None:
        """Evaluate a symbol and return a result, or ``None`` to skip it."""

    @property
    def config(self) -> dict[str, float]:
        """Return this scanner's configuration as numeric parameters."""
        return self._config

    def config_key(self) -> tuple[tuple[str, float], ...]:
        """Return a hashable representation of the configuration."""
        return tuple(sorted(self.config.items()))

    # -- helpers -----------------------------------------------------------

    def _last_value(
        self, context: ScannerContext, indicator: str, **params: float
    ) -> float | None:
        """Return the most recent value of an indicator, or ``None``."""
        results = context.safe_compute(indicator, **params)
        return results[-1].value if results else None

    def _result(
        self,
        context: ScannerContext,
        *,
        score: float,
        confidence: float,
        metadata: dict[str, float],
    ) -> ScannerResult:
        """Build a clamped :class:`ScannerResult` for this scanner."""
        bounded = clamp(score)
        return ScannerResult(
            symbol=context.key.symbol,
            scanner_name=self.name,
            score=bounded,
            confidence=clamp(confidence, 0.0, 1.0),
            timestamp=context.timestamp,
            metadata=metadata,
        )
