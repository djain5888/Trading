"""The scanner registry.

Maps scanner names to factories. Registering a factory is the only step needed
to add a scanner.
"""

from __future__ import annotations

from collections.abc import Callable

from app.scanner import scanners
from app.scanner.base import Scanner
from app.scanner.exceptions import ScannerConfigError, UnknownScannerError

ScannerFactory = Callable[..., Scanner]


class ScannerRegistry:
    """A registry of scanner factories."""

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._factories: dict[str, ScannerFactory] = {}

    def register(self, name: str, factory: ScannerFactory) -> None:
        """Register a scanner factory under ``name``."""
        self._factories[name] = factory

    def create(self, name: str, **params: float) -> Scanner:
        """Create a configured scanner.

        Args:
            name: The registered scanner name.
            **params: Constructor parameters.

        Returns:
            A configured :class:`Scanner`.

        Raises:
            UnknownScannerError: If ``name`` is not registered.
            ScannerConfigError: If parameters are unexpected or invalid.
        """
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise UnknownScannerError(f"Unknown scanner '{name}'.") from exc
        try:
            return factory(**params)
        except TypeError as exc:
            raise ScannerConfigError(f"Invalid parameters for '{name}': {exc}") from exc

    def create_all(self) -> list[Scanner]:
        """Create every registered scanner with default parameters."""
        return [factory() for factory in self._factories.values()]

    def available(self) -> list[str]:
        """Return the sorted list of registered scanner names."""
        return sorted(self._factories)


def default_registry() -> ScannerRegistry:
    """Return a registry populated with all built-in scanners."""
    registry = ScannerRegistry()
    registry.register("ema_alignment", scanners.EmaAlignmentScanner)
    registry.register("breakout", scanners.BreakoutScanner)
    registry.register("relative_volume", scanners.RelativeVolumeScanner)
    registry.register("atr_expansion", scanners.AtrExpansionScanner)
    registry.register("vwap_strength", scanners.VwapStrengthScanner)
    registry.register("rsi_momentum", scanners.RsiMomentumScanner)
    registry.register("macd_cross", scanners.MacdCrossScanner)
    registry.register("bollinger_expansion", scanners.BollingerExpansionScanner)
    registry.register("gap_up", scanners.GapUpScanner)
    registry.register("gap_down", scanners.GapDownScanner)
    return registry


__all__ = [
    "ScannerConfigError",
    "ScannerRegistry",
    "default_registry",
]
