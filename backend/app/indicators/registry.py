"""The indicator registry.

Maps indicator names to calculator factories. Registering a new calculator is
the only step needed to add an indicator; nothing else changes.
"""

from __future__ import annotations

from collections.abc import Callable

from app.indicators import calculators
from app.indicators.base import IndicatorCalculator
from app.indicators.exceptions import InvalidPeriodError, UnknownIndicatorError

CalculatorFactory = Callable[..., IndicatorCalculator]


class IndicatorRegistry:
    """A registry of indicator calculator factories."""

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._factories: dict[str, CalculatorFactory] = {}

    def register(self, name: str, factory: CalculatorFactory) -> None:
        """Register a calculator factory under ``name``.

        Args:
            name: The indicator name.
            factory: A callable returning a configured calculator.
        """
        self._factories[name] = factory

    def create(self, name: str, **params: float) -> IndicatorCalculator:
        """Create a configured calculator.

        Args:
            name: The registered indicator name.
            **params: Constructor parameters for the calculator.

        Returns:
            A configured :class:`IndicatorCalculator`.

        Raises:
            UnknownIndicatorError: If ``name`` is not registered.
            InvalidPeriodError: If parameters are unexpected or invalid.
        """
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise UnknownIndicatorError(f"Unknown indicator '{name}'.") from exc
        try:
            return factory(**params)
        except TypeError as exc:
            raise InvalidPeriodError(f"Invalid parameters for '{name}': {exc}") from exc

    def available(self) -> list[str]:
        """Return the sorted list of registered indicator names."""
        return sorted(self._factories)


def default_registry() -> IndicatorRegistry:
    """Return a registry populated with all built-in indicators."""
    registry = IndicatorRegistry()
    registry.register("sma", calculators.SmaCalculator)
    registry.register("ema", calculators.EmaCalculator)
    registry.register("rsi", calculators.RsiCalculator)
    registry.register("roc", calculators.RocCalculator)
    registry.register("momentum", calculators.MomentumCalculator)
    registry.register("macd", calculators.MacdCalculator)
    registry.register("adx", calculators.AdxCalculator)
    registry.register("dmi", calculators.DmiCalculator)
    registry.register("atr", calculators.AtrCalculator)
    registry.register("bollinger", calculators.BollingerCalculator)
    registry.register("stddev", calculators.StdDevCalculator)
    registry.register("vwap", calculators.VwapCalculator)
    registry.register("obv", calculators.ObvCalculator)
    registry.register("volume_sma", calculators.VolumeSmaCalculator)
    registry.register("typical_price", calculators.TypicalPriceCalculator)
    registry.register("median_price", calculators.MedianPriceCalculator)
    registry.register("weighted_close", calculators.WeightedCloseCalculator)
    return registry
