"""The strategy registry.

Maps strategy names to factories. Registering a new strategy is the only step
needed to add one; nothing else changes.
"""

from __future__ import annotations

from collections.abc import Callable

from app.strategy.base import Strategy
from app.strategy.strategies import (
    BreakoutStrategy,
    MomentumStrategy,
    PullbackStrategy,
)

StrategyFactory = Callable[[], Strategy]


class StrategyRegistry:
    """A registry of strategy factories."""

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._factories: dict[str, StrategyFactory] = {}

    def register(self, name: str, factory: StrategyFactory) -> None:
        """Register a strategy factory under ``name``."""
        self._factories[name] = factory

    def create_all(self) -> list[Strategy]:
        """Instantiate every registered strategy."""
        return [factory() for factory in self._factories.values()]

    def available(self) -> list[str]:
        """Return the sorted registered strategy names."""
        return sorted(self._factories)


def default_registry() -> StrategyRegistry:
    """Return a registry populated with the built-in strategies."""
    registry = StrategyRegistry()
    registry.register("breakout", BreakoutStrategy)
    registry.register("pullback", PullbackStrategy)
    registry.register("momentum", MomentumStrategy)
    return registry
