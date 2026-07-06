"""Dependency-injection wiring for the indicator engine."""

from __future__ import annotations

from functools import lru_cache

from app.core.clock import Clock
from app.indicators.engine import IndicatorEngine
from app.indicators.registry import IndicatorRegistry, default_registry
from app.market.calendar.dependencies import get_clock
from app.market.historical.dependencies import get_historical_data_engine
from app.market.historical.engine import HistoricalDataEngine


@lru_cache(maxsize=1)
def get_indicator_registry() -> IndicatorRegistry:
    """Return the shared indicator registry."""
    return default_registry()


def build_indicator_engine(
    data_engine: HistoricalDataEngine | None = None,
    clock: Clock | None = None,
    registry: IndicatorRegistry | None = None,
) -> IndicatorEngine:
    """Construct an indicator engine.

    Args:
        data_engine: Candle source; defaults to the shared engine.
        clock: Time source; defaults to the shared clock.
        registry: Indicator registry; defaults to the shared registry.

    Returns:
        A ready-to-use :class:`IndicatorEngine`.
    """
    return IndicatorEngine(
        data_engine=data_engine or get_historical_data_engine(),
        clock=clock or get_clock(),
        registry=registry or get_indicator_registry(),
    )


@lru_cache(maxsize=1)
def _engine_singleton() -> IndicatorEngine:
    """Return the cached indicator engine."""
    return build_indicator_engine()


def get_indicator_engine() -> IndicatorEngine:
    """FastAPI dependency returning the shared indicator engine."""
    return _engine_singleton()
