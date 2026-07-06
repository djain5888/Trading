"""Dependency-injection wiring for the historical import engine.

Wires the engine to the provider, data engine, calendar and clock — all through
their interfaces. Nothing here references a concrete provider.
"""

from __future__ import annotations

from functools import lru_cache

from app.market.calendar.dependencies import (
    get_clock,
    get_market_calendar_service,
)
from app.market.historical.dependencies import get_historical_data_engine
from app.market.historical.importer.engine import HistoricalImportEngine
from app.providers.dependencies import get_market_data_provider


@lru_cache(maxsize=1)
def _import_engine_singleton() -> HistoricalImportEngine:
    """Return the cached historical import engine."""
    return HistoricalImportEngine(
        provider=get_market_data_provider(),
        data_engine=get_historical_data_engine(),
        calendar=get_market_calendar_service(),
        clock=get_clock(),
    )


def get_historical_import_engine() -> HistoricalImportEngine:
    """FastAPI dependency returning the shared historical import engine."""
    return _import_engine_singleton()
