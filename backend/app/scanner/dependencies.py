"""Dependency-injection wiring for the scanner engine."""

from __future__ import annotations

from functools import lru_cache

from app.indicators.dependencies import get_indicator_engine
from app.market.calendar.dependencies import (
    get_clock,
    get_market_calendar_service,
)
from app.market.historical.dependencies import get_historical_data_engine
from app.scanner.engine import ScannerEngine
from app.scanner.registry import ScannerRegistry, default_registry


@lru_cache(maxsize=1)
def get_scanner_registry() -> ScannerRegistry:
    """Return the shared scanner registry."""
    return default_registry()


@lru_cache(maxsize=1)
def _engine_singleton() -> ScannerEngine:
    """Return the cached scanner engine."""
    return ScannerEngine(
        data_engine=get_historical_data_engine(),
        indicator_engine=get_indicator_engine(),
        calendar=get_market_calendar_service(),
        clock=get_clock(),
        registry=get_scanner_registry(),
    )


def get_scanner_engine() -> ScannerEngine:
    """FastAPI dependency returning the shared scanner engine."""
    return _engine_singleton()
