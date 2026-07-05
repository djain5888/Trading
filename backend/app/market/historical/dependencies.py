"""Dependency-injection wiring for the historical-data engine.

Collaborators are exposed as cached singletons bound to interfaces; the backing
store is chosen here and nowhere else. There is no global mutable state beyond
the deliberately shared repository singleton.
"""

from __future__ import annotations

from functools import lru_cache

from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.repository import CandleRepository
from app.market.historical.validation import ValidationEngine


@lru_cache(maxsize=1)
def get_candle_repository() -> CandleRepository:
    """Return the process-wide candle repository (in-memory by default)."""
    return InMemoryCandleRepository()


@lru_cache(maxsize=1)
def get_validation_engine() -> ValidationEngine:
    """Return the shared validation engine."""
    return ValidationEngine()


@lru_cache(maxsize=1)
def _engine_singleton() -> HistoricalDataEngine:
    """Return the cached historical-data engine."""
    return HistoricalDataEngine(
        repository=get_candle_repository(), validator=get_validation_engine()
    )


def get_historical_data_engine() -> HistoricalDataEngine:
    """FastAPI dependency returning the shared historical-data engine."""
    return _engine_singleton()
