"""Dependency-injection wiring for the DuckDB candle repository.

Registers the DuckDB repository as a cached singleton bound to the
:class:`CandleRepository` interface, built from the DuckDB path in application
settings. The historical engine can be assembled over it without knowing the
backend. There is no global mutable state beyond the shared singletons.
"""

from __future__ import annotations

from functools import lru_cache

from app.config.settings import get_settings
from app.market.historical.dependencies import get_validation_engine
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.repository import CandleRepository
from app.market.historical.storage.duckdb_repository import DuckDBCandleRepository


@lru_cache(maxsize=1)
def get_duckdb_candle_repository() -> CandleRepository:
    """Return the process-wide DuckDB candle repository."""
    settings = get_settings()
    return DuckDBCandleRepository(settings.duckdb.path)


@lru_cache(maxsize=1)
def _duckdb_engine_singleton() -> HistoricalDataEngine:
    """Return the cached historical engine backed by DuckDB."""
    return HistoricalDataEngine(
        repository=get_duckdb_candle_repository(),
        validator=get_validation_engine(),
    )


def get_duckdb_historical_data_engine() -> HistoricalDataEngine:
    """FastAPI dependency returning a DuckDB-backed historical engine."""
    return _duckdb_engine_singleton()
