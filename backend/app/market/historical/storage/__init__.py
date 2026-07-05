"""Storage adapters for the historical-data engine."""

from app.market.historical.storage.dependencies import (
    get_duckdb_candle_repository,
    get_duckdb_historical_data_engine,
)
from app.market.historical.storage.duckdb_repository import DuckDBCandleRepository
from app.market.historical.storage.exceptions import (
    DuplicateCandleError,
    QueryError,
    SchemaInitializationError,
    StorageConnectionError,
    StorageError,
)

__all__ = [
    "DuckDBCandleRepository",
    "DuplicateCandleError",
    "QueryError",
    "SchemaInitializationError",
    "StorageConnectionError",
    "StorageError",
    "get_duckdb_candle_repository",
    "get_duckdb_historical_data_engine",
]
