"""Historical market-data engine package.

Stores, validates, updates and retrieves OHLCV candles behind storage-
independent interfaces.
"""

from app.market.historical.dependencies import (
    get_candle_repository,
    get_historical_data_engine,
    get_validation_engine,
)
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.enums import DuplicatePolicy, IssueType
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import Candle, SeriesKey
from app.market.historical.report import DataQualityReport, ValidationIssue
from app.market.historical.repository import CandleRepository
from app.market.historical.validation import ValidationEngine

__all__ = [
    "Candle",
    "CandleRepository",
    "DataQualityReport",
    "DuplicatePolicy",
    "HistoricalDataEngine",
    "InMemoryCandleRepository",
    "IssueType",
    "SeriesKey",
    "ValidationEngine",
    "ValidationIssue",
    "get_candle_repository",
    "get_historical_data_engine",
    "get_validation_engine",
]
