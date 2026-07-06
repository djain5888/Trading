"""Titan indicator engine package.

The single place technical indicators are computed. Independent of scanners,
strategies and AI; it only reads historical candles.
"""

from app.indicators.base import IndicatorCalculator
from app.indicators.dependencies import (
    build_indicator_engine,
    get_indicator_engine,
    get_indicator_registry,
)
from app.indicators.engine import IndicatorEngine
from app.indicators.exceptions import (
    ComputationError,
    IndicatorError,
    InsufficientHistoryError,
    InvalidPeriodError,
    UnknownIndicatorError,
)
from app.indicators.models import IndicatorResult, PriceSeries
from app.indicators.registry import IndicatorRegistry, default_registry

__all__ = [
    "ComputationError",
    "IndicatorCalculator",
    "IndicatorEngine",
    "IndicatorError",
    "IndicatorRegistry",
    "IndicatorResult",
    "InsufficientHistoryError",
    "InvalidPeriodError",
    "PriceSeries",
    "UnknownIndicatorError",
    "build_indicator_engine",
    "default_registry",
    "get_indicator_engine",
    "get_indicator_registry",
]
