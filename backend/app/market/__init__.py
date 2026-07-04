"""Titan market-data domain package.

Exposes the provider-agnostic enums and value objects that define the
market-data contract. Concrete providers live under :mod:`app.providers`.
"""

from app.market.enums import Exchange, Interval, MarketStatus
from app.market.models import (
    Candle,
    DepthLevel,
    HistoricalData,
    HistoricalDataRequest,
    MarketDepth,
    MarketDepthRequest,
    MarketModel,
    MarketStatusInfo,
    MarketStatusRequest,
    Quote,
    QuoteRequest,
    QuotesRequest,
    SymbolSearchRequest,
    SymbolSearchResponse,
    SymbolSearchResult,
)

__all__ = [
    "Candle",
    "DepthLevel",
    "Exchange",
    "HistoricalData",
    "HistoricalDataRequest",
    "Interval",
    "MarketDepth",
    "MarketDepthRequest",
    "MarketModel",
    "MarketStatus",
    "MarketStatusInfo",
    "MarketStatusRequest",
    "Quote",
    "QuoteRequest",
    "QuotesRequest",
    "SymbolSearchRequest",
    "SymbolSearchResponse",
    "SymbolSearchResult",
]
