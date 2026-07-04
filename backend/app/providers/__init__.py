"""Titan market-data providers package.

Exposes the provider interface, the exception hierarchy and the
dependency-injection entry points. Titan code should depend on
:class:`MarketDataProvider`, never on a concrete provider.
"""

from app.providers.base import MarketDataProvider, ResponseParser
from app.providers.dependencies import (
    build_market_data_provider,
    get_market_data_provider,
)
from app.providers.exceptions import (
    AuthenticationError,
    InvalidSymbolError,
    MarketClosedError,
    NetworkError,
    ProviderError,
    RateLimitError,
)
from app.providers.http import HttpMethod, RequestSpec

__all__ = [
    "AuthenticationError",
    "HttpMethod",
    "InvalidSymbolError",
    "MarketClosedError",
    "MarketDataProvider",
    "NetworkError",
    "ProviderError",
    "RateLimitError",
    "RequestSpec",
    "ResponseParser",
    "build_market_data_provider",
    "get_market_data_provider",
]
