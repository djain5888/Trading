"""Dependency-injection wiring for market-data providers.

The rest of Titan asks for a :class:`MarketDataProvider` and never names a
broker. Swapping providers is a single change here. The concrete provider is
built from application settings and cached as a process singleton.
"""

from __future__ import annotations

from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.providers.base import MarketDataProvider
from app.providers.groww.provider import GrowwProvider


def build_market_data_provider(settings: Settings | None = None) -> MarketDataProvider:
    """Construct the configured market-data provider.

    Args:
        settings: Application settings; defaults to the cached settings.

    Returns:
        A concrete :class:`MarketDataProvider` implementation.
    """
    settings = settings or get_settings()
    return GrowwProvider(settings=settings.groww)


@lru_cache(maxsize=1)
def _provider_singleton() -> MarketDataProvider:
    """Return the cached provider instance."""
    return build_market_data_provider()


def get_market_data_provider() -> MarketDataProvider:
    """FastAPI dependency returning the shared market-data provider.

    Returns:
        The process-wide :class:`MarketDataProvider` singleton.
    """
    return _provider_singleton()
