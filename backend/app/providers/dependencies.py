"""Dependency-injection wiring for market-data providers.

The rest of Titan asks for a :class:`MarketDataProvider` and never names a
broker. The concrete backend is chosen by ``market_data_provider`` in settings:
``groww`` (the real HTTP provider, default), ``groww_sdk`` (the official
``growwapi`` SDK) or ``fake`` (the offline skeleton, for tests/CI without
credentials). Swapping backends is a single config change.
"""

from __future__ import annotations

from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.providers.base import MarketDataProvider
from app.providers.groww.market_data_provider import GrowwMarketDataProvider
from app.providers.groww.provider import GrowwProvider
from app.providers.groww.sdk_provider import GrowwSDKProvider


def build_market_data_provider(settings: Settings | None = None) -> MarketDataProvider:
    """Construct the configured market-data provider.

    Args:
        settings: Application settings; defaults to the cached settings.

    Returns:
        A concrete :class:`MarketDataProvider` implementation. ``groww`` builds
        the real HTTP provider; ``groww_sdk`` the ``growwapi`` SDK provider;
        ``fake`` the offline skeleton.
    """
    settings = settings or get_settings()
    if settings.market_data_provider == "fake":
        return GrowwProvider(settings=settings.groww)
    if settings.market_data_provider == "groww_sdk":
        return GrowwSDKProvider(settings=settings.groww)
    return GrowwMarketDataProvider(settings=settings.groww)


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
