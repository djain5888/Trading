"""Dependency-injection wiring for application services."""

from __future__ import annotations

from functools import lru_cache

from app.providers.base import MarketDataProvider
from app.providers.dependencies import get_market_data_provider
from app.services.market_data import MarketDataService


def build_market_data_service(
    provider: MarketDataProvider | None = None,
) -> MarketDataService:
    """Construct a :class:`MarketDataService`.

    Args:
        provider: Provider to wrap; defaults to the configured provider.

    Returns:
        A ready-to-use market-data service.
    """
    provider = provider or get_market_data_provider()
    return MarketDataService(provider)


@lru_cache(maxsize=1)
def _service_singleton() -> MarketDataService:
    """Return the cached market-data service instance."""
    return build_market_data_service()


def get_market_data_service() -> MarketDataService:
    """FastAPI dependency returning the shared market-data service.

    Returns:
        The process-wide :class:`MarketDataService` singleton.
    """
    return _service_singleton()
