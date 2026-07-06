"""Dependency-injection wiring for the live market collector.

A collector needs a runtime configuration (watchlist), so it is built via a
factory rather than a cached singleton. Collaborators default to the shared
interface-bound singletons; none of them reference a concrete provider.
"""

from __future__ import annotations

from app.core.clock import Clock
from app.market.calendar.dependencies import get_clock, get_market_calendar_service
from app.market.calendar.service import MarketCalendarService
from app.market.historical.dependencies import get_candle_repository
from app.market.historical.repository import CandleRepository
from app.market.live.collector import LiveMarketCollector
from app.market.live.models import CollectorConfig
from app.providers.base import MarketDataProvider
from app.providers.dependencies import get_market_data_provider


def build_live_market_collector(
    config: CollectorConfig,
    *,
    provider: MarketDataProvider | None = None,
    calendar: MarketCalendarService | None = None,
    clock: Clock | None = None,
    repository: CandleRepository | None = None,
) -> LiveMarketCollector:
    """Construct a live market collector.

    Args:
        config: Watchlist and timing configuration.
        provider: Market-data source; defaults to the configured provider.
        calendar: Calendar service; defaults to the shared service.
        clock: Time source; defaults to the shared clock.
        repository: Snapshot sink; defaults to the shared candle repository.

    Returns:
        A ready-to-run :class:`LiveMarketCollector`.
    """
    return LiveMarketCollector(
        provider=provider or get_market_data_provider(),
        calendar=calendar or get_market_calendar_service(),
        clock=clock or get_clock(),
        repository=repository or get_candle_repository(),
        config=config,
    )
