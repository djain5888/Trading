"""Live market collector package.

Continuously collects live market data from a provider and persists snapshots,
gated by the market calendar. Contains no trading logic.
"""

from app.market.live.collector import LiveMarketCollector
from app.market.live.dependencies import build_live_market_collector
from app.market.live.enums import CollectionMode
from app.market.live.models import CollectorConfig, CollectorMetricsSnapshot

__all__ = [
    "CollectionMode",
    "CollectorConfig",
    "CollectorMetricsSnapshot",
    "LiveMarketCollector",
    "build_live_market_collector",
]
