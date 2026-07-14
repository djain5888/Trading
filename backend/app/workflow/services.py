"""Service container resolved through the existing DI accessors.

The workflow engine never constructs services directly; it resolves everything
here so orchestration stays decoupled from wiring.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.config.settings import Settings, get_settings
from app.core.clock import Clock
from app.indicators.dependencies import get_indicator_engine
from app.indicators.engine import IndicatorEngine
from app.market.calendar.dependencies import (
    get_clock,
    get_market_calendar_service,
)
from app.market.calendar.service import MarketCalendarService
from app.market.historical.dependencies import get_historical_data_engine
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.importer.dependencies import (
    get_historical_import_engine,
)
from app.market.historical.importer.engine import HistoricalImportEngine
from app.market.live.collector import LiveMarketCollector
from app.market.live.dependencies import build_live_market_collector
from app.market.live.models import CollectorConfig
from app.market.regime.dependencies import get_market_regime_engine
from app.market.regime.engine import MarketRegimeEngine
from app.market.relative.dependencies import get_relative_strength_engine
from app.market.relative.engine import RelativeStrengthEngine
from app.market.sector.dependencies import get_sector_strength_engine
from app.market.sector.engine import SectorStrengthEngine
from app.providers.base import MarketDataProvider
from app.providers.dependencies import get_market_data_provider
from app.scanner.dependencies import get_scanner_engine
from app.scanner.engine import ScannerEngine
from app.strategy.dependencies import get_strategy_engine
from app.strategy.engine import StrategyEngine

CollectorFactory = Callable[[CollectorConfig], LiveMarketCollector]


@dataclass(frozen=True)
class WorkflowServices:
    """All services a workflow may need, resolved from DI."""

    settings: Settings
    clock: Clock
    calendar: MarketCalendarService
    data_engine: HistoricalDataEngine
    import_engine: HistoricalImportEngine
    indicator_engine: IndicatorEngine
    scanner_engine: ScannerEngine
    regime_engine: MarketRegimeEngine
    sector_engine: SectorStrengthEngine
    relative_engine: RelativeStrengthEngine
    strategy_engine: StrategyEngine
    provider: MarketDataProvider
    collector_factory: CollectorFactory

    @classmethod
    def resolve(cls) -> WorkflowServices:
        """Resolve all services through the DI container."""
        return cls(
            settings=get_settings(),
            clock=get_clock(),
            calendar=get_market_calendar_service(),
            data_engine=get_historical_data_engine(),
            import_engine=get_historical_import_engine(),
            indicator_engine=get_indicator_engine(),
            scanner_engine=get_scanner_engine(),
            regime_engine=get_market_regime_engine(),
            sector_engine=get_sector_strength_engine(),
            relative_engine=get_relative_strength_engine(),
            strategy_engine=get_strategy_engine(),
            provider=get_market_data_provider(),
            collector_factory=build_live_market_collector,
        )
