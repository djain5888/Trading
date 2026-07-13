"""Dependency-injection wiring for the sector-strength engine."""

from __future__ import annotations

from functools import lru_cache

from app.config.watchlist import get_watchlist_config
from app.core.clock import Clock
from app.indicators.dependencies import get_indicator_engine
from app.indicators.engine import IndicatorEngine
from app.market.calendar.dependencies import get_clock
from app.market.historical.dependencies import get_historical_data_engine
from app.market.historical.engine import HistoricalDataEngine
from app.market.sector.engine import SectorStrengthEngine
from app.market.sector.models import SectorConfig


def build_sector_strength_engine(
    data_engine: HistoricalDataEngine | None = None,
    indicator_engine: IndicatorEngine | None = None,
    clock: Clock | None = None,
    config: SectorConfig | None = None,
) -> SectorStrengthEngine:
    """Construct a sector-strength engine from DI-resolved collaborators.

    Args:
        data_engine: Candle source; defaults to the shared engine.
        indicator_engine: Indicator engine; defaults to the shared one.
        clock: Time source; defaults to the shared clock.
        config: Metadata and tuning; defaults to an empty (degraded) map.

    Returns:
        A ready-to-use :class:`SectorStrengthEngine`.
    """
    return SectorStrengthEngine(
        data_engine=data_engine or get_historical_data_engine(),
        indicator_engine=indicator_engine or get_indicator_engine(),
        clock=clock or get_clock(),
        config=config or SectorConfig(),
    )


@lru_cache(maxsize=1)
def _engine_singleton() -> SectorStrengthEngine:
    """Return the cached sector-strength engine, wired to the sector map."""
    return build_sector_strength_engine(
        config=SectorConfig(sector_map=dict(get_watchlist_config().sectors))
    )


def get_sector_strength_engine() -> SectorStrengthEngine:
    """FastAPI/CLI dependency returning the shared sector-strength engine."""
    return _engine_singleton()
