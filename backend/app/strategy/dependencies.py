"""Dependency-injection wiring for the strategy engine."""

from __future__ import annotations

from functools import lru_cache

from app.config.watchlist import get_watchlist_config
from app.core.clock import Clock
from app.indicators.dependencies import get_indicator_engine
from app.indicators.engine import IndicatorEngine
from app.market.calendar.dependencies import get_clock
from app.market.historical.dependencies import get_historical_data_engine
from app.market.historical.engine import HistoricalDataEngine
from app.strategy.engine import StrategyEngine
from app.strategy.models import StrategyConfig


def build_strategy_engine(
    data_engine: HistoricalDataEngine | None = None,
    indicator_engine: IndicatorEngine | None = None,
    clock: Clock | None = None,
    config: StrategyConfig | None = None,
) -> StrategyEngine:
    """Construct a strategy engine from DI-resolved collaborators.

    Args:
        data_engine: Candle source; defaults to the shared engine.
        indicator_engine: Indicator engine; defaults to the shared one.
        clock: Time source; defaults to the shared clock.
        config: Tuning and blend weights; defaults to daily settings.

    Returns:
        A ready-to-use :class:`StrategyEngine`.
    """
    return StrategyEngine(
        data_engine=data_engine or get_historical_data_engine(),
        indicator_engine=indicator_engine or get_indicator_engine(),
        clock=clock or get_clock(),
        config=config or StrategyConfig(),
    )


@lru_cache(maxsize=1)
def _engine_singleton() -> StrategyEngine:
    """Return the cached strategy engine, wired to the sector map."""
    return build_strategy_engine(
        config=StrategyConfig(sector_map=dict(get_watchlist_config().sectors))
    )


def get_strategy_engine() -> StrategyEngine:
    """FastAPI/CLI dependency returning the shared strategy engine."""
    return _engine_singleton()
