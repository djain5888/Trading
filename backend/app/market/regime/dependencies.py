"""Dependency-injection wiring for the market-regime engine."""

from __future__ import annotations

from functools import lru_cache

from app.config.watchlist import get_watchlist_config
from app.core.clock import Clock
from app.indicators.dependencies import get_indicator_engine
from app.indicators.engine import IndicatorEngine
from app.market.calendar.dependencies import get_clock
from app.market.historical.dependencies import get_historical_data_engine
from app.market.historical.engine import HistoricalDataEngine
from app.market.regime.engine import MarketRegimeEngine
from app.market.regime.models import RegimeConfig


def build_market_regime_engine(
    data_engine: HistoricalDataEngine | None = None,
    indicator_engine: IndicatorEngine | None = None,
    clock: Clock | None = None,
    config: RegimeConfig | None = None,
) -> MarketRegimeEngine:
    """Construct a market-regime engine from DI-resolved collaborators.

    Args:
        data_engine: Candle source; defaults to the shared engine.
        indicator_engine: Indicator engine; defaults to the shared one.
        clock: Time source; defaults to the shared clock.
        config: Tuning parameters; defaults to NIFTY/daily settings.

    Returns:
        A ready-to-use :class:`MarketRegimeEngine`.
    """
    return MarketRegimeEngine(
        data_engine=data_engine or get_historical_data_engine(),
        indicator_engine=indicator_engine or get_indicator_engine(),
        clock=clock or get_clock(),
        config=config or RegimeConfig(),
    )


@lru_cache(maxsize=1)
def _engine_singleton() -> MarketRegimeEngine:
    """Return the cached market-regime engine, wired to the benchmark index."""
    return build_market_regime_engine(
        config=RegimeConfig(index_symbol=get_watchlist_config().index_symbol)
    )


def get_market_regime_engine() -> MarketRegimeEngine:
    """FastAPI/CLI dependency returning the shared market-regime engine."""
    return _engine_singleton()
