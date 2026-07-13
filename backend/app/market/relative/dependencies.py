"""Dependency-injection wiring for the relative-strength engine."""

from __future__ import annotations

from functools import lru_cache

from app.config.watchlist import get_watchlist_config
from app.core.clock import Clock
from app.market.calendar.dependencies import get_clock
from app.market.historical.dependencies import get_historical_data_engine
from app.market.historical.engine import HistoricalDataEngine
from app.market.relative.engine import RelativeStrengthEngine
from app.market.relative.models import RSConfig


def build_relative_strength_engine(
    data_engine: HistoricalDataEngine | None = None,
    clock: Clock | None = None,
    config: RSConfig | None = None,
) -> RelativeStrengthEngine:
    """Construct a relative-strength engine from DI-resolved collaborators.

    Args:
        data_engine: Candle source; defaults to the shared engine.
        clock: Time source; defaults to the shared clock.
        config: Tuning and sector metadata; defaults to NIFTY/daily.

    Returns:
        A ready-to-use :class:`RelativeStrengthEngine`.
    """
    return RelativeStrengthEngine(
        data_engine=data_engine or get_historical_data_engine(),
        clock=clock or get_clock(),
        config=config or RSConfig(),
    )


@lru_cache(maxsize=1)
def _engine_singleton() -> RelativeStrengthEngine:
    """Return the cached RS engine, wired to the benchmark index and sectors."""
    watchlist = get_watchlist_config()
    return build_relative_strength_engine(
        config=RSConfig(
            index_symbol=watchlist.index_symbol,
            sector_map=dict(watchlist.sectors),
        )
    )


def get_relative_strength_engine() -> RelativeStrengthEngine:
    """FastAPI/CLI dependency returning the shared relative-strength engine."""
    return _engine_singleton()
