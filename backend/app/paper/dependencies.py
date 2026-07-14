"""Dependency-injection wiring for the paper-trading engine."""

from __future__ import annotations

from functools import lru_cache

from app.config.settings import get_settings
from app.core.clock import Clock
from app.market.calendar.dependencies import get_clock
from app.market.historical.dependencies import get_historical_data_engine
from app.market.historical.engine import HistoricalDataEngine
from app.paper.engine import PaperTradingEngine
from app.paper.models import PaperConfig
from app.paper.repository import PaperTradeRepository
from app.paper.storage import DuckDBPaperTradeRepository


@lru_cache(maxsize=1)
def get_paper_trade_repository() -> PaperTradeRepository:
    """Return the process-wide paper-trade repository (DuckDB-backed).

    Trades persist across runs in the same analytical store as the candles.
    """
    return DuckDBPaperTradeRepository(get_settings().duckdb.path)


def build_paper_trading_engine(
    repository: PaperTradeRepository | None = None,
    data_engine: HistoricalDataEngine | None = None,
    clock: Clock | None = None,
    config: PaperConfig | None = None,
) -> PaperTradingEngine:
    """Construct a paper-trading engine from DI-resolved collaborators.

    Args:
        repository: Trade store; defaults to the shared DuckDB repository.
        data_engine: Candle source for marking; defaults to the shared engine.
        clock: Time source; defaults to the shared clock.
        config: Portfolio/risk config; defaults to 100k / 1% / 5 positions.

    Returns:
        A ready-to-use :class:`PaperTradingEngine`.
    """
    return PaperTradingEngine(
        repository=repository or get_paper_trade_repository(),
        data_engine=data_engine or get_historical_data_engine(),
        clock=clock or get_clock(),
        config=config or PaperConfig(),
    )


@lru_cache(maxsize=1)
def _engine_singleton() -> PaperTradingEngine:
    """Return the cached paper-trading engine."""
    return build_paper_trading_engine()


def get_paper_trading_engine() -> PaperTradingEngine:
    """FastAPI/CLI dependency returning the shared paper-trading engine."""
    return _engine_singleton()
