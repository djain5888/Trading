"""Dependency-injection wiring for the backtesting engine.

The backtester must exercise the *exact* live strategy path, so it does not
build new strategies: it reuses the DI-resolved strategy registry/config and the
regime config, re-pointing them at a :class:`LookaheadGuard` view of the shared
data engine. Only the data view changes — the decision logic is identical to live.
"""

from __future__ import annotations

from app.backtest.engine import BacktestEngine
from app.backtest.guard import LookaheadGuard
from app.backtest.models import BacktestConfig
from app.market.regime.engine import MarketRegimeEngine
from app.strategy.engine import StrategyEngine
from app.workflow.services import WorkflowServices


def build_backtest_engine(
    services: WorkflowServices, config: BacktestConfig | None = None
) -> BacktestEngine:
    """Construct a backtest engine wired to the live strategy/regime code path.

    Args:
        services: The resolved service container (engines, calendar, clock).
        config: Market, risk and cost configuration; defaults if omitted.

    Returns:
        A ready-to-use :class:`BacktestEngine` reading through a look-ahead guard.
    """
    guard = LookaheadGuard(services.data_engine)
    strategy_engine = StrategyEngine(
        data_engine=guard,
        indicator_engine=services.indicator_engine,
        clock=services.clock,
        config=services.strategy_engine.config,
        registry=services.strategy_engine.registry,
    )
    regime_engine = MarketRegimeEngine(
        data_engine=guard,
        indicator_engine=services.indicator_engine,
        clock=services.clock,
        config=services.regime_engine.config,
    )
    return BacktestEngine(
        data=guard,
        strategy_engine=strategy_engine,
        regime_engine=regime_engine,
        calendar=services.calendar,
        config=config or BacktestConfig(),
    )
