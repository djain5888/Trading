"""Dependency-injection wiring for the backtesting engine.

The backtester must exercise the *exact* live strategy path, so it does not
build new strategies: it reuses the DI-resolved strategy registry/config and the
regime config, re-pointing them at a :class:`LookaheadGuard` view of the shared
data engine. Only the data view changes — the decision logic is identical to live.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from app.backtest.engine import BacktestEngine
from app.backtest.guard import LookaheadGuard
from app.backtest.models import BacktestConfig
from app.core.timezone import INDIA_TZ
from app.market.regime.engine import MarketRegimeEngine
from app.strategy.engine import StrategyEngine
from app.workflow.services import WorkflowServices

#: Extra calendar days pulled in before ``start`` so day one has full lookback.
_LOOKBACK_BUFFER_DAYS = 30


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


async def load_backtest_history(
    services: WorkflowServices,
    symbols: list[str],
    start: date,
    end: date,
    config: BacktestConfig,
) -> None:
    """Load the candles the replay needs into the shared store.

    The candle store is process-scoped, so — exactly as the morning workflow
    imports before it analyses — the backtest must load history before replaying,
    or the guarded view is empty and no setup can fire. Enough history is pulled
    before ``start`` to satisfy the strategy's full lookback on day one, and the
    benchmark index is loaded too so the regime can classify.
    """
    lookback = services.strategy_engine.config.history_days + _LOOKBACK_BUFFER_DAYS
    load_start = datetime(
        start.year, start.month, start.day, tzinfo=INDIA_TZ
    ) - timedelta(days=lookback)
    load_end = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=INDIA_TZ)
    engine = services.import_engine
    await engine.import_history(
        list(symbols), config.interval, load_start, load_end, config.exchange
    )
    regime_config = services.regime_engine.config
    index = regime_config.index_symbol
    if index and index.upper() not in {symbol.strip().upper() for symbol in symbols}:
        await engine.import_history(
            [index],
            config.interval,
            load_start,
            load_end,
            regime_config.index_exchange,
        )
