"""Dependency-injection wiring for the backtesting engine.

The backtester must exercise the *exact* live strategy path, so it does not
build new strategies: it reuses the DI-resolved strategy registry/config and the
regime config, re-pointing them at a :class:`LookaheadGuard` view of the shared
data engine. Only the data view changes — the decision logic is identical to live.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from app.backtest.adjust import SplitSummary, adjust_for_splits
from app.backtest.engine import BacktestEngine
from app.backtest.guard import LookaheadGuard
from app.backtest.models import BacktestConfig
from app.core.logging import get_logger
from app.core.timezone import INDIA_TZ
from app.market.enums import Exchange, Interval
from app.market.historical.enums import DuplicatePolicy
from app.market.historical.models import SeriesKey
from app.market.regime.engine import MarketRegimeEngine
from app.strategy.engine import StrategyEngine
from app.workflow.services import WorkflowServices

logger = get_logger(__name__)

#: Extra calendar days pulled in before ``start`` so day one has full lookback.
_LOOKBACK_BUFFER_DAYS = 30
#: Wide, tz-aware bounds for reading a symbol's full stored series.
_EPOCH = datetime(1970, 1, 1, tzinfo=INDIA_TZ)
_FAR_FUTURE = datetime(2100, 1, 1, tzinfo=INDIA_TZ)


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
) -> list[SplitSummary]:
    """Load the candles the replay needs into the shared store, split-adjusted.

    The candle store is process-scoped, so — exactly as the morning workflow
    imports before it analyses — the backtest must load history before replaying,
    or the guarded view is empty and no setup can fire. Enough history is pulled
    before ``start`` to satisfy the strategy's full lookback on day one, and the
    benchmark index is loaded too so the regime can classify.

    Provider candles are unadjusted, so after import each series is back-adjusted
    for suspected splits/bonuses (:func:`app.backtest.adjust.adjust_for_splits`).
    A symbol whose jumps cannot be cleanly explained is left unadjusted and
    flagged UNUSABLE in the returned summary so callers can exclude it rather
    than silently trusting discontinuous prices.
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
    targets = [(symbol, config.exchange) for symbol in symbols]
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
        targets.append((index, regime_config.index_exchange))

    return await _adjust_stored_series(services, targets, config.interval)


async def _adjust_stored_series(
    services: WorkflowServices,
    targets: list[tuple[str, Exchange]],
    interval: Interval,
) -> list[SplitSummary]:
    """Back-adjust each stored series in place; return per-symbol summaries."""
    summaries: list[SplitSummary] = []
    for symbol, exchange in targets:
        key = SeriesKey(symbol=symbol, exchange=exchange, interval=interval)
        stored = await services.data_engine.get_candles(key, _EPOCH, _FAR_FUTURE)
        result = adjust_for_splits(symbol, stored)
        if not result.usable:
            logger.error(
                "UNUSABLE prices for %s: %s. Excluding from the run rather than "
                "trusting unadjusted candles.",
                symbol,
                result.reason,
            )
        elif result.events_applied:
            await services.data_engine.import_candles(
                list(result.adjusted), on_duplicate=DuplicatePolicy.UPDATE
            )
            logger.info(
                "Adjusted %s for %d split/bonus event(s).",
                symbol,
                result.events_applied,
            )
        summaries.append(
            SplitSummary(
                symbol=symbol,
                usable=result.usable,
                events_applied=result.events_applied,
                reason=result.reason,
            )
        )
    return summaries
