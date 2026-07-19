"""Dependency-injection wiring and registry for the baseline search.

Baselines run over a :class:`LookaheadGuard` view of the shared data engine and
reuse the DI-resolved indicator engine and calendar, so their realism matches
the main backtest exactly.
"""

from __future__ import annotations

from collections.abc import Callable

from app.backtest.baselines.base import Baseline
from app.backtest.baselines.engine import BaselineEngine
from app.backtest.baselines.models import BaselineConfig, BaselineName
from app.backtest.baselines.strategies import (
    BuyAndHold,
    DualMomentum,
    MeanRevert,
    Momentum121,
    RSMomentumCash,
    RSMomentumTrail,
    RSRotationMonthly,
    TopRSTrailing,
    TopRSWeekly,
    TrendFollow,
    VolatilityBreak,
)
from app.backtest.guard import LookaheadGuard
from app.backtest.models import BacktestConfig
from app.workflow.services import WorkflowServices

_REGISTRY: dict[BaselineName, Callable[[BaselineConfig], Baseline]] = {
    BaselineName.BUY_AND_HOLD: BuyAndHold,
    BaselineName.TOP_RS_WEEKLY: TopRSWeekly,
    BaselineName.TOP_RS_TRAILING: TopRSTrailing,
    BaselineName.MOMENTUM_12_1: Momentum121,
    BaselineName.MEAN_REVERT: MeanRevert,
    BaselineName.RS_MOMENTUM_TRAIL: RSMomentumTrail,
    BaselineName.RS_MOMENTUM_CASH: RSMomentumCash,
    BaselineName.DUAL_MOMENTUM: DualMomentum,
    BaselineName.TREND_FOLLOW: TrendFollow,
    BaselineName.VOLATILITY_BREAK: VolatilityBreak,
    BaselineName.RS_ROTATION_MONTHLY: RSRotationMonthly,
}


def create_baseline(
    name: BaselineName, config: BaselineConfig | None = None
) -> Baseline:
    """Construct a fresh baseline instance (state resets per run).

    Raises:
        KeyError: If ``name`` is not a registered baseline.
    """
    return _REGISTRY[name](config or BaselineConfig())


def build_baseline_engine(
    services: WorkflowServices,
    name: BaselineName,
    *,
    config: BacktestConfig | None = None,
    baseline_config: BaselineConfig | None = None,
) -> BaselineEngine:
    """Construct a baseline engine wired to a guarded view of the data engine."""
    resolved = config or BacktestConfig()
    tuning = baseline_config or BaselineConfig()
    guard = LookaheadGuard(services.data_engine)
    return BaselineEngine(
        data=guard,
        indicators=services.indicator_engine,
        calendar=services.calendar,
        baseline=create_baseline(name, tuning),
        config=resolved,
        history_days=tuning.history_days,
    )
