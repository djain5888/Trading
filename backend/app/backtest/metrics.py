"""Pure expectancy maths for the backtester.

Every function here is a deterministic transform of a sequence of closed
:class:`BacktestTrade` records — no I/O, no engines — so the arithmetic (win
rate, profit factor, expectancy in R, drawdown, streaks) can be verified against
hand-computed inputs. The engine owns the simulation; this module owns the maths.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean, median

from app.backtest.models import (
    BacktestTrade,
    ExecutionLeakage,
    ExitAnalysis,
    ExitBreakdown,
    ExitKind,
    MonthlyReturn,
    PnlDecomposition,
    RConsistency,
    RegimePerformance,
    StopLeakage,
    StrategyPerformance,
    TargetLeakage,
    WinnerRDistribution,
)
from app.paper.models import ExitReason

_UNKNOWN_REGIME = "UNKNOWN"
_BULL_REGIME = "BULL"


def _holding_days(trade: BacktestTrade) -> int:
    """Return the calendar days a trade was held (entry to exit)."""
    return (trade.exit_date - trade.entry_date).days


def _ordered(trades: Sequence[BacktestTrade]) -> list[BacktestTrade]:
    """Return trades in a stable chronological order (deterministic ties)."""
    return sorted(trades, key=lambda t: (t.exit_date, t.symbol, t.strategy))


def win_rate(trades: Sequence[BacktestTrade]) -> float:
    """Percentage of trades with positive net P&L."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.net_pnl > 0)
    return round(100.0 * wins / len(trades), 1)


def profit_factor(trades: Sequence[BacktestTrade]) -> float:
    """Gross profit divided by gross loss (0.0 when there are no losses)."""
    gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
    return round(gross_profit / gross_loss, 2) if gross_loss else 0.0


def expectancy_r(trades: Sequence[BacktestTrade]) -> float:
    """Average R multiple per trade — the expectancy of the edge."""
    if not trades:
        return 0.0
    return round(fmean(t.r_multiple for t in trades), 3)


def avg_win(trades: Sequence[BacktestTrade]) -> float:
    """Average net P&L of winning trades (0.0 when there are none)."""
    wins = [t.net_pnl for t in trades if t.net_pnl > 0]
    return round(fmean(wins), 2) if wins else 0.0


def avg_loss(trades: Sequence[BacktestTrade]) -> float:
    """Average net P&L of losing trades, a non-positive number."""
    losses = [t.net_pnl for t in trades if t.net_pnl < 0]
    return round(fmean(losses), 2) if losses else 0.0


def max_drawdown_pct(trades: Sequence[BacktestTrade], starting_capital: float) -> float:
    """Deepest peak-to-trough drop of the realized equity curve, in percent."""
    equity = starting_capital
    peak = starting_capital
    max_drop = 0.0
    for trade in _ordered(trades):
        equity += trade.net_pnl
        peak = max(peak, equity)
        if peak > 0:
            max_drop = max(max_drop, (peak - equity) / peak)
    return round(100.0 * max_drop, 2)


def longest_losing_streak(trades: Sequence[BacktestTrade]) -> int:
    """Longest run of consecutive losing trades in chronological order."""
    longest = 0
    current = 0
    for trade in _ordered(trades):
        if trade.net_pnl < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def monthly_returns(
    trades: Sequence[BacktestTrade], starting_capital: float
) -> tuple[MonthlyReturn, ...]:
    """Group net P&L by the calendar month each trade closed in."""
    buckets: dict[str, list[BacktestTrade]] = defaultdict(list)
    for trade in trades:
        buckets[f"{trade.exit_date:%Y-%m}"].append(trade)
    rows = [
        MonthlyReturn(
            month=month,
            trades=len(bucket),
            net_pnl=round(sum(t.net_pnl for t in bucket), 2),
            return_pct=(
                round(100.0 * sum(t.net_pnl for t in bucket) / starting_capital, 2)
                if starting_capital
                else 0.0
            ),
        )
        for month, bucket in buckets.items()
    ]
    return tuple(sorted(rows, key=lambda r: r.month))


def per_strategy(trades: Sequence[BacktestTrade]) -> tuple[StrategyPerformance, ...]:
    """Break results down by the strategy that produced each setup."""
    buckets: dict[str, list[BacktestTrade]] = defaultdict(list)
    for trade in trades:
        buckets[trade.strategy].append(trade)
    rows = [
        StrategyPerformance(
            strategy=strategy,
            trades=len(bucket),
            wins=sum(1 for t in bucket if t.net_pnl > 0),
            win_rate=win_rate(bucket),
            expectancy_r=expectancy_r(bucket),
            profit_factor=profit_factor(bucket),
            total_pnl=round(sum(t.net_pnl for t in bucket), 2),
        )
        for strategy, bucket in buckets.items()
    ]
    return tuple(sorted(rows, key=lambda r: r.total_pnl, reverse=True))


def per_regime(trades: Sequence[BacktestTrade]) -> tuple[RegimePerformance, ...]:
    """Segment results by the market regime classified at entry."""
    buckets: dict[str, list[BacktestTrade]] = defaultdict(list)
    for trade in trades:
        buckets[trade.regime or _UNKNOWN_REGIME].append(trade)
    rows = [
        RegimePerformance(
            regime=regime,
            trades=len(bucket),
            win_rate=win_rate(bucket),
            expectancy_r=expectancy_r(bucket),
            total_pnl=round(sum(t.net_pnl for t in bucket), 2),
        )
        for regime, bucket in buckets.items()
    ]
    return tuple(sorted(rows, key=lambda r: r.total_pnl, reverse=True))


def exit_breakdown(trades: Sequence[BacktestTrade]) -> tuple[ExitBreakdown, ...]:
    """Count, expectancy and holding time grouped by fine-grained exit kind."""
    buckets: dict[str, list[BacktestTrade]] = defaultdict(list)
    for trade in trades:
        buckets[trade.exit_kind.value].append(trade)
    rows = [
        ExitBreakdown(
            kind=kind,
            trades=len(bucket),
            wins=sum(1 for t in bucket if t.net_pnl > 0),
            avg_r=round(fmean(t.r_multiple for t in bucket), 3),
            avg_holding_days=round(fmean(_holding_days(t) for t in bucket), 1),
            total_pnl=round(sum(t.net_pnl for t in bucket), 2),
        )
        for kind, bucket in buckets.items()
    ]
    return tuple(sorted(rows, key=lambda r: r.trades, reverse=True))


def winner_r_distribution(trades: Sequence[BacktestTrade]) -> WinnerRDistribution:
    """Bucket winning trades by how far they ran in R versus the 2R target."""
    winners = [t for t in trades if t.net_pnl > 0]
    return WinnerRDistribution(
        winners=len(winners),
        reached_2r=sum(1 for t in winners if t.r_multiple >= 2.0),
        between_1_and_2r=sum(1 for t in winners if 1.0 <= t.r_multiple < 2.0),
        between_0_and_1r=sum(1 for t in winners if 0.0 < t.r_multiple < 1.0),
    )


def exit_analysis(trades: Sequence[BacktestTrade]) -> ExitAnalysis:
    """Assemble the exit diagnostics that explain where expectancy leaks out."""
    if not trades:
        return ExitAnalysis()
    timeouts = [t for t in trades if t.exit_kind is ExitKind.TIMEOUT]
    bull = [t for t in trades if t.regime == _BULL_REGIME]
    return ExitAnalysis(
        by_reason=exit_breakdown(trades),
        winner_r=winner_r_distribution(trades),
        timeout_trades=len(timeouts),
        timeout_profitable=sum(1 for t in timeouts if t.net_pnl > 0),
        avg_entry_slippage_pct=round(
            fmean(
                (t.entry_price - t.signal_price) / t.signal_price * 100.0
                for t in trades
            ),
            4,
        ),
        avg_entry_slippage=round(
            fmean(t.entry_price - t.signal_price for t in trades), 4
        ),
        bull_by_reason=exit_breakdown(bull),
    )


# -- Execution leakage -----------------------------------------------------


def _intended_risk(trade: BacktestTrade) -> float:
    """Per-share risk the strategy planned to take: signal entry minus stop."""
    return trade.signal_price - trade.stop_price


def _stop_leakage(trades: Sequence[BacktestTrade]) -> StopLeakage:
    """Measure how far stop exits fill beyond the stop, split gap/slippage/cost."""
    stops = [t for t in trades if t.exit_reason is ExitReason.STOP]
    if not stops:
        return StopLeakage()
    overshoot_pct = [
        (t.stop_price - t.exit_price) / t.stop_price * 100.0 for t in stops
    ]
    gap_r: list[float] = []
    slippage_r: list[float] = []
    cost_r: list[float] = []
    total_r: list[float] = []
    for trade in stops:
        risk = _intended_risk(trade)
        if risk <= 0:
            continue
        gap = (trade.stop_price - trade.exit_level) / risk  # gapped below the stop
        slip = (trade.exit_level - trade.exit_price) / risk  # sold below the level
        cost = (trade.costs / trade.size) / risk
        gap_r.append(gap)
        slippage_r.append(slip)
        cost_r.append(cost)
        total_r.append(gap + slip + cost)
    return StopLeakage(
        trades=len(stops),
        avg_overshoot_pct=round(fmean(overshoot_pct), 4),
        median_overshoot_pct=round(median(overshoot_pct), 4),
        worst_overshoot_pct=round(max(overshoot_pct), 4),
        avg_overshoot_r=round(fmean(total_r), 4) if total_r else 0.0,
        avg_gap_r=round(fmean(gap_r), 4) if gap_r else 0.0,
        avg_slippage_r=round(fmean(slippage_r), 4) if slippage_r else 0.0,
        avg_cost_r=round(fmean(cost_r), 4) if cost_r else 0.0,
    )


def _target_leakage(trades: Sequence[BacktestTrade]) -> TargetLeakage:
    """Measure why target-hit R falls short of the planned reward."""
    targets = [t for t in trades if t.exit_kind is ExitKind.TARGET_HIT]
    intended: list[float] = []
    recorded: list[float] = []
    intended_risk_r: list[float] = []
    displacement: list[float] = []
    for trade in targets:
        risk = _intended_risk(trade)
        if risk <= 0:
            continue
        intended.append((trade.target_price - trade.signal_price) / risk)
        recorded.append(trade.r_multiple)
        intended_risk_r.append(trade.net_pnl / (risk * trade.size))
        displacement.append(
            (trade.entry_price - trade.signal_price) / trade.signal_price * 100.0
        )
    if not intended:
        return TargetLeakage(trades=len(targets))
    return TargetLeakage(
        trades=len(targets),
        intended_r=round(fmean(intended), 4),
        avg_realised_r=round(fmean(recorded), 4),
        avg_realised_r_intended_risk=round(fmean(intended_risk_r), 4),
        avg_entry_displacement_pct=round(fmean(displacement), 4),
    )


def _r_consistency(trades: Sequence[BacktestTrade]) -> RConsistency:
    """Audit the stored R multiple against R measured on intended risk."""
    recorded = [t.r_multiple for t in trades]
    intended: list[float] = []
    discrepancy: list[float] = []
    for trade in trades:
        risk = _intended_risk(trade) * trade.size
        if risk <= 0:
            continue
        intended_r = trade.net_pnl / risk
        intended.append(intended_r)
        discrepancy.append(abs(trade.r_multiple - intended_r))
    if not recorded:
        return RConsistency()
    return RConsistency(
        # The engine's numerator uses actual fills; its denominator uses the
        # actual entry (entry_price - stop), not the intended (signal - stop).
        numerator_uses_actual_fill=True,
        denominator_uses_actual_entry=True,
        avg_r_recorded=round(fmean(recorded), 4),
        avg_r_intended_risk=round(fmean(intended), 4) if intended else 0.0,
        avg_abs_discrepancy_r=round(fmean(discrepancy), 4) if discrepancy else 0.0,
    )


def _pnl_decomposition(trades: Sequence[BacktestTrade]) -> PnlDecomposition:
    """Split total net P&L into frictionless edge and each execution drag.

    The identity ``net = edge - entry_gap - entry_slip - exit_slip - charges``
    holds exactly (up to rounding), so the drag lines isolate execution cost
    from raw signal edge.
    """
    if not trades:
        return PnlDecomposition()
    edge = sum((t.exit_level - t.signal_price) * t.size for t in trades)
    entry_gap = sum((t.entry_open - t.signal_price) * t.size for t in trades)
    entry_slippage = sum((t.entry_price - t.entry_open) * t.size for t in trades)
    exit_slippage = sum((t.exit_level - t.exit_price) * t.size for t in trades)
    charges = sum(t.costs for t in trades)
    net = edge - entry_gap - entry_slippage - exit_slippage - charges
    return PnlDecomposition(
        net_pnl=round(net, 2),
        strategy_edge=round(edge, 2),
        entry_gap=round(entry_gap, 2),
        entry_slippage=round(entry_slippage, 2),
        exit_slippage=round(exit_slippage, 2),
        charges=round(charges, 2),
    )


def execution_leakage(trades: Sequence[BacktestTrade]) -> ExecutionLeakage:
    """Assemble the execution-leakage diagnostics for a set of closed trades."""
    if not trades:
        return ExecutionLeakage()
    win_fraction = sum(1 for t in trades if t.net_pnl > 0) / len(trades)
    return ExecutionLeakage(
        stops=_stop_leakage(trades),
        targets=_target_leakage(trades),
        r_consistency=_r_consistency(trades),
        pnl=_pnl_decomposition(trades),
        win_rate=round(100.0 * win_fraction, 1),
        # Perfect execution: every win exactly +2R, every loss exactly -1R.
        perfect_expectancy_r=round(win_fraction * 2.0 - (1.0 - win_fraction), 4),
    )
