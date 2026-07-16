"""Pure expectancy maths for the backtester.

Every function here is a deterministic transform of a sequence of closed
:class:`BacktestTrade` records — no I/O, no engines — so the arithmetic (win
rate, profit factor, expectancy in R, drawdown, streaks) can be verified against
hand-computed inputs. The engine owns the simulation; this module owns the maths.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean

from app.backtest.models import (
    BacktestTrade,
    MonthlyReturn,
    RegimePerformance,
    StrategyPerformance,
)

_UNKNOWN_REGIME = "UNKNOWN"


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
