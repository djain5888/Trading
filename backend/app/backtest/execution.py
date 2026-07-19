"""Shared execution mechanics for the backtester and its baselines.

Fills, slippage, charges, 1%-risk sizing, trade construction, the trading-day
calendar walk and the report build all live here so the main strategy backtest
and the baseline search share *identical* realism. A baseline's metrics are
therefore directly comparable to the main strategy's — the whole point of the
baseline search.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Protocol

from app.backtest import metrics
from app.backtest.guard import LookaheadGuard
from app.backtest.models import (
    BacktestConfig,
    BacktestCosts,
    BacktestReport,
    BacktestTrade,
    ExitKind,
)
from app.core.timezone import INDIA_TZ
from app.market.calendar.service import MarketCalendarService
from app.market.enums import Exchange, Interval
from app.market.historical.models import Candle, SeriesKey
from app.paper.models import ExitReason

#: A tz-aware lower bound used to probe whether any candle is stored at all.
_EPOCH = datetime(1970, 1, 1, tzinfo=INDIA_TZ)


class PositionLike(Protocol):
    """The fields the trade builder reads from any open position."""

    symbol: str
    strategy: str
    regime: str | None
    entry_date: date
    signal_price: float
    entry_open: float
    entry_price: float
    stop: float
    target: float
    size: float


def eod(day: date) -> datetime:
    """Return an end-of-day IST timestamp so a day's candle is inclusive."""
    return datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=INDIA_TZ)


def entry_fill(raw_open: float, costs: BacktestCosts) -> float:
    """Apply entry slippage (a buy fills higher)."""
    return raw_open * (1.0 + costs.slippage_pct / 100.0)


def exit_fill(level: float, costs: BacktestCosts) -> float:
    """Apply exit slippage (a sell fills lower)."""
    return level * (1.0 - costs.slippage_pct / 100.0)


def charges(entry_notional: float, exit_notional: float, costs: BacktestCosts) -> float:
    """Return brokerage/STT charges levied on both sides' notional."""
    return (entry_notional + exit_notional) * costs.charge_pct / 100.0


def position_size(equity: float, entry: float, stop: float, risk_pct: float) -> float:
    """Return the size risking ``risk_pct`` of equity to the stop (paper rule)."""
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0.0
    return round(equity * (risk_pct / 100.0) / risk_per_share, 4)


def build_trade(
    position: PositionLike,
    *,
    exit_date: date,
    level: float,
    reason: ExitReason,
    kind: ExitKind,
    costs: BacktestCosts,
) -> BacktestTrade:
    """Build a closed-trade record, net of slippage and charges."""
    fill = exit_fill(level, costs)
    gross = (fill - position.entry_price) * position.size
    charge = charges(position.entry_price * position.size, fill * position.size, costs)
    net = gross - charge
    risk_amount = (position.entry_price - position.stop) * position.size
    r_multiple = net / risk_amount if risk_amount > 0 else 0.0
    return BacktestTrade(
        symbol=position.symbol,
        strategy=position.strategy,
        regime=position.regime,
        entry_date=position.entry_date,
        exit_date=exit_date,
        signal_price=round(position.signal_price, 4),
        entry_open=round(position.entry_open, 4),
        entry_price=round(position.entry_price, 4),
        exit_level=round(level, 4),
        exit_price=round(fill, 4),
        stop_price=round(position.stop, 4),
        target_price=round(position.target, 4),
        size=position.size,
        exit_reason=reason,
        exit_kind=kind,
        gross_pnl=round(gross, 2),
        costs=round(charge, 2),
        net_pnl=round(net, 2),
        r_multiple=round(r_multiple, 3),
    )


def trading_days(
    calendar: MarketCalendarService, exchange: Exchange, start: date, end: date
) -> list[date]:
    """Return the trading days in ``[start, end]`` for the exchange."""
    days: list[date] = []
    day = start
    while day <= end:
        if calendar.is_trading_day(day, exchange):
            days.append(day)
        day += timedelta(days=1)
    return days


async def day_candle(
    data: LookaheadGuard, symbol: str, exchange: Exchange, interval: Interval, day: date
) -> Candle | None:
    """Return the candle dated exactly ``day`` for ``symbol``, or ``None``."""
    key = SeriesKey(symbol=symbol, exchange=exchange, interval=interval)
    start = datetime(day.year, day.month, day.day, tzinfo=INDIA_TZ)
    candles = await data.get_candles(key, start, eod(day))
    for candle in reversed(candles):
        if candle.timestamp.date() == day:
            return candle
    return None


async def any_stored(
    data: LookaheadGuard,
    universe: Sequence[str],
    exchange: Exchange,
    interval: Interval,
    end: date,
) -> bool:
    """Return whether the store holds any candle for the universe up to ``end``."""
    end_dt = eod(end)
    for symbol in universe:
        key = SeriesKey(symbol=symbol, exchange=exchange, interval=interval)
        if await data.get_candles(key, _EPOCH, end_dt):
            return True
    return False


def build_report(
    *,
    universe: tuple[str, ...],
    start: date,
    end: date,
    closed: Sequence[BacktestTrade],
    realized: float,
    config: BacktestConfig,
) -> BacktestReport:
    """Assemble the final report from closed trades and realized P&L."""
    ordered = sorted(
        closed, key=lambda t: (t.entry_date, t.exit_date, t.symbol, t.strategy)
    )
    starting = config.paper.starting_capital
    total_pnl = round(realized, 2)
    return BacktestReport(
        start=start,
        end=end,
        symbols=universe,
        starting_capital=starting,
        ending_equity=round(starting + total_pnl, 2),
        total_return_pct=round(100.0 * total_pnl / starting, 2) if starting else 0.0,
        trades=len(ordered),
        wins=sum(1 for t in ordered if t.net_pnl > 0),
        losses=sum(1 for t in ordered if t.net_pnl < 0),
        win_rate=metrics.win_rate(ordered),
        avg_win=metrics.avg_win(ordered),
        avg_loss=metrics.avg_loss(ordered),
        expectancy_r=metrics.expectancy_r(ordered),
        profit_factor=metrics.profit_factor(ordered),
        max_drawdown_pct=metrics.max_drawdown_pct(ordered, starting),
        longest_losing_streak=metrics.longest_losing_streak(ordered),
        total_pnl=total_pnl,
        per_strategy=metrics.per_strategy(ordered),
        per_regime=metrics.per_regime(ordered),
        monthly=metrics.monthly_returns(ordered, starting),
        closed_trades=tuple(ordered),
        exit_analysis=metrics.exit_analysis(ordered),
        execution_leakage=metrics.execution_leakage(ordered),
        generated_at=eod(end),
    )
