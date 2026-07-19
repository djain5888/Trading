"""The baseline engine: a walk-forward runner shared by every baseline.

It reuses the exact execution realism of the main backtest — next-day-open
fills, entry/exit slippage, brokerage/STT charges, 1%-risk sizing and the
look-ahead guard — and delegates only the entry/exit *decisions* to a
:class:`Baseline`. That makes each baseline's report directly comparable to the
main strategy backtest and to the other baselines.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from app.backtest import execution
from app.backtest.baselines.base import Baseline, BaselineContext, EntryIntent
from app.backtest.guard import LookaheadGuard
from app.backtest.models import BacktestConfig, BacktestReport, BacktestTrade, ExitKind
from app.core.logging import get_logger
from app.indicators.engine import IndicatorEngine
from app.market.calendar.service import MarketCalendarService
from app.paper.models import ExitReason

logger = get_logger(__name__)


@dataclass
class _BaselinePosition:
    """A live baseline position (mutable: the trailing stop ratchets up)."""

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
    uses_stop: bool
    trailing_distance: float | None
    max_hold: int | None
    running_high: float


class BaselineEngine:
    """Runs one :class:`Baseline` over history with shared execution realism."""

    def __init__(
        self,
        *,
        data: LookaheadGuard,
        indicators: IndicatorEngine,
        calendar: MarketCalendarService,
        baseline: Baseline,
        config: BacktestConfig,
        history_days: int,
    ) -> None:
        """Initialise the engine.

        Args:
            data: The look-ahead-guarded data view.
            indicators: The shared indicator engine (pure on candles).
            calendar: Trading-calendar authority for the replay days.
            baseline: The baseline decision layer.
            config: Execution config (costs, sizing, exchange, interval).
            history_days: Calendar days of history each plan reads.
        """
        self._data = data
        self._indicators = indicators
        self._calendar = calendar
        self._baseline = baseline
        self._config = config
        self._history_days = history_days

    async def run(
        self, symbols: Sequence[str], start: date, end: date
    ) -> BacktestReport:
        """Replay ``[start, end]`` for the baseline and return its report.

        Raises:
            ValueError: If ``end`` precedes ``start``.
        """
        if end < start:
            raise ValueError("Baseline 'end' must not precede 'start'.")
        universe = tuple(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
        days = execution.trading_days(self._calendar, self._config.exchange, start, end)
        ctx = BaselineContext(
            self._data, self._indicators, self._config, self._history_days
        )

        open_positions: list[_BaselinePosition] = []
        pending: list[EntryIntent] = []
        closed: list[BacktestTrade] = []
        realized = 0.0

        for day in days:
            self._data.freeze_at(day)
            held = {p.symbol for p in open_positions} | {e.symbol for e in pending}
            plan = await self._baseline.plan(ctx, day, universe, frozenset(held))
            realized += await self._exit_day(open_positions, closed, day, plan.exits)
            await self._open(pending, open_positions, day, realized)
            pending = list(plan.entries)

        realized += await self._finalize(open_positions, closed, days)
        logger.info(
            "Baseline '%s' %s..%s over %d symbol(s): %d trade(s).",
            self._baseline.name,
            start,
            end,
            len(universe),
            len(closed),
        )
        return execution.build_report(
            universe=universe,
            start=start,
            end=end,
            closed=closed,
            realized=realized,
            config=self._config,
        )

    # -- Daily steps -------------------------------------------------------

    async def _exit_day(
        self,
        open_positions: list[_BaselinePosition],
        closed: list[BacktestTrade],
        day: date,
        exits: frozenset[str],
    ) -> float:
        """Settle exits for positions opened before ``day``; return realized P&L."""
        realized = 0.0
        survivors: list[_BaselinePosition] = []
        for position in open_positions:
            trade = await self._try_exit(position, day, exits, force=False)
            if trade is None:
                survivors.append(position)
            else:
                closed.append(trade)
                realized += trade.net_pnl
        open_positions[:] = survivors
        return realized

    async def _open(
        self,
        pending: Sequence[EntryIntent],
        open_positions: list[_BaselinePosition],
        day: date,
        realized: float,
    ) -> None:
        """Fill yesterday's intents at today's open with 1%-risk sizing."""
        equity = self._config.paper.starting_capital + realized
        held = {position.symbol for position in open_positions}
        for intent in pending:
            if intent.symbol in held:
                continue
            candle = await execution.day_candle(
                self._data,
                intent.symbol,
                self._config.exchange,
                self._config.interval,
                day,
            )
            if candle is None:
                continue
            raw_open = float(candle.open)
            entry = execution.entry_fill(raw_open, self._config.costs)
            stop = entry - intent.stop_distance
            if stop <= 0:
                continue
            size = execution.position_size(
                equity, entry, stop, self._config.paper.risk_pct
            )
            if size <= 0:
                continue
            open_positions.append(
                _BaselinePosition(
                    symbol=intent.symbol,
                    strategy=self._baseline.name,
                    regime=None,
                    entry_date=day,
                    signal_price=intent.signal_price,
                    entry_open=raw_open,
                    entry_price=entry,
                    stop=stop,
                    target=entry,  # baselines carry no target; placeholder for R model
                    size=size,
                    uses_stop=intent.uses_stop,
                    trailing_distance=intent.trailing_distance,
                    max_hold=intent.max_hold,
                    running_high=entry,
                )
            )
            held.add(intent.symbol)

    async def _finalize(
        self,
        open_positions: list[_BaselinePosition],
        closed: list[BacktestTrade],
        days: Sequence[date],
    ) -> float:
        """Force-close any positions still open at the end of the window."""
        if not days:
            return 0.0
        last = days[-1]
        self._data.freeze_at(last)
        realized = 0.0
        for position in open_positions:
            trade = await self._try_exit(position, last, frozenset(), force=True)
            if trade is not None:
                closed.append(trade)
                realized += trade.net_pnl
        open_positions.clear()
        return realized

    # -- Exit logic --------------------------------------------------------

    async def _try_exit(
        self,
        position: _BaselinePosition,
        day: date,
        exits: frozenset[str],
        *,
        force: bool,
    ) -> BacktestTrade | None:
        """Close ``position`` on ``day`` if a rule triggers, else ``None``.

        Precedence: gap/trailing stop (when armed) → forced exit (rebalance or
        signal) → holding-period timeout → end-of-window force close. A trailing
        stop ratchets up on the day's high *after* the exit check, so it only
        affects subsequent days (no same-bar look-ahead).
        """
        candle = await execution.day_candle(
            self._data,
            position.symbol,
            self._config.exchange,
            self._config.interval,
            day,
        )
        if candle is None:
            if force:
                return self._close(
                    position,
                    day,
                    position.entry_price,
                    ExitReason.TIMEOUT,
                    ExitKind.END_OF_BACKTEST,
                )
            return None
        if candle.timestamp.date() <= position.entry_date and not force:
            return None  # never exit on the entry bar
        open_p, low, high, close_p = (
            float(candle.open),
            float(candle.low),
            float(candle.high),
            float(candle.close),
        )
        if position.uses_stop:
            if open_p <= position.stop:  # gapped through the stop
                return self._close(
                    position, day, open_p, ExitReason.STOP, ExitKind.GAP_EXIT
                )
            if low <= position.stop:
                return self._close(
                    position, day, position.stop, ExitReason.STOP, ExitKind.STOP_HIT
                )
        if position.symbol in exits:  # rebalance / signal exit at the close
            return self._close(
                position, day, close_p, ExitReason.TIMEOUT, ExitKind.TIMEOUT
            )
        held_days = (day - position.entry_date).days
        if position.max_hold is not None and held_days >= position.max_hold:
            return self._close(
                position, day, close_p, ExitReason.TIMEOUT, ExitKind.TIMEOUT
            )
        if force:
            return self._close(
                position, day, close_p, ExitReason.TIMEOUT, ExitKind.END_OF_BACKTEST
            )
        if position.uses_stop and position.trailing_distance is not None:
            position.running_high = max(position.running_high, high)
            position.stop = max(
                position.stop, position.running_high - position.trailing_distance
            )
        return None

    def _close(
        self,
        position: _BaselinePosition,
        day: date,
        level: float,
        reason: ExitReason,
        kind: ExitKind,
    ) -> BacktestTrade:
        """Build the closed-trade record via the shared execution core."""
        return execution.build_trade(
            position,
            exit_date=day,
            level=level,
            reason=reason,
            kind=kind,
            costs=self._config.costs,
        )
