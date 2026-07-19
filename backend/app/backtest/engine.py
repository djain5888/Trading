"""The backtesting engine: a walk-forward replay of the live strategy path.

For each trading day the engine freezes a :class:`LookaheadGuard` at that date
and runs the **same** :class:`StrategyEngine` and :class:`MarketRegimeEngine`
the live pipeline uses, reading only data up to that day. Setups discovered at a
day's close are filled at the *next* day's open (never same-bar), sized by the
paper book's 1%-risk rules, and exited on stop, target or timeout — with gap-aware
fills and realistic costs. It measures expectancy; it never places a real order.

Determinism: no wall-clock or randomness enters the simulation. The trading
calendar, the stored candles and the strategy code path are all deterministic,
so the same data and dates always produce the same report.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from app.backtest import metrics
from app.backtest.guard import LookaheadGuard
from app.backtest.models import (
    BacktestConfig,
    BacktestReport,
    BacktestTrade,
    ExitKind,
)
from app.core.logging import get_logger
from app.core.timezone import INDIA_TZ
from app.market.calendar.service import MarketCalendarService
from app.market.historical.models import Candle, SeriesKey
from app.market.regime.engine import MarketRegimeEngine
from app.market.regime.models import RegimeReport
from app.paper.models import ExitReason
from app.strategy.engine import StrategyEngine
from app.strategy.models import StrategySetup

logger = get_logger(__name__)

#: A tz-aware lower bound used to probe whether any candle is stored at all.
_EPOCH = datetime(1970, 1, 1, tzinfo=INDIA_TZ)


@dataclass
class _Diag:
    """Running diagnostics so a zero-trade replay is never silent."""

    candles_seen: bool = False
    days_replayed: int = 0
    days_with_data: int = 0
    total_setups: int = 0
    trades_opened: int = 0
    skips: Counter[str] = field(default_factory=Counter)


@dataclass
class _OpenPosition:
    """A live simulated position (mutable during the walk)."""

    symbol: str
    strategy: str
    regime: str | None
    entry_date: date
    signal_price: float
    entry_price: float
    stop: float
    target: float
    size: float


@dataclass(frozen=True)
class _PendingSetup:
    """A setup awaiting a next-day-open fill, tagged with the entry regime."""

    setup: StrategySetup
    regime: str | None


class BacktestEngine:
    """Replays strategy setups over history and reports their expectancy."""

    def __init__(
        self,
        *,
        data: LookaheadGuard,
        strategy_engine: StrategyEngine,
        regime_engine: MarketRegimeEngine,
        calendar: MarketCalendarService,
        config: BacktestConfig | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            data: The look-ahead-guarded data view shared by the sub-engines.
            strategy_engine: The live strategy engine, wired to ``data``.
            regime_engine: The live regime engine, wired to ``data``.
            calendar: Trading-calendar authority for the replay days.
            config: Market, risk and cost configuration.
        """
        self._data = data
        self._strategy = strategy_engine
        self._regime = regime_engine
        self._calendar = calendar
        self._config = config or BacktestConfig()

    @property
    def config(self) -> BacktestConfig:
        """Return the engine's configuration."""
        return self._config

    async def run(
        self, symbols: Sequence[str], start: date, end: date
    ) -> BacktestReport:
        """Replay ``[start, end]`` and return the measured expectancy.

        Args:
            symbols: The trading universe.
            start: First calendar date to replay (inclusive).
            end: Last calendar date to replay (inclusive).

        Returns:
            A :class:`BacktestReport` of the run.

        Raises:
            ValueError: If ``end`` precedes ``start``.
        """
        if end < start:
            raise ValueError("Backtest 'end' must not precede 'start'.")
        universe = tuple(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
        days = self._trading_days(start, end)

        open_positions: list[_OpenPosition] = []
        pending: list[_PendingSetup] = []
        closed: list[BacktestTrade] = []
        realized = 0.0
        diag = _Diag(candles_seen=await self._any_stored(universe, end))

        for day in days:
            self._data.freeze_at(day)
            realized += await self._exit_day(open_positions, closed, day)
            await self._open_pending(pending, open_positions, day, realized, diag)
            regime = await self._classify_regime(universe, day)
            report = await self._strategy.analyze(
                universe,
                exchange=self._config.exchange,
                interval=self._config.interval,
                end=self._eod(day),
                regime=regime.report,
            )
            diag.days_replayed += 1
            diag.total_setups += len(report.setups)
            if report.available:
                diag.days_with_data += 1
            if logger.isEnabledFor(logging.DEBUG):
                ready = await self._ready_count(universe, day)
                logger.debug(
                    "Backtest %s: symbols_with_data=%d setups=%d",
                    day,
                    ready,
                    len(report.setups),
                )
            pending = [
                _PendingSetup(setup=s, regime=regime.label) for s in report.setups
            ]

        realized += await self._finalize(open_positions, closed, days)
        self._log_diagnostics(diag, universe, start, end)
        return self._build_report(universe, start, end, closed, realized)

    # -- Daily steps -------------------------------------------------------

    async def _exit_day(
        self,
        open_positions: list[_OpenPosition],
        closed: list[BacktestTrade],
        day: date,
    ) -> float:
        """Settle exits for positions opened before ``day``; return realized P&L."""
        realized = 0.0
        survivors: list[_OpenPosition] = []
        for position in open_positions:
            trade = await self._try_exit(position, day, force=False)
            if trade is None:
                survivors.append(position)
            else:
                closed.append(trade)
                realized += trade.net_pnl
        open_positions[:] = survivors
        return realized

    async def _open_pending(
        self,
        pending: Sequence[_PendingSetup],
        open_positions: list[_OpenPosition],
        day: date,
        realized: float,
        diag: _Diag,
    ) -> None:
        """Fill yesterday's setups at today's open, respecting the risk limits."""
        equity = self._config.paper.starting_capital + realized
        held = {position.symbol for position in open_positions}
        for item in pending:  # already ranked best-first by the strategy engine
            if len(open_positions) >= self._config.paper.max_positions:
                diag.skips["max_positions"] += 1
                break
            setup = item.setup
            symbol = setup.symbol.strip().upper()
            if symbol in held:
                diag.skips["already_open"] += 1
                continue
            if setup.confidence < self._config.paper.min_confidence:
                diag.skips["below_min_confidence"] += 1
                continue
            if not (setup.stop < setup.entry < setup.target):
                diag.skips["invalid_setup_levels"] += 1
                continue
            candle = await self._day_candle(symbol, day)
            if candle is None:
                diag.skips["no_next_day_candle"] += 1
                continue
            entry = self._entry_fill(float(candle.open))
            if not (setup.stop < entry < setup.target):
                diag.skips["gapped_out_of_levels"] += 1
                continue  # opened past the stop/target — no valid long to take
            size = self._size(equity, entry, setup.stop)
            if size <= 0:
                diag.skips["zero_size"] += 1
                continue
            open_positions.append(
                _OpenPosition(
                    symbol=symbol,
                    strategy=setup.strategy,
                    regime=item.regime,
                    entry_date=day,
                    signal_price=setup.entry,
                    entry_price=entry,
                    stop=setup.stop,
                    target=setup.target,
                    size=size,
                )
            )
            held.add(symbol)
            diag.trades_opened += 1

    async def _finalize(
        self,
        open_positions: list[_OpenPosition],
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
            trade = await self._try_exit(position, last, force=True)
            if trade is not None:
                closed.append(trade)
                realized += trade.net_pnl
        open_positions.clear()
        return realized

    # -- Exit logic (gap-aware, stop-before-target) ------------------------

    async def _try_exit(
        self, position: _OpenPosition, day: date, *, force: bool
    ) -> BacktestTrade | None:
        """Return a closed trade if an exit level is hit on ``day``, else ``None``.

        Worst case first: a gap through the stop fills at the (worse) open, not
        the stop; only then is an intraday stop, then target, then timeout
        considered. ``force`` closes an unresolved position at the day's close.
        """
        candle = await self._day_candle(position.symbol, day)
        if candle is None:
            if force:  # no candle at the window end — close flat at the entry price
                return self._close(
                    position,
                    day,
                    position.entry_price,
                    ExitReason.TIMEOUT,
                    ExitKind.END_OF_BACKTEST,
                )
            return None
        if candle.timestamp.date() <= position.entry_date and not force:
            return None  # never exit on the entry bar (hold the open day)
        open_p, low, high, close = (
            float(candle.open),
            float(candle.low),
            float(candle.high),
            float(candle.close),
        )
        if open_p <= position.stop:  # gapped through the stop → filled worse
            return self._close(
                position, day, open_p, ExitReason.STOP, ExitKind.GAP_EXIT
            )
        if low <= position.stop:
            return self._close(
                position, day, position.stop, ExitReason.STOP, ExitKind.STOP_HIT
            )
        if open_p >= position.target:  # gapped through the target
            return self._close(
                position, day, open_p, ExitReason.TARGET, ExitKind.GAP_EXIT
            )
        if high >= position.target:
            return self._close(
                position, day, position.target, ExitReason.TARGET, ExitKind.TARGET_HIT
            )
        held_days = (day - position.entry_date).days
        if held_days >= self._config.paper.max_holding_days:
            return self._close(
                position, day, close, ExitReason.TIMEOUT, ExitKind.TIMEOUT
            )
        if force:  # survived to the end of the window — force a flat close
            return self._close(
                position, day, close, ExitReason.TIMEOUT, ExitKind.END_OF_BACKTEST
            )
        return None

    def _close(
        self,
        position: _OpenPosition,
        day: date,
        level: float,
        reason: ExitReason,
        kind: ExitKind,
    ) -> BacktestTrade:
        """Build the closed-trade record, net of slippage and charges."""
        exit_fill = self._exit_fill(level)
        gross = (exit_fill - position.entry_price) * position.size
        charges = self._charges(
            position.entry_price * position.size, exit_fill * position.size
        )
        net = gross - charges
        risk_amount = (position.entry_price - position.stop) * position.size
        r_multiple = net / risk_amount if risk_amount > 0 else 0.0
        return BacktestTrade(
            symbol=position.symbol,
            strategy=position.strategy,
            regime=position.regime,
            entry_date=position.entry_date,
            exit_date=day,
            signal_price=round(position.signal_price, 4),
            entry_price=round(position.entry_price, 4),
            exit_price=round(exit_fill, 4),
            stop_price=round(position.stop, 4),
            target_price=round(position.target, 4),
            size=position.size,
            exit_reason=reason,
            exit_kind=kind,
            gross_pnl=round(gross, 2),
            costs=round(charges, 2),
            net_pnl=round(net, 2),
            r_multiple=round(r_multiple, 3),
        )

    # -- Cost model --------------------------------------------------------

    def _entry_fill(self, raw_open: float) -> float:
        """Apply entry slippage (a buy fills higher)."""
        return raw_open * (1.0 + self._config.costs.slippage_pct / 100.0)

    def _exit_fill(self, level: float) -> float:
        """Apply exit slippage (a sell fills lower)."""
        return level * (1.0 - self._config.costs.slippage_pct / 100.0)

    def _charges(self, entry_notional: float, exit_notional: float) -> float:
        """Return brokerage/STT charges levied on both sides' notional."""
        return (entry_notional + exit_notional) * self._config.costs.charge_pct / 100.0

    def _size(self, equity: float, entry: float, stop: float) -> float:
        """Return the size risking ``risk_pct`` of equity to the stop (paper rule)."""
        risk_per_share = entry - stop
        if risk_per_share <= 0:
            return 0.0
        return round(equity * (self._config.paper.risk_pct / 100.0) / risk_per_share, 4)

    # -- Regime tagging ----------------------------------------------------

    async def _classify_regime(
        self, universe: Sequence[str], day: date
    ) -> _RegimeAtEntry:
        """Classify the regime as of ``day``'s close for tagging and blending."""
        report = await self._regime.analyze(
            universe,
            exchange=self._config.exchange,
            interval=self._config.interval,
            end=self._eod(day),
        )
        label = (
            report.regime.value
            if report.available and report.regime is not None
            else None
        )
        return _RegimeAtEntry(report=report, label=label)

    # -- Data helpers ------------------------------------------------------

    async def _day_candle(self, symbol: str, day: date) -> Candle | None:
        """Return the candle dated exactly ``day`` for ``symbol``, or ``None``."""
        key = SeriesKey(
            symbol=symbol,
            exchange=self._config.exchange,
            interval=self._config.interval,
        )
        start = datetime(day.year, day.month, day.day, tzinfo=INDIA_TZ)
        candles = await self._data.get_candles(key, start, self._eod(day))
        for candle in reversed(candles):
            if candle.timestamp.date() == day:
                return candle
        return None

    def _trading_days(self, start: date, end: date) -> list[date]:
        """Return the trading days in ``[start, end]`` for the exchange."""
        days: list[date] = []
        day = start
        while day <= end:
            if self._calendar.is_trading_day(day, self._config.exchange):
                days.append(day)
            day += timedelta(days=1)
        return days

    @staticmethod
    def _eod(day: date) -> datetime:
        """Return an end-of-day IST timestamp so a day's candle is inclusive."""
        return datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=INDIA_TZ)

    async def _any_stored(self, universe: Sequence[str], end: date) -> bool:
        """Return whether the store holds any candle for the universe up to ``end``.

        This distinguishes the "no candles were ever loaded" failure (the store
        is empty for this process) from "candles exist but history is too short".
        """
        end_dt = self._eod(end)
        for symbol in universe:
            key = SeriesKey(
                symbol=symbol,
                exchange=self._config.exchange,
                interval=self._config.interval,
            )
            if await self._data.get_candles(key, _EPOCH, end_dt):
                return True
        return False

    async def _ready_count(self, universe: Sequence[str], day: date) -> int:
        """Count symbols with enough history for the strategy as of ``day``.

        Returns ``-1`` when the strategy's lookback is unknown (e.g. a test stub).
        """
        try:
            config = self._strategy.config
            history_days, min_candles = config.history_days, config.min_candles
        except AttributeError:
            return -1
        window_start = self._eod(day) - timedelta(days=history_days)
        ready = 0
        for symbol in universe:
            key = SeriesKey(
                symbol=symbol,
                exchange=self._config.exchange,
                interval=self._config.interval,
            )
            candles = await self._data.get_candles(key, window_start, self._eod(day))
            if len(candles) >= min_candles:
                ready += 1
        return ready

    def _log_diagnostics(
        self, diag: _Diag, universe: Sequence[str], start: date, end: date
    ) -> None:
        """Emit a summary and a loud warning when a replay produced no trades."""
        logger.info(
            "Backtest %s..%s over %d symbol(s): %d day(s) replayed, %d with data, "
            "%d setup(s) generated, %d trade(s) opened.",
            start,
            end,
            len(universe),
            diag.days_replayed,
            diag.days_with_data,
            diag.total_setups,
            diag.trades_opened,
        )
        if not diag.candles_seen:
            logger.warning(
                "Backtest saw NO stored candles for [%s] (%s/%s). Likely cause: the "
                "candles were never loaded into this process's store (run an import "
                "first), or the symbols/exchange/interval/date window do not match "
                "the stored data.",
                ", ".join(universe) or "<empty universe>",
                self._config.exchange.value,
                self._config.interval.value,
            )
        elif diag.total_setups == 0:
            logger.warning(
                "Backtest saw candles but generated 0 setups over %s..%s — history "
                "may be shorter than the strategy lookback, or no setup qualified.",
                start,
                end,
            )
        elif diag.trades_opened == 0:
            logger.warning(
                "Backtest generated %d setup(s) but opened 0 trades; skip reasons: %s.",
                diag.total_setups,
                dict(diag.skips),
            )

    # -- Reporting ---------------------------------------------------------

    def _build_report(
        self,
        universe: tuple[str, ...],
        start: date,
        end: date,
        closed: list[BacktestTrade],
        realized: float,
    ) -> BacktestReport:
        """Assemble the final report from the closed trades and realized P&L."""
        ordered = sorted(
            closed, key=lambda t: (t.entry_date, t.exit_date, t.symbol, t.strategy)
        )
        starting = self._config.paper.starting_capital
        total_pnl = round(realized, 2)
        return BacktestReport(
            start=start,
            end=end,
            symbols=universe,
            starting_capital=starting,
            ending_equity=round(starting + total_pnl, 2),
            total_return_pct=(
                round(100.0 * total_pnl / starting, 2) if starting else 0.0
            ),
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
            generated_at=self._eod(end),
        )


@dataclass(frozen=True)
class _RegimeAtEntry:
    """A day's regime report plus its label, used for blending and tagging."""

    report: RegimeReport
    label: str | None
