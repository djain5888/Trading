"""The paper-trading engine.

Opens hypothetical trades from strategy setups, marks open positions to market
on each run, closes them on a stop hit, target hit or max-holding timeout, and
persists everything through a :class:`PaperTradeRepository`. It measures whether
setups work; it never places a real order.

Determinism: entry timestamps come from the entry candle and ids are derived
from ``symbol/strategy/entry-time``, so identical inputs produce identical
trades. Missing price data holds a position rather than force-closing it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from statistics import fmean

from app.core.clock import Clock
from app.core.logging import get_logger
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.models import Candle, SeriesKey
from app.paper.models import (
    ExitReason,
    PaperConfig,
    PaperReport,
    PaperTrade,
    StrategyStats,
    TradeStatus,
)
from app.paper.repository import PaperTradeRepository
from app.strategy.models import StrategySetup

logger = get_logger(__name__)


class PaperTradingEngine:
    """Tracks hypothetical trades opened from strategy setups."""

    def __init__(
        self,
        repository: PaperTradeRepository,
        data_engine: HistoricalDataEngine,
        clock: Clock,
        config: PaperConfig | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            repository: Persistence for paper trades.
            data_engine: Source of latest candles for marking positions.
            clock: Time source for the report timestamp.
            config: Portfolio and risk configuration.
        """
        self._repo = repository
        self._data = data_engine
        self._clock = clock
        self._config = config or PaperConfig()

    @property
    def config(self) -> PaperConfig:
        """Return the engine's configuration."""
        return self._config

    async def run(
        self,
        setups: Sequence[StrategySetup],
        *,
        exchange: Exchange = Exchange.NSE,
        interval: Interval = Interval.ONE_DAY,
        end: datetime | None = None,
    ) -> PaperReport:
        """Update open trades, then open new ones from ``setups``.

        Args:
            setups: Today's ranked setups (best first).
            exchange: Exchange the setups trade on.
            interval: Candle interval to mark/read.
            end: Report timestamp; defaults to the clock's now.

        Returns:
            The refreshed :class:`PaperReport`.
        """
        now = end or self._clock.now()
        trades = await self._repo.all()
        open_symbols: set[str] = set()
        realized = 0.0
        for trade in trades:
            if trade.status is TradeStatus.CLOSED:
                realized += trade.pnl
                continue
            updated = await self._update(trade)
            await self._repo.upsert(updated)
            if updated.status is TradeStatus.CLOSED:
                realized += updated.pnl
            else:
                open_symbols.add(updated.symbol)

        await self._open_new(setups, exchange, interval, open_symbols, realized)
        return await self.report(now)

    async def report(self, end: datetime | None = None) -> PaperReport:
        """Build a performance report from the persisted trades (no updates)."""
        now = end or self._clock.now()
        trades = await self._repo.all()
        open_trades = tuple(t for t in trades if t.status is TradeStatus.OPEN)
        closed = sorted(
            (t for t in trades if t.status is TradeStatus.CLOSED),
            key=lambda t: t.closed_at or now,
            reverse=True,
        )
        return self._build_report(now, open_trades, tuple(closed))

    # -- Position updates --------------------------------------------------

    async def _update(self, trade: PaperTrade) -> PaperTrade:
        """Mark a trade to the latest candle, closing it if a level is hit."""
        candle = await self._latest(trade.symbol, trade.exchange, trade.interval)
        if candle is None or candle.timestamp <= trade.opened_at:
            return trade  # missing/no new price data -> hold, never force-close
        low = float(candle.low)
        high = float(candle.high)
        close = float(candle.close)
        if low <= trade.stop_price:  # worst case first: assume stop before target
            return trade.closed_with(
                trade.stop_price, ExitReason.STOP, candle.timestamp
            )
        if high >= trade.target_price:
            return trade.closed_with(
                trade.target_price, ExitReason.TARGET, candle.timestamp
            )
        held_days = (candle.timestamp.date() - trade.opened_at.date()).days
        if held_days >= self._config.max_holding_days:
            return trade.closed_with(close, ExitReason.TIMEOUT, candle.timestamp)
        return trade.marked(close)

    async def _open_new(
        self,
        setups: Sequence[StrategySetup],
        exchange: Exchange,
        interval: Interval,
        open_symbols: set[str],
        realized: float,
    ) -> None:
        """Open trades from the best setups, respecting the position limits."""
        equity = self._config.starting_capital + realized
        for setup in setups:
            if len(open_symbols) >= self._config.max_positions:
                break
            symbol = setup.symbol.strip().upper()
            if symbol in open_symbols or setup.confidence < self._config.min_confidence:
                continue
            if not (setup.stop < setup.entry < setup.target):
                continue
            candle = await self._latest(symbol, exchange, interval)
            if candle is None:
                continue  # no price data -> cannot open
            size = self._size(equity, setup.entry, setup.stop)
            if size <= 0:
                continue
            trade = PaperTrade(
                id=f"{symbol}-{setup.strategy}-{candle.timestamp:%Y%m%dT%H%M%S}",
                symbol=symbol,
                strategy=setup.strategy,
                exchange=exchange,
                interval=interval,
                entry_price=setup.entry,
                stop_price=setup.stop,
                target_price=setup.target,
                size=size,
                opened_at=candle.timestamp,
                last_price=setup.entry,
            )
            await self._repo.upsert(trade)
            open_symbols.add(symbol)

    def _size(self, equity: float, entry: float, stop: float) -> float:
        """Return the size that risks ``risk_pct`` of equity to the stop."""
        risk_per_share = entry - stop
        if risk_per_share <= 0:
            return 0.0
        return round(equity * (self._config.risk_pct / 100.0) / risk_per_share, 4)

    async def _latest(
        self, symbol: str, exchange: Exchange, interval: Interval
    ) -> Candle | None:
        """Return the latest stored candle for a symbol, or ``None``."""
        key = SeriesKey(symbol=symbol, exchange=exchange, interval=interval)
        return await self._data.get_latest(key)

    # -- Reporting ---------------------------------------------------------

    def _build_report(
        self,
        now: datetime,
        open_trades: tuple[PaperTrade, ...],
        closed: tuple[PaperTrade, ...],
    ) -> PaperReport:
        """Compute performance metrics from open and closed trades."""
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl < 0]
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        realized = sum(t.pnl for t in closed)
        open_pnl = sum(t.pnl for t in open_trades)
        return PaperReport(
            generated_at=now,
            starting_capital=self._config.starting_capital,
            equity=self._config.starting_capital + realized,
            open_positions=open_trades,
            closed_trades=closed,
            total_pnl=round(realized + open_pnl, 2),
            realized_pnl=round(realized, 2),
            open_pnl=round(open_pnl, 2),
            win_rate=round(100.0 * len(wins) / len(closed), 1) if closed else 0.0,
            avg_win=round(fmean(t.pnl for t in wins), 2) if wins else 0.0,
            avg_loss=round(fmean(t.pnl for t in losses), 2) if losses else 0.0,
            profit_factor=round(gross_profit / gross_loss, 2) if gross_loss else 0.0,
            wins=len(wins),
            losses=len(losses),
            per_strategy=self._per_strategy(closed),
        )

    @staticmethod
    def _per_strategy(closed: tuple[PaperTrade, ...]) -> tuple[StrategyStats, ...]:
        """Aggregate closed trades into a per-strategy breakdown."""
        buckets: dict[str, list[PaperTrade]] = defaultdict(list)
        for trade in closed:
            buckets[trade.strategy].append(trade)
        stats = [
            StrategyStats(
                strategy=strategy,
                trades=len(trades),
                wins=sum(1 for t in trades if t.pnl > 0),
                win_rate=round(
                    100.0 * sum(1 for t in trades if t.pnl > 0) / len(trades), 1
                ),
                total_pnl=round(sum(t.pnl for t in trades), 2),
            )
            for strategy, trades in buckets.items()
        ]
        return tuple(sorted(stats, key=lambda s: s.total_pnl, reverse=True))
