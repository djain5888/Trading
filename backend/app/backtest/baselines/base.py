"""The baseline abstraction: context, plan and shared indicator helpers.

A :class:`Baseline` decides, for one simulation day, which symbols to enter
(filled at the next open) and which open positions to close at today's close.
The :class:`BaselineEngine` supplies identical fills, sizing and costs to every
baseline, so their metrics are directly comparable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Protocol

from app.backtest import execution
from app.backtest.guard import LookaheadGuard
from app.backtest.models import BacktestConfig
from app.indicators.engine import IndicatorEngine
from app.indicators.exceptions import IndicatorError
from app.market.historical.models import Candle, SeriesKey


@dataclass(frozen=True)
class EntryIntent:
    """A desired entry, filled at the next day's open by the engine."""

    symbol: str
    signal_price: float  # the plan-day close (reference for slippage/leakage)
    stop_distance: float  # entry-side risk distance for sizing and R
    uses_stop: bool  # whether the stop is an actual exit trigger
    trailing_distance: float | None  # ATR trailing distance (None = fixed/no stop)
    max_hold: int | None  # holding cap in trading days (None = until stop/exit/end)


@dataclass(frozen=True)
class BaselinePlan:
    """A baseline's decision for one day: new entries and forced exits."""

    entries: tuple[EntryIntent, ...] = ()
    exits: frozenset[str] = field(default_factory=frozenset)


class BaselineContext:
    """Read-only market access for baselines, bounded by the look-ahead guard."""

    def __init__(
        self,
        data: LookaheadGuard,
        indicators: IndicatorEngine,
        config: BacktestConfig,
        history_days: int,
    ) -> None:
        """Initialise the context.

        Args:
            data: The look-ahead-guarded data view (frozen to the sim day).
            indicators: The shared indicator engine (pure on candles).
            config: Execution config carrying the exchange and interval.
            history_days: Calendar days of history each read spans.
        """
        self._data = data
        self._indicators = indicators
        self._exchange = config.exchange
        self._interval = config.interval
        self._history_days = history_days

    async def history(self, symbol: str, day: date) -> list[Candle]:
        """Return the candles for ``symbol`` up to and including ``day``."""
        key = SeriesKey(symbol=symbol, exchange=self._exchange, interval=self._interval)
        start = execution.eod(day) - timedelta(days=self._history_days)
        return await self._data.get_candles(key, start, execution.eod(day))

    def _last(
        self, candles: Sequence[Candle], indicator: str, **params: float
    ) -> float | None:
        """Return the latest value of ``indicator`` over ``candles``, or ``None``."""
        try:
            results = self._indicators.compute_from_candles(
                candles, indicator, **params
            )
        except IndicatorError:
            return None
        return results[-1].value if results else None

    def atr(self, candles: Sequence[Candle], period: int) -> float | None:
        """Return the latest ATR, or ``None`` on short history."""
        return self._last(candles, "atr", period=period)

    def rsi(self, candles: Sequence[Candle], period: int) -> float | None:
        """Return the latest RSI, or ``None`` on short history."""
        return self._last(candles, "rsi", period=period)

    def ema(self, candles: Sequence[Candle], period: int) -> float | None:
        """Return the latest EMA, or ``None`` on short history."""
        return self._last(candles, "ema", period=period)


class Baseline(Protocol):
    """A pluggable baseline strategy driven by the :class:`BaselineEngine`."""

    name: str

    async def plan(
        self,
        ctx: BaselineContext,
        day: date,
        universe: Sequence[str],
        held: frozenset[str],
    ) -> BaselinePlan:
        """Return the entries and forced exits for ``day`` (data <= day only)."""
        ...


# -- Shared computations ---------------------------------------------------


def closes(candles: Sequence[Candle]) -> list[float]:
    """Return the close prices of ``candles`` as floats."""
    return [float(candle.close) for candle in candles]


def trailing_return(prices: Sequence[float], lookback: int) -> float | None:
    """Return the simple return over ``lookback`` bars, or ``None`` if too short."""
    if len(prices) <= lookback:
        return None
    past = prices[-1 - lookback]
    return prices[-1] / past - 1.0 if past > 0 else None


def momentum_12_1(prices: Sequence[float], lookback: int, skip: int) -> float | None:
    """Return the 12-month-minus-1-month momentum, or ``None`` if too short."""
    if len(prices) <= lookback:
        return None
    recent = prices[-1 - skip]
    past = prices[-1 - lookback]
    return recent / past - 1.0 if past > 0 else None
