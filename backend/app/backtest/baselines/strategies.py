"""The five baseline strategies.

Each is a thin decision layer over :class:`BaselineContext`; all fills, sizing
and costs come from the shared engine, so results are comparable. Parameters are
fixed defaults from :class:`BaselineConfig` and are never swept.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from app.backtest.baselines.base import (
    BaselineContext,
    BaselinePlan,
    EntryIntent,
    atr_expanding,
    closes,
    is_new_high,
    momentum_12_1,
    trailing_return,
)
from app.backtest.baselines.models import BaselineConfig, BaselineName


async def _index_bullish(
    ctx: BaselineContext, config: BaselineConfig, day: date
) -> bool | None:
    """Return whether the benchmark index is above its trend EMA, or ``None``."""
    candles = await ctx.history(config.index_symbol, day)
    ema = ctx.ema(candles, config.trend_ema)
    if not candles or ema is None:
        return None
    return float(candles[-1].close) > ema


async def _atr_stop(
    ctx: BaselineContext, config: BaselineConfig, symbol: str, day: date
) -> tuple[float, float] | None:
    """Return ``(signal_close, atr)`` for ``symbol`` at ``day``, or ``None``."""
    candles = await ctx.history(symbol, day)
    if not candles:
        return None
    atr = ctx.atr(candles, config.atr_period)
    if atr is None or atr <= 0:
        return None
    return float(candles[-1].close), atr


async def _rank_by(
    ctx: BaselineContext,
    config: BaselineConfig,
    universe: Sequence[str],
    day: date,
    score: str,
) -> list[str]:
    """Rank the universe by a return score, strongest first."""
    scored: list[tuple[float, str]] = []
    for symbol in universe:
        candles = await ctx.history(symbol, day)
        prices = closes(candles)
        value = (
            trailing_return(prices, config.rs_lookback)
            if score == "rs"
            else momentum_12_1(prices, config.momentum_lookback, config.momentum_skip)
        )
        if value is not None:
            scored.append((value, symbol))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [symbol for _, symbol in scored]


async def _top_entries(
    ctx: BaselineContext,
    config: BaselineConfig,
    universe: Sequence[str],
    day: date,
    held: frozenset[str],
    score: str,
    *,
    uses_stop: bool,
    trailing_distance: float | None,
    max_hold: int | None,
    trail_for_sizing: bool,
) -> list[EntryIntent]:
    """Build entry intents for the top-N ranked symbols not already held."""
    ranked = await _rank_by(ctx, config, universe, day, score)
    entries: list[EntryIntent] = []
    for symbol in ranked[: config.top_n]:
        if symbol in held:
            continue
        levels = await _atr_stop(ctx, config, symbol, day)
        if levels is None:
            continue
        signal_close, atr = levels
        trail = atr * config.trail_atr_mult if trailing_distance is not None else None
        stop_distance = (
            trail
            if trail_for_sizing and trail is not None
            else atr * config.stop_atr_mult
        )
        entries.append(
            EntryIntent(
                symbol=symbol,
                signal_price=signal_close,
                stop_distance=stop_distance,
                uses_stop=uses_stop,
                trailing_distance=trail,
                max_hold=max_hold,
            )
        )
    return entries


def _new_week(day: date, last: tuple[int, int] | None) -> bool:
    """Return whether ``day`` starts a new ISO week versus ``last``."""
    return (day.isocalendar().year, day.isocalendar().week) != last


class BuyAndHold:
    """Buy every symbol on day one and hold to the end (the benchmark floor)."""

    name = BaselineName.BUY_AND_HOLD.value

    def __init__(self, config: BaselineConfig) -> None:
        """Store fixed baseline parameters."""
        self._config = config

    async def plan(
        self,
        ctx: BaselineContext,
        day: date,
        universe: Sequence[str],
        held: frozenset[str],
    ) -> BaselinePlan:
        """Enter any not-yet-held symbol; positions never exit before the end."""
        entries: list[EntryIntent] = []
        for symbol in universe:
            if symbol in held:
                continue
            levels = await _atr_stop(ctx, self._config, symbol, day)
            if levels is None:
                continue
            signal_close, atr = levels
            entries.append(
                EntryIntent(
                    symbol=symbol,
                    signal_price=signal_close,
                    stop_distance=atr * self._config.stop_atr_mult,
                    uses_stop=False,
                    trailing_distance=None,
                    max_hold=None,
                )
            )
        return BaselinePlan(tuple(entries))


class TopRSWeekly:
    """Each week, buy the top-N by relative strength and hold a fixed period."""

    name = BaselineName.TOP_RS_WEEKLY.value

    def __init__(self, config: BaselineConfig) -> None:
        """Store fixed baseline parameters; per-run rebalance state resets here."""
        self._config = config
        self._last_week: tuple[int, int] | None = None

    async def plan(
        self,
        ctx: BaselineContext,
        day: date,
        universe: Sequence[str],
        held: frozenset[str],
    ) -> BaselinePlan:
        """Rebalance-in the top-N by RS on the first trading day of each week."""
        if not _new_week(day, self._last_week):
            return BaselinePlan()
        self._last_week = (day.isocalendar().year, day.isocalendar().week)
        entries = await _top_entries(
            ctx,
            self._config,
            universe,
            day,
            held,
            "rs",
            uses_stop=False,
            trailing_distance=None,
            max_hold=self._config.hold_days,
            trail_for_sizing=False,
        )
        return BaselinePlan(tuple(entries))


class TopRSTrailing:
    """Weekly top-N by RS, held with an ATR trailing stop and no fixed target."""

    name = BaselineName.TOP_RS_TRAILING.value

    def __init__(self, config: BaselineConfig) -> None:
        """Store fixed baseline parameters; per-run rebalance state resets here."""
        self._config = config
        self._last_week: tuple[int, int] | None = None

    async def plan(
        self,
        ctx: BaselineContext,
        day: date,
        universe: Sequence[str],
        held: frozenset[str],
    ) -> BaselinePlan:
        """Rebalance-in the top-N by RS; exits are left to the trailing stop."""
        if not _new_week(day, self._last_week):
            return BaselinePlan()
        self._last_week = (day.isocalendar().year, day.isocalendar().week)
        entries = await _top_entries(
            ctx,
            self._config,
            universe,
            day,
            held,
            "rs",
            uses_stop=True,
            trailing_distance=1.0,  # sentinel: a trail is computed per symbol
            max_hold=None,
            trail_for_sizing=True,
        )
        return BaselinePlan(tuple(entries))


class Momentum121:
    """Classic 12-1 momentum: hold the top-N, rebalanced monthly."""

    name = BaselineName.MOMENTUM_12_1.value

    def __init__(self, config: BaselineConfig) -> None:
        """Store fixed baseline parameters; per-run rebalance state resets here."""
        self._config = config
        self._last_month: tuple[int, int] | None = None

    async def plan(
        self,
        ctx: BaselineContext,
        day: date,
        universe: Sequence[str],
        held: frozenset[str],
    ) -> BaselinePlan:
        """Monthly: hold exactly the top-N; drop those that fell out of it."""
        month = (day.year, day.month)
        if month == self._last_month:
            return BaselinePlan()
        self._last_month = month
        ranked = await _rank_by(ctx, self._config, universe, day, "momentum")
        target = set(ranked[: self._config.top_n])
        entries = await _top_entries(
            ctx,
            self._config,
            universe,
            day,
            held,
            "momentum",
            uses_stop=False,
            trailing_distance=None,
            max_hold=None,
            trail_for_sizing=False,
        )
        exits = frozenset(symbol for symbol in held if symbol not in target)
        return BaselinePlan(tuple(entries), exits)


class MeanRevert:
    """Buy oversold (RSI<30) pullbacks in an uptrend; exit on RSI>50 or timeout."""

    name = BaselineName.MEAN_REVERT.value

    def __init__(self, config: BaselineConfig) -> None:
        """Store fixed baseline parameters."""
        self._config = config

    async def plan(
        self,
        ctx: BaselineContext,
        day: date,
        universe: Sequence[str],
        held: frozenset[str],
    ) -> BaselinePlan:
        """Enter oversold uptrends; flag held names whose RSI has recovered."""
        entries: list[EntryIntent] = []
        exits: set[str] = set()
        for symbol in universe:
            candles = await ctx.history(symbol, day)
            if not candles:
                continue
            rsi = ctx.rsi(candles, self._config.rsi_period)
            if symbol in held:
                if rsi is not None and rsi > self._config.rsi_exit:
                    exits.add(symbol)
                continue
            ema = ctx.ema(candles, self._config.trend_ema)
            close = float(candles[-1].close)
            atr = ctx.atr(candles, self._config.atr_period)
            if rsi is None or ema is None or atr is None or atr <= 0:
                continue
            if rsi < self._config.rsi_entry and close > ema:
                entries.append(
                    EntryIntent(
                        symbol=symbol,
                        signal_price=close,
                        stop_distance=atr * self._config.stop_atr_mult,
                        uses_stop=False,
                        trailing_distance=None,
                        max_hold=self._config.mean_revert_hold,
                    )
                )
        return BaselinePlan(tuple(entries), frozenset(exits))


# -- TASK-024 exploration candidates ---------------------------------------


class RSMomentumTrail(TopRSTrailing):
    """Top-N by RS, ATR trailing stop, weekly rebalance (a named alias)."""

    name = BaselineName.RS_MOMENTUM_TRAIL.value


class RSMomentumCash(TopRSTrailing):
    """RS momentum with a trailing stop that goes fully to cash in a bear index."""

    name = BaselineName.RS_MOMENTUM_CASH.value

    async def plan(
        self,
        ctx: BaselineContext,
        day: date,
        universe: Sequence[str],
        held: frozenset[str],
    ) -> BaselinePlan:
        """Exit everything (hold cash) when the index is below its trend EMA."""
        if await _index_bullish(ctx, self._config, day) is False:
            return BaselinePlan((), held)  # risk-off: close all, open nothing
        return await super().plan(ctx, day, universe, held)


class DualMomentum:
    """Top-N by RS, but only enter while the index is above its 200-EMA."""

    name = BaselineName.DUAL_MOMENTUM.value

    def __init__(self, config: BaselineConfig) -> None:
        """Store fixed baseline parameters; per-run rebalance state resets here."""
        self._config = config
        self._last_week: tuple[int, int] | None = None

    async def plan(
        self,
        ctx: BaselineContext,
        day: date,
        universe: Sequence[str],
        held: frozenset[str],
    ) -> BaselinePlan:
        """Weekly: buy the top-N by RS only when the index is in an uptrend."""
        if not _new_week(day, self._last_week):
            return BaselinePlan()
        self._last_week = (day.isocalendar().year, day.isocalendar().week)
        if await _index_bullish(ctx, self._config, day) is False:
            return BaselinePlan()  # absolute filter fails → hold cash for new capital
        entries = await _top_entries(
            ctx,
            self._config,
            universe,
            day,
            held,
            "rs",
            uses_stop=False,
            trailing_distance=None,
            max_hold=self._config.hold_days,
            trail_for_sizing=False,
        )
        return BaselinePlan(tuple(entries))


class TrendFollow:
    """Own each symbol while it trades above its 200-EMA; exit when it drops below."""

    name = BaselineName.TREND_FOLLOW.value

    def __init__(self, config: BaselineConfig) -> None:
        """Store fixed baseline parameters."""
        self._config = config

    async def plan(
        self,
        ctx: BaselineContext,
        day: date,
        universe: Sequence[str],
        held: frozenset[str],
    ) -> BaselinePlan:
        """Enter symbols above the trend EMA; flag held names that fell below it."""
        entries: list[EntryIntent] = []
        exits: set[str] = set()
        for symbol in universe:
            candles = await ctx.history(symbol, day)
            ema = ctx.ema(candles, self._config.trend_ema)
            if not candles or ema is None:
                continue
            close = float(candles[-1].close)
            if symbol in held:
                if close < ema:
                    exits.add(symbol)
                continue
            atr = ctx.atr(candles, self._config.atr_period)
            if close > ema and atr is not None and atr > 0:
                entries.append(
                    EntryIntent(
                        symbol=symbol,
                        signal_price=close,
                        stop_distance=atr * self._config.stop_atr_mult,
                        uses_stop=False,
                        trailing_distance=None,
                        max_hold=None,
                    )
                )
        return BaselinePlan(tuple(entries), frozenset(exits))


class VolatilityBreak:
    """Buy N-day-high breakouts confirmed by ATR expansion; ride an ATR trail."""

    name = BaselineName.VOLATILITY_BREAK.value

    def __init__(self, config: BaselineConfig) -> None:
        """Store fixed baseline parameters."""
        self._config = config

    async def plan(
        self,
        ctx: BaselineContext,
        day: date,
        universe: Sequence[str],
        held: frozenset[str],
    ) -> BaselinePlan:
        """Enter fresh breakouts on expanding volatility, held by a trailing stop."""
        entries: list[EntryIntent] = []
        for symbol in universe:
            if symbol in held:
                continue
            candles = await ctx.history(symbol, day)
            if not is_new_high(candles, self._config.vol_breakout_lookback):
                continue
            series = ctx.atr_series(candles, self._config.atr_period)
            if not atr_expanding(series, self._config.vol_expansion_lookback):
                continue
            atr = series[-1]
            if atr <= 0:
                continue
            trail = atr * self._config.trail_atr_mult
            entries.append(
                EntryIntent(
                    symbol=symbol,
                    signal_price=float(candles[-1].close),
                    stop_distance=trail,
                    uses_stop=True,
                    trailing_distance=trail,
                    max_hold=None,
                )
            )
        return BaselinePlan(tuple(entries))


class RSRotationMonthly:
    """Top-N by RS, rotated monthly (not weekly): rebalance to the leaders."""

    name = BaselineName.RS_ROTATION_MONTHLY.value

    def __init__(self, config: BaselineConfig) -> None:
        """Store fixed baseline parameters; per-run rebalance state resets here."""
        self._config = config
        self._last_month: tuple[int, int] | None = None

    async def plan(
        self,
        ctx: BaselineContext,
        day: date,
        universe: Sequence[str],
        held: frozenset[str],
    ) -> BaselinePlan:
        """Monthly: hold exactly the top-N by RS; drop names that fell out."""
        month = (day.year, day.month)
        if month == self._last_month:
            return BaselinePlan()
        self._last_month = month
        ranked = await _rank_by(ctx, self._config, universe, day, "rs")
        target = set(ranked[: self._config.top_n])
        entries = await _top_entries(
            ctx,
            self._config,
            universe,
            day,
            held,
            "rs",
            uses_stop=False,
            trailing_distance=None,
            max_hold=None,
            trail_for_sizing=False,
        )
        exits = frozenset(symbol for symbol in held if symbol not in target)
        return BaselinePlan(tuple(entries), exits)
