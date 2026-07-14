"""The built-in strategies: breakout, pullback and momentum.

Each names one setup from indicator/price structure (computed through the
:class:`IndicatorEngine`, never reimplemented) and returns a signal strength.
None of them emits a buy/sell decision.
"""

from __future__ import annotations

from typing import ClassVar

from app.strategy.base import Strategy, StrategyContext, StrategySignal, clamp


class BreakoutStrategy(Strategy):
    """Price breaking the recent high on a volume expansion."""

    name: ClassVar[str] = "breakout"
    reinforcing_scanners: ClassVar[frozenset[str]] = frozenset(
        {"breakout", "relative_volume", "gap_up"}
    )

    def evaluate(self, context: StrategyContext) -> StrategySignal | None:
        """Fire when the close clears the prior-high band with rising volume."""
        config = context.config
        highs = context.highs
        volumes = context.volumes
        if len(highs) < config.breakout_lookback + 1:
            return None
        prior_high = max(highs[-config.breakout_lookback - 1 : -1])
        close = context.last_close
        if prior_high <= 0 or close <= prior_high:
            return None
        average_volume = context.last("volume_sma", period=config.volume_period)
        if average_volume is None or average_volume <= 0:
            return None
        volume_ratio = volumes[-1] / average_volume
        if volume_ratio < config.breakout_volume_factor:
            return None
        margin = (close / prior_high - 1.0) * 100.0
        strength = clamp(
            55.0
            + margin * 5.0
            + (volume_ratio - config.breakout_volume_factor) * 15.0
            + self._scanner_bonus(context)
        )
        return StrategySignal(self.name, strength, close)


class PullbackStrategy(Strategy):
    """An uptrend that has retraced to a moving average and turned back up."""

    name: ClassVar[str] = "pullback"
    reinforcing_scanners: ClassVar[frozenset[str]] = frozenset({"rsi_momentum"})

    def evaluate(self, context: StrategyContext) -> StrategySignal | None:
        """Fire on a bullish reversal bar at the fast EMA within an uptrend."""
        config = context.config
        fast = context.last("ema", period=config.pullback_fast)
        slow = context.last("ema", period=config.pullback_slow)
        if fast is None or slow is None or fast <= slow:
            return None  # not an uptrend structure
        close = context.last_close
        open_ = context.opens[-1]
        low = context.lows[-1]
        if close < slow:
            return None  # broke the trend
        touched = low <= fast * (1.0 + config.pullback_touch_pct / 100.0)
        reversal = close > open_ and close >= fast
        if not (touched and reversal):
            return None
        depth = (fast - low) / fast * 100.0 if fast > 0 else 0.0
        strength = clamp(60.0 + depth * 5.0 + self._scanner_bonus(context))
        return StrategySignal(self.name, strength, close)


class MomentumStrategy(Strategy):
    """Strong trend continuation: EMA alignment + bullish MACD + RSI band."""

    name: ClassVar[str] = "momentum"
    reinforcing_scanners: ClassVar[frozenset[str]] = frozenset(
        {"ema_alignment", "macd_cross", "rsi_momentum"}
    )

    def evaluate(self, context: StrategyContext) -> StrategySignal | None:
        """Fire on sustained EMA alignment with bullish MACD and RSI in-band.

        Alignment must hold on the last two bars — a *continuation*, so a single
        breakout bar out of consolidation does not qualify.
        """
        config = context.config
        fast = context.values("ema", period=config.ema_fast)
        mid = context.values("ema", period=config.ema_mid)
        slow = context.values("ema", period=config.ema_slow)
        if len(fast) < 2 or len(mid) < 2 or len(slow) < 2:
            return None
        close = context.last_close
        aligned_now = close > fast[-1] > mid[-1] > slow[-1]
        aligned_prev = fast[-2] > mid[-2] > slow[-2]
        if not (aligned_now and aligned_prev):
            return None
        macd = context.macd(config.macd_fast, config.macd_slow, config.macd_signal)
        if macd is None or macd[0] <= macd[1]:
            return None
        rsi = context.last("rsi", period=config.rsi_period)
        if rsi is None or not (
            config.rsi_momentum_low <= rsi <= config.rsi_momentum_high
        ):
            return None
        histogram = macd[0] - macd[1]
        strength = clamp(60.0 + histogram * 5.0 + self._scanner_bonus(context))
        return StrategySignal(self.name, strength, close)
