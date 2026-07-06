"""Built-in market scanners.

Each scanner is configurable (all thresholds are named parameters, no magic
numbers), returns a 0-100 score, and never emits a trading decision. A ``None``
return filters the symbol out.
"""

from __future__ import annotations

from typing import ClassVar

from app.scanner.base import Scanner, require_positive, require_positive_int
from app.scanner.exceptions import ScannerConfigError
from app.scanner.models import ScannerContext, ScannerResult


class EmaAlignmentScanner(Scanner):
    """Fires when EMAs are stacked bullishly (short > medium > long)."""

    name: ClassVar[str] = "ema_alignment"

    def __init__(
        self,
        short_period: int = 20,
        medium_period: int = 50,
        long_period: int = 200,
        spread_scale: float = 2000.0,
    ) -> None:
        """Configure the EMA periods and score scaling."""
        self._short = require_positive_int(short_period, "short_period")
        self._medium = require_positive_int(medium_period, "medium_period")
        self._long = require_positive_int(long_period, "long_period")
        self._scale = require_positive(spread_scale, "spread_scale")
        self._config = {
            "short_period": self._short,
            "medium_period": self._medium,
            "long_period": self._long,
            "spread_scale": self._scale,
        }

    def scan(self, context: ScannerContext) -> ScannerResult | None:
        """Return a result when the EMAs are bullishly aligned."""
        short = self._last_value(context, "ema", period=self._short)
        medium = self._last_value(context, "ema", period=self._medium)
        long = self._last_value(context, "ema", period=self._long)
        if short is None or medium is None or long is None:
            return None
        if not (short > medium > long):
            return None
        spread = (short - medium) / medium + (medium - long) / long
        score = spread * self._scale
        return self._result(
            context,
            score=score,
            confidence=score / 100.0,
            metadata={"ema_short": short, "ema_medium": medium, "ema_long": long},
        )


class BreakoutScanner(Scanner):
    """Fires when the close breaks above the prior N-candle high."""

    name: ClassVar[str] = "breakout"

    def __init__(self, lookback: int = 20, score_scale: float = 2000.0) -> None:
        """Configure the breakout lookback and score scaling."""
        self._lookback = require_positive_int(lookback, "lookback")
        self._scale = require_positive(score_scale, "score_scale")
        self._config = {"lookback": self._lookback, "score_scale": self._scale}

    def scan(self, context: ScannerContext) -> ScannerResult | None:
        """Return a result when the close exceeds the prior-high band."""
        highs = context.series.high
        close = context.series.close[-1]
        if len(highs) < self._lookback + 1:
            return None
        prior_high = max(highs[-self._lookback - 1 : -1])
        if close <= prior_high:
            return None
        strength = (close - prior_high) / prior_high
        return self._result(
            context,
            score=strength * self._scale,
            confidence=strength * self._scale / 100.0,
            metadata={"prior_high": prior_high, "close": close},
        )


class RelativeVolumeScanner(Scanner):
    """Fires when volume exceeds a multiple of its moving average."""

    name: ClassVar[str] = "relative_volume"

    def __init__(
        self, period: int = 20, min_ratio: float = 1.5, score_scale: float = 50.0
    ) -> None:
        """Configure the average window, ratio threshold and scaling."""
        self._period = require_positive_int(period, "period")
        self._min_ratio = require_positive(min_ratio, "min_ratio")
        self._scale = require_positive(score_scale, "score_scale")
        self._config = {
            "period": self._period,
            "min_ratio": self._min_ratio,
            "score_scale": self._scale,
        }

    def scan(self, context: ScannerContext) -> ScannerResult | None:
        """Return a result when volume exceeds its average by the threshold."""
        average = self._last_value(context, "volume_sma", period=self._period)
        if average is None or average <= 0:
            return None
        current = context.series.volume[-1]
        ratio = current / average
        if ratio < self._min_ratio:
            return None
        return self._result(
            context,
            score=(ratio - 1.0) * self._scale,
            confidence=(ratio - 1.0) * self._scale / 100.0,
            metadata={"ratio": ratio, "average_volume": average},
        )


class AtrExpansionScanner(Scanner):
    """Fires when ATR is expanding versus a lookback ago."""

    name: ClassVar[str] = "atr_expansion"

    def __init__(
        self, period: int = 14, lookback: int = 5, score_scale: float = 500.0
    ) -> None:
        """Configure the ATR period, comparison lookback and scaling."""
        self._period = require_positive_int(period, "period")
        self._lookback = require_positive_int(lookback, "lookback")
        self._scale = require_positive(score_scale, "score_scale")
        self._config = {
            "period": self._period,
            "lookback": self._lookback,
            "score_scale": self._scale,
        }

    def scan(self, context: ScannerContext) -> ScannerResult | None:
        """Return a result when ATR has risen versus the lookback."""
        results = context.safe_compute("atr", period=self._period)
        if len(results) < self._lookback + 1:
            return None
        current = results[-1].value
        previous = results[-1 - self._lookback].value
        if previous <= 0 or current <= previous:
            return None
        change = (current - previous) / previous
        return self._result(
            context,
            score=change * self._scale,
            confidence=change * self._scale / 100.0,
            metadata={"atr": current, "atr_prev": previous},
        )


class VwapStrengthScanner(Scanner):
    """Fires when price trades above VWAP."""

    name: ClassVar[str] = "vwap_strength"

    def __init__(self, score_scale: float = 2000.0) -> None:
        """Configure the score scaling."""
        self._scale = require_positive(score_scale, "score_scale")
        self._config = {"score_scale": self._scale}

    def scan(self, context: ScannerContext) -> ScannerResult | None:
        """Return a result when the close trades above VWAP."""
        vwap = self._last_value(context, "vwap")
        if vwap is None or vwap <= 0:
            return None
        close = context.series.close[-1]
        if close <= vwap:
            return None
        strength = (close - vwap) / vwap
        return self._result(
            context,
            score=strength * self._scale,
            confidence=strength * self._scale / 100.0,
            metadata={"vwap": vwap, "close": close},
        )


class RsiMomentumScanner(Scanner):
    """Fires when RSI sits within a configurable band."""

    name: ClassVar[str] = "rsi_momentum"

    def __init__(
        self, period: int = 14, lower: float = 50.0, upper: float = 70.0
    ) -> None:
        """Configure the RSI period and inclusive band."""
        self._period = require_positive_int(period, "period")
        if not 0 <= lower < upper <= 100:
            raise ScannerConfigError("Require 0 <= lower < upper <= 100.")
        self._lower = lower
        self._upper = upper
        self._config = {
            "period": self._period,
            "lower": self._lower,
            "upper": self._upper,
        }

    def scan(self, context: ScannerContext) -> ScannerResult | None:
        """Return a result when RSI falls inside the configured band."""
        rsi = self._last_value(context, "rsi", period=self._period)
        if rsi is None or not (self._lower <= rsi <= self._upper):
            return None
        position = (rsi - self._lower) / (self._upper - self._lower)
        score = position * 100.0
        return self._result(
            context, score=score, confidence=score / 100.0, metadata={"rsi": rsi}
        )


class MacdCrossScanner(Scanner):
    """Fires on a recent bullish or bearish MACD/signal cross."""

    name: ClassVar[str] = "macd_cross"

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        cross_lookback: int = 3,
        score_scale: float = 20.0,
    ) -> None:
        """Configure the MACD periods, cross window and scaling."""
        self._fast = require_positive_int(fast, "fast")
        self._slow = require_positive_int(slow, "slow")
        self._signal = require_positive_int(signal, "signal")
        self._cross_lookback = require_positive_int(cross_lookback, "cross_lookback")
        self._scale = require_positive(score_scale, "score_scale")
        self._config = {
            "fast": self._fast,
            "slow": self._slow,
            "signal": self._signal,
            "cross_lookback": self._cross_lookback,
            "score_scale": self._scale,
        }

    def scan(self, context: ScannerContext) -> ScannerResult | None:
        """Return a result when a MACD/signal cross occurred recently."""
        results = context.safe_compute(
            "macd", fast=self._fast, slow=self._slow, signal=self._signal
        )
        diffs = [
            r.value - r.metadata["signal"] for r in results if "signal" in r.metadata
        ]
        if len(diffs) < 2:
            return None
        window = diffs[-(self._cross_lookback + 1) :]
        direction = 0.0
        for previous, current in zip(window, window[1:], strict=False):
            if previous <= 0 < current:
                direction = 1.0
            elif previous >= 0 > current:
                direction = -1.0
        if direction == 0.0:
            return None
        magnitude = abs(diffs[-1])
        return self._result(
            context,
            score=magnitude * self._scale,
            confidence=magnitude * self._scale / 100.0,
            metadata={"direction": direction, "histogram": diffs[-1]},
        )


class BollingerExpansionScanner(Scanner):
    """Fires when Bollinger band width is expanding."""

    name: ClassVar[str] = "bollinger_expansion"

    def __init__(
        self,
        period: int = 20,
        multiplier: float = 2.0,
        lookback: int = 5,
        score_scale: float = 500.0,
    ) -> None:
        """Configure the band period, multiplier, lookback and scaling."""
        self._period = require_positive_int(period, "period")
        self._multiplier = require_positive(multiplier, "multiplier")
        self._lookback = require_positive_int(lookback, "lookback")
        self._scale = require_positive(score_scale, "score_scale")
        self._config = {
            "period": self._period,
            "multiplier": self._multiplier,
            "lookback": self._lookback,
            "score_scale": self._scale,
        }

    @staticmethod
    def _width(metadata: dict[str, float], middle: float) -> float:
        """Return the normalised band width for one Bollinger result."""
        if middle == 0:
            return 0.0
        return (metadata["upper"] - metadata["lower"]) / abs(middle)

    def scan(self, context: ScannerContext) -> ScannerResult | None:
        """Return a result when the band width has widened."""
        results = context.safe_compute(
            "bollinger", period=self._period, multiplier=self._multiplier
        )
        if len(results) < self._lookback + 1:
            return None
        current = self._width(results[-1].metadata, results[-1].value)
        previous = self._width(
            results[-1 - self._lookback].metadata, results[-1 - self._lookback].value
        )
        if previous <= 0 or current <= previous:
            return None
        change = (current - previous) / previous
        return self._result(
            context,
            score=change * self._scale,
            confidence=change * self._scale / 100.0,
            metadata={"width": current, "width_prev": previous},
        )


class GapUpScanner(Scanner):
    """Fires when the latest candle gaps up from the prior close."""

    name: ClassVar[str] = "gap_up"

    def __init__(self, min_gap: float = 0.02, score_scale: float = 1000.0) -> None:
        """Configure the minimum gap fraction and scaling."""
        self._min_gap = require_positive(min_gap, "min_gap")
        self._scale = require_positive(score_scale, "score_scale")
        self._config = {"min_gap": self._min_gap, "score_scale": self._scale}

    def scan(self, context: ScannerContext) -> ScannerResult | None:
        """Return a result when the latest open gaps up beyond the threshold."""
        closes = context.series.close
        opens = context.series.open
        if len(closes) < 2:
            return None
        prior_close = closes[-2]
        if prior_close <= 0:
            return None
        gap = (opens[-1] - prior_close) / prior_close
        if gap < self._min_gap:
            return None
        return self._result(
            context,
            score=gap * self._scale,
            confidence=gap * self._scale / 100.0,
            metadata={"gap": gap, "prior_close": prior_close, "open": opens[-1]},
        )


class GapDownScanner(Scanner):
    """Fires when the latest candle gaps down from the prior close."""

    name: ClassVar[str] = "gap_down"

    def __init__(self, min_gap: float = 0.02, score_scale: float = 1000.0) -> None:
        """Configure the minimum gap fraction and scaling."""
        self._min_gap = require_positive(min_gap, "min_gap")
        self._scale = require_positive(score_scale, "score_scale")
        self._config = {"min_gap": self._min_gap, "score_scale": self._scale}

    def scan(self, context: ScannerContext) -> ScannerResult | None:
        """Return a result when the latest open gaps down beyond the threshold."""
        closes = context.series.close
        opens = context.series.open
        if len(closes) < 2:
            return None
        prior_close = closes[-2]
        if prior_close <= 0:
            return None
        gap = (prior_close - opens[-1]) / prior_close
        if gap < self._min_gap:
            return None
        return self._result(
            context,
            score=gap * self._scale,
            confidence=gap * self._scale / 100.0,
            metadata={"gap": gap, "prior_close": prior_close, "open": opens[-1]},
        )
