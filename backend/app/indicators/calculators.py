"""Concrete indicator calculators.

Each calculator wraps a pure calculation, declares its lookback via
``_lookback``, and assembles :class:`IndicatorResult` objects. Multi-output
indicators expose their extra series through result metadata.
"""

from __future__ import annotations

from typing import ClassVar

from app.indicators import calculations
from app.indicators.base import IndicatorCalculator, positive_int
from app.indicators.models import IndicatorResult, PriceSeries

# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


class SmaCalculator(IndicatorCalculator):
    """Simple moving average of close."""

    name: ClassVar[str] = "sma"
    supports_incremental: ClassVar[bool] = True

    def __init__(self, period: int = 20) -> None:
        """Configure the calculator with its period."""
        self._period = positive_int(period, "period")
        self._lookback = self._period

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        values = calculations.sma(series.close, self._period)
        return self._results(series, values, params={"period": self._period})


class EmaCalculator(IndicatorCalculator):
    """Exponential moving average of close."""

    name: ClassVar[str] = "ema"

    def __init__(self, period: int = 20) -> None:
        """Configure the calculator with its period."""
        self._period = positive_int(period, "period")
        self._lookback = self._period

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        values = calculations.ema(series.close, self._period)
        return self._results(series, values, params={"period": self._period})


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


class RsiCalculator(IndicatorCalculator):
    """Wilder's Relative Strength Index."""

    name: ClassVar[str] = "rsi"

    def __init__(self, period: int = 14) -> None:
        """Configure the calculator with its period."""
        self._period = positive_int(period, "period")
        self._lookback = self._period + 1

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        values = calculations.rsi(series.close, self._period)
        return self._results(series, values, params={"period": self._period})


class RocCalculator(IndicatorCalculator):
    """Rate of change (percentage)."""

    name: ClassVar[str] = "roc"
    supports_incremental: ClassVar[bool] = True

    def __init__(self, period: int = 12) -> None:
        """Configure the calculator with its period."""
        self._period = positive_int(period, "period")
        self._lookback = self._period + 1

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        values = calculations.roc(series.close, self._period)
        return self._results(series, values, params={"period": self._period})


class MomentumCalculator(IndicatorCalculator):
    """Absolute momentum."""

    name: ClassVar[str] = "momentum"
    supports_incremental: ClassVar[bool] = True

    def __init__(self, period: int = 10) -> None:
        """Configure the calculator with its period."""
        self._period = positive_int(period, "period")
        self._lookback = self._period + 1

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        values = calculations.momentum(series.close, self._period)
        return self._results(series, values, params={"period": self._period})


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


class MacdCalculator(IndicatorCalculator):
    """Moving Average Convergence Divergence."""

    name: ClassVar[str] = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        """Configure the fast, slow and signal periods."""
        self._fast = positive_int(fast, "fast")
        self._slow = positive_int(slow, "slow")
        self._signal = positive_int(signal, "signal")
        self._lookback = self._slow

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        macd_line, signal_line, histogram = calculations.macd(
            series.close, self._fast, self._slow, self._signal
        )
        return self._results(
            series,
            macd_line,
            metadata={"signal": signal_line, "histogram": histogram},
            params={"fast": self._fast, "slow": self._slow, "signal": self._signal},
        )


class DmiCalculator(IndicatorCalculator):
    """Directional Movement Index (+DI primary, -DI in metadata)."""

    name: ClassVar[str] = "dmi"

    def __init__(self, period: int = 14) -> None:
        """Configure the calculator with its period."""
        self._period = positive_int(period, "period")
        self._lookback = self._period + 1

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        plus_di, minus_di, _ = calculations.directional(
            series.high, series.low, series.close, self._period
        )
        return self._results(
            series,
            plus_di,
            metadata={"minus_di": minus_di},
            params={"period": self._period},
        )


class AdxCalculator(IndicatorCalculator):
    """Average Directional Index (with +DI/-DI in metadata)."""

    name: ClassVar[str] = "adx"

    def __init__(self, period: int = 14) -> None:
        """Configure the calculator with its period."""
        self._period = positive_int(period, "period")
        self._lookback = 2 * self._period

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        plus_di, minus_di, adx = calculations.directional(
            series.high, series.low, series.close, self._period
        )
        return self._results(
            series,
            adx,
            metadata={"plus_di": plus_di, "minus_di": minus_di},
            params={"period": self._period},
        )


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


class AtrCalculator(IndicatorCalculator):
    """Wilder's Average True Range."""

    name: ClassVar[str] = "atr"

    def __init__(self, period: int = 14) -> None:
        """Configure the calculator with its period."""
        self._period = positive_int(period, "period")
        self._lookback = self._period + 1

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        values = calculations.atr(series.high, series.low, series.close, self._period)
        return self._results(series, values, params={"period": self._period})


class BollingerCalculator(IndicatorCalculator):
    """Bollinger Bands (middle primary, upper/lower in metadata)."""

    name: ClassVar[str] = "bollinger"
    supports_incremental: ClassVar[bool] = True

    def __init__(self, period: int = 20, multiplier: float = 2.0) -> None:
        """Configure the calculator with its period."""
        self._period = positive_int(period, "period")
        self._multiplier = float(multiplier)
        self._lookback = self._period

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        middle, upper, lower = calculations.bollinger(
            series.close, self._period, self._multiplier
        )
        return self._results(
            series,
            middle,
            metadata={"upper": upper, "lower": lower},
            params={"period": self._period, "multiplier": self._multiplier},
        )


class StdDevCalculator(IndicatorCalculator):
    """Rolling population standard deviation of close."""

    name: ClassVar[str] = "stddev"
    supports_incremental: ClassVar[bool] = True

    def __init__(self, period: int = 20) -> None:
        """Configure the calculator with its period."""
        self._period = positive_int(period, "period")
        self._lookback = self._period

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        values = calculations.rolling_std(series.close, self._period)
        return self._results(series, values, params={"period": self._period})


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


class VwapCalculator(IndicatorCalculator):
    """Cumulative volume-weighted average price."""

    name: ClassVar[str] = "vwap"

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        values = calculations.vwap(series.high, series.low, series.close, series.volume)
        return self._results(series, values)


class ObvCalculator(IndicatorCalculator):
    """On-balance volume."""

    name: ClassVar[str] = "obv"

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        values = calculations.obv(series.close, series.volume)
        return self._results(series, values)


class VolumeSmaCalculator(IndicatorCalculator):
    """Simple moving average of volume."""

    name: ClassVar[str] = "volume_sma"
    supports_incremental: ClassVar[bool] = True

    def __init__(self, period: int = 20) -> None:
        """Configure the calculator with its period."""
        self._period = positive_int(period, "period")
        self._lookback = self._period

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        values = calculations.sma(series.volume, self._period)
        return self._results(series, values, params={"period": self._period})


# ---------------------------------------------------------------------------
# Price transforms
# ---------------------------------------------------------------------------


class TypicalPriceCalculator(IndicatorCalculator):
    """Typical price ``(H + L + C) / 3``."""

    name: ClassVar[str] = "typical_price"
    supports_incremental: ClassVar[bool] = True

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        values = calculations.typical_price(series.high, series.low, series.close)
        return self._results(series, values)


class MedianPriceCalculator(IndicatorCalculator):
    """Median price ``(H + L) / 2``."""

    name: ClassVar[str] = "median_price"
    supports_incremental: ClassVar[bool] = True

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        values = calculations.median_price(series.high, series.low)
        return self._results(series, values)


class WeightedCloseCalculator(IndicatorCalculator):
    """Weighted close ``(H + L + 2C) / 4``."""

    name: ClassVar[str] = "weighted_close"
    supports_incremental: ClassVar[bool] = True

    def _compute(self, series: PriceSeries) -> list[IndicatorResult]:
        values = calculations.weighted_close(series.high, series.low, series.close)
        return self._results(series, values)
