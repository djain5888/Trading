"""Pure, vectorised indicator calculations over float sequences.

Each function returns a list aligned to the input length, with ``None`` at
indices where the indicator is not yet defined. These functions have no
knowledge of candles, caching or the engine — only arithmetic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Aligned = list[float | None]


def sma(values: Sequence[float], period: int) -> Aligned:
    """Rolling simple moving average."""
    out: Aligned = [None] * len(values)
    running = 0.0
    for i, value in enumerate(values):
        running += value
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: Sequence[float], period: int) -> Aligned:
    """Exponential moving average seeded with the initial SMA."""
    out: Aligned = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1)
    prev = sum(values[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = values[i] * alpha + prev * (1 - alpha)
        out[i] = prev
    return out


def rsi(values: Sequence[float], period: int) -> Aligned:
    """Wilder's Relative Strength Index."""
    n = len(values)
    out: Aligned = [None] * n
    if n < period + 1:
        return out
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        delta = values[i] - values[i - 1]
        gains[i] = max(delta, 0.0)
        losses[i] = max(-delta, 0.0)
    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    out[period] = _rsi(avg_gain, avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i] = _rsi(avg_gain, avg_loss)
    return out


def _rsi(avg_gain: float, avg_loss: float) -> float:
    """Convert average gain/loss into an RSI value."""
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def roc(values: Sequence[float], period: int) -> Aligned:
    """Rate of change (percentage)."""
    out: Aligned = [None] * len(values)
    for i in range(period, len(values)):
        base = values[i - period]
        if base != 0:
            out[i] = (values[i] / base - 1.0) * 100.0
    return out


def momentum(values: Sequence[float], period: int) -> Aligned:
    """Absolute momentum (difference over the period)."""
    out: Aligned = [None] * len(values)
    for i in range(period, len(values)):
        out[i] = values[i] - values[i - period]
    return out


def rolling_std(values: Sequence[float], period: int) -> Aligned:
    """Rolling population standard deviation."""
    out: Aligned = [None] * len(values)
    total = 0.0
    total_sq = 0.0
    for i, value in enumerate(values):
        total += value
        total_sq += value * value
        if i >= period:
            old = values[i - period]
            total -= old
            total_sq -= old * old
        if i >= period - 1:
            mean = total / period
            variance = max(total_sq / period - mean * mean, 0.0)
            out[i] = math.sqrt(variance)
    return out


def true_range(
    high: Sequence[float], low: Sequence[float], close: Sequence[float]
) -> list[float]:
    """True range for each candle."""
    n = len(close)
    out = [0.0] * n
    out[0] = high[0] - low[0]
    for i in range(1, n):
        out[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return out


def atr(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int,
) -> Aligned:
    """Wilder's Average True Range."""
    n = len(close)
    out: Aligned = [None] * n
    if n < period + 1:
        return out
    tr = true_range(high, low, close)
    prev = sum(tr[1 : period + 1]) / period
    out[period] = prev
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def bollinger(
    values: Sequence[float], period: int, multiplier: float
) -> tuple[Aligned, Aligned, Aligned]:
    """Bollinger Bands, returning ``(middle, upper, lower)``."""
    middle = sma(values, period)
    deviation = rolling_std(values, period)
    upper: Aligned = [None] * len(values)
    lower: Aligned = [None] * len(values)
    for i in range(len(values)):
        mid = middle[i]
        dev = deviation[i]
        if mid is not None and dev is not None:
            upper[i] = mid + multiplier * dev
            lower[i] = mid - multiplier * dev
    return middle, upper, lower


def macd(
    values: Sequence[float], fast: int, slow: int, signal: int
) -> tuple[Aligned, Aligned, Aligned]:
    """MACD, returning ``(macd_line, signal_line, histogram)``."""
    fast_ema = ema(values, fast)
    slow_ema = ema(values, slow)
    n = len(values)
    macd_line: Aligned = [None] * n
    for i in range(n):
        f = fast_ema[i]
        s = slow_ema[i]
        if f is not None and s is not None:
            macd_line[i] = f - s
    start = slow - 1
    defined = [v for v in macd_line[start:] if v is not None]
    signal_sub = ema(defined, signal)
    signal_line: Aligned = [None] * n
    histogram: Aligned = [None] * n
    for offset, sig in enumerate(signal_sub):
        if sig is not None:
            index = start + offset
            signal_line[index] = sig
            line = macd_line[index]
            if line is not None:
                histogram[index] = line - sig
    return macd_line, signal_line, histogram


def directional(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int,
) -> tuple[Aligned, Aligned, Aligned]:
    """Wilder DMI/ADX, returning ``(plus_di, minus_di, adx)``."""
    n = len(close)
    plus_di: Aligned = [None] * n
    minus_di: Aligned = [None] * n
    adx: Aligned = [None] * n
    if n < period + 1:
        return plus_di, minus_di, adx

    tr = true_range(high, low, close)
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    atr_s = sum(tr[1 : period + 1])
    plus_s = sum(plus_dm[1 : period + 1])
    minus_s = sum(minus_dm[1 : period + 1])
    dx: Aligned = [None] * n
    plus_di[period], minus_di[period], dx[period] = _dx(plus_s, minus_s, atr_s)
    for i in range(period + 1, n):
        atr_s = atr_s - atr_s / period + tr[i]
        plus_s = plus_s - plus_s / period + plus_dm[i]
        minus_s = minus_s - minus_s / period + minus_dm[i]
        plus_di[i], minus_di[i], dx[i] = _dx(plus_s, minus_s, atr_s)

    first = period
    if n >= first + period:
        window = [v for v in dx[first : first + period] if v is not None]
        prev = sum(window) / period
        adx[first + period - 1] = prev
        for i in range(first + period, n):
            value = dx[i]
            if value is not None:
                prev = (prev * (period - 1) + value) / period
                adx[i] = prev
    return plus_di, minus_di, adx


def _dx(plus_s: float, minus_s: float, atr_s: float) -> tuple[float, float, float]:
    """Return ``(+DI, -DI, DX)`` from smoothed sums."""
    if atr_s == 0:
        return 0.0, 0.0, 0.0
    plus = 100.0 * plus_s / atr_s
    minus = 100.0 * minus_s / atr_s
    total = plus + minus
    dx = 0.0 if total == 0 else 100.0 * abs(plus - minus) / total
    return plus, minus, dx


def vwap(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
) -> Aligned:
    """Cumulative volume-weighted average price."""
    out: Aligned = [None] * len(close)
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(len(close)):
        typical = (high[i] + low[i] + close[i]) / 3.0
        cum_pv += typical * volume[i]
        cum_v += volume[i]
        if cum_v != 0:
            out[i] = cum_pv / cum_v
    return out


def obv(close: Sequence[float], volume: Sequence[float]) -> Aligned:
    """On-balance volume."""
    n = len(close)
    out: Aligned = [None] * n
    running = 0.0
    out[0] = 0.0
    for i in range(1, n):
        if close[i] > close[i - 1]:
            running += volume[i]
        elif close[i] < close[i - 1]:
            running -= volume[i]
        out[i] = running
    return out


def typical_price(
    high: Sequence[float], low: Sequence[float], close: Sequence[float]
) -> Aligned:
    """Typical price ``(H + L + C) / 3``."""
    return [(high[i] + low[i] + close[i]) / 3.0 for i in range(len(close))]


def median_price(high: Sequence[float], low: Sequence[float]) -> Aligned:
    """Median price ``(H + L) / 2``."""
    return [(high[i] + low[i]) / 2.0 for i in range(len(high))]


def weighted_close(
    high: Sequence[float], low: Sequence[float], close: Sequence[float]
) -> Aligned:
    """Weighted close ``(H + L + 2C) / 4``."""
    return [(high[i] + low[i] + 2.0 * close[i]) / 4.0 for i in range(len(close))]


def is_finite(value: float) -> bool:
    """Return whether a value is finite (not NaN or infinite)."""
    return not (math.isnan(value) or math.isinf(value))
