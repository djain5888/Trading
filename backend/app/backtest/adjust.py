"""Split/bonus back-adjustment for unadjusted provider candles.

Groww serves raw, unadjusted OHLCV, so a corporate action (a 5:1 split, a 1:1
bonus) appears as a huge overnight gap — an ~80% "crash" that is not a real
move. Left alone it destroys return continuity and produces impossible
backtest numbers.

This module detects suspected corporate actions (large overnight jumps, via
:func:`app.backtest.diagnostics.suspected_adjustments`), fits each to a clean
split/bonus factor, and back-adjusts every bar *before* the event so the series
is continuous in post-event terms. If a suspected jump cannot be explained by a
clean factor, the symbol is declared UNUSABLE and the caller must fail loudly
rather than silently trust unadjusted prices.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.backtest.diagnostics import DEFAULT_JUMP_THRESHOLD_PCT, suspected_adjustments
from app.market.historical.models import Candle

#: Relative tolerance when fitting an observed gap to a clean split/bonus factor.
DEFAULT_FACTOR_TOLERANCE = 0.03


def _candidate_factors() -> list[float]:
    """Return plausible corporate-action factors (prev_close / next_close).

    Splits give integer factors (2:1 -> 2, 5:1 -> 5); bonuses give ``(a+b)/b``
    for a small ``a:b`` ratio; reverse splits give the reciprocals. Restricting
    to these keeps an ordinary price move (which is not a clean ratio) from
    being mistaken for a corporate action.
    """
    factors: set[float] = set()
    for split in range(2, 21):  # 2:1 .. 20:1 splits and their reverses
        factors.add(float(split))
        factors.add(1.0 / split)
    for new in range(1, 7):  # bonus a:b, held b in 1..4, issued a in 1..6
        for held in range(1, 5):
            factors.add((new + held) / held)
            factors.add(held / (new + held))
    return sorted(factors)


_CANDIDATES = _candidate_factors()


def _fit_factor(raw_factor: float, tol: float) -> float | None:
    """Return the clean corporate-action factor nearest ``raw_factor``, or None.

    ``raw_factor`` is ``prev_close / next_close`` (>1 for a split/bonus drop,
    <1 for a reverse split). A match must be within ``tol`` (relative) of a
    plausible factor and not ~1 (which would be no action at all).
    """
    best: float | None = None
    best_diff = tol * raw_factor
    for candidate in _CANDIDATES:
        diff = abs(candidate - raw_factor)
        if diff <= best_diff and abs(candidate - 1.0) > tol:
            best, best_diff = candidate, diff
    return best


class SplitEvent(BaseModel):
    """One detected, cleanly-fitted corporate action."""

    model_config = ConfigDict(frozen=True)

    on: date = Field(description="Date of the first post-action candle.")
    factor: float = Field(gt=0, description="Clean split/bonus factor (prev/next).")
    change_pct: float = Field(description="Observed overnight change (%).")


class SplitAdjustment(BaseModel):
    """The result of back-adjusting one symbol's series."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Symbol adjusted.")
    usable: bool = Field(description="Whether every suspected jump was explained.")
    reason: str = Field(default="", description="Why the symbol is unusable, if so.")
    events: tuple[SplitEvent, ...] = Field(
        default=(), description="Corporate actions applied."
    )
    adjusted: tuple[Candle, ...] = Field(
        default=(), description="Back-adjusted candles (empty when unusable)."
    )

    @property
    def events_applied(self) -> int:
        """Return the number of adjustment events applied."""
        return len(self.events)


class SplitSummary(BaseModel):
    """A lightweight per-symbol adjustment summary (no candle payload)."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Symbol.")
    usable: bool = Field(description="Whether the series is usable after adjustment.")
    events_applied: int = Field(ge=0, description="Adjustment events applied.")
    reason: str = Field(default="", description="Why unusable, if applicable.")


def _scale(candle: Candle, multiplier: Decimal) -> Candle:
    """Return ``candle`` with prices scaled by ``multiplier`` (volume inverse)."""
    quant = Decimal("0.0001")
    inverse = Decimal(1) / multiplier
    return candle.model_copy(
        update={
            "open": (candle.open * multiplier).quantize(quant),
            "high": (candle.high * multiplier).quantize(quant),
            "low": (candle.low * multiplier).quantize(quant),
            "close": (candle.close * multiplier).quantize(quant),
            "volume": int(candle.volume * inverse),
        }
    )


def adjust_for_splits(
    symbol: str,
    candles: Sequence[Candle],
    *,
    threshold_pct: float = DEFAULT_JUMP_THRESHOLD_PCT,
    tol: float = DEFAULT_FACTOR_TOLERANCE,
) -> SplitAdjustment:
    """Back-adjust ``candles`` for suspected splits/bonuses.

    Returns a :class:`SplitAdjustment`. When a suspected jump cannot be fitted
    to a clean factor the series is marked ``usable=False`` with an explanatory
    ``reason`` and no adjusted candles — the caller must then fail loudly for
    this symbol rather than use the raw, discontinuous prices.
    """
    ordered = sorted(candles, key=lambda c: c.timestamp)
    if len(ordered) < 2:
        return SplitAdjustment(symbol=symbol, usable=True, adjusted=tuple(ordered))

    jumps = suspected_adjustments(ordered, threshold_pct)
    events: list[SplitEvent] = []
    unexplained: list[str] = []
    for jump in jumps:
        raw_factor = jump.prev_close / jump.close
        factor = _fit_factor(raw_factor, tol)
        if factor is None:
            unexplained.append(f"{jump.on} ({jump.change_pct:+.1f}%)")
            continue
        events.append(SplitEvent(on=jump.on, factor=factor, change_pct=jump.change_pct))

    if unexplained:
        reason = (
            f"{len(unexplained)} overnight jump(s) not explainable as a clean "
            f"split/bonus — cannot adjust from price alone: {', '.join(unexplained)}"
        )
        return SplitAdjustment(symbol=symbol, usable=False, reason=reason)

    adjusted = list(ordered)
    for event in sorted(events, key=lambda e: e.on):
        multiplier = Decimal(1) / Decimal(str(event.factor))
        adjusted = [
            _scale(candle, multiplier) if candle.timestamp.date() < event.on else candle
            for candle in adjusted
        ]

    return SplitAdjustment(
        symbol=symbol,
        usable=True,
        events=tuple(events),
        adjusted=tuple(adjusted),
    )
