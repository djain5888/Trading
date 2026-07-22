"""Return-integrity diagnostics: audit stored candles and the hold-return math.

Motivated by impossible benchmark numbers (buy_and_hold reporting +744% then
-83% on NSE large-caps in consecutive years). These are pure functions that
prove, per symbol per window and *independently of the trading engine*:

* whether candles are actually missing, duplicated, out of order or dated
  outside the requested window (wrong-year rows leaking in);
* whether prices look split/bonus adjusted (a large overnight jump on
  unadjusted data is a corporate action, not a real move);
* what the true, simple buy-and-hold return is — ``last_close / first_close``.

If the engine's benchmark return diverges from :func:`hold_return_pct`, the
benchmark math is the problem. If they agree but the number is still absurd,
the underlying prices are wrong (typically unadjusted splits).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from app.market.enums import Interval
from app.market.historical.models import Candle
from app.market.historical.validation import ValidationEngine

#: An overnight move at or beyond this magnitude (%) is treated as a suspected
#: corporate action (split/bonus) rather than a genuine price move.
DEFAULT_JUMP_THRESHOLD_PCT = 25.0


class PriceJump(BaseModel):
    """One suspected split/bonus: a large gap between two consecutive closes."""

    model_config = ConfigDict(frozen=True)

    on: date = Field(description="Date of the candle after the jump.")
    prev_close: float = Field(gt=0, description="Previous session's close.")
    close: float = Field(gt=0, description="This session's close.")
    change_pct: float = Field(description="Overnight close-to-close change (%).")


class CandleDateAudit(BaseModel):
    """A per-symbol, per-window audit of stored candles and the hold return."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Symbol audited.")
    window_start: date = Field(description="Requested window start (inclusive).")
    window_end: date = Field(description="Requested window end (inclusive).")
    candles: int = Field(ge=0, description="Stored candles in the window.")
    first_date: date | None = Field(default=None, description="Earliest candle date.")
    last_date: date | None = Field(default=None, description="Latest candle date.")
    first_close: float | None = Field(default=None, description="Earliest close.")
    last_close: float | None = Field(default=None, description="Latest close.")
    missing_sessions: int = Field(
        ge=0, description="Missing weekday sessions (weekends excluded)."
    )
    duplicate_dates: tuple[date, ...] = Field(
        default=(), description="Dates present more than once."
    )
    out_of_order: int = Field(
        ge=0, description="Adjacent pairs where the later row predates the earlier."
    )
    out_of_window_dates: tuple[date, ...] = Field(
        default=(), description="Candle dates outside the requested window."
    )
    suspected_adjustments: tuple[PriceJump, ...] = Field(
        default=(), description="Overnight jumps consistent with splits/bonuses."
    )

    @property
    def hold_return_pct(self) -> float | None:
        """Return the simple buy-and-hold return ``last/first - 1`` (%)."""
        if not self.first_close or not self.last_close:
            return None
        return round(100.0 * (self.last_close / self.first_close - 1.0), 2)

    @property
    def clean(self) -> bool:
        """Whether the window has no structural date defects."""
        return (
            not self.duplicate_dates
            and self.out_of_order == 0
            and not self.out_of_window_dates
        )


class ReturnCheck(BaseModel):
    """One window's audit paired with the engine's reported benchmark return."""

    model_config = ConfigDict(frozen=True)

    window_label: str = Field(description="Human-readable window label.")
    audit: CandleDateAudit = Field(description="The candle/date audit.")
    engine_return_pct: float | None = Field(
        default=None, description="buy_and_hold return the engine reported (%)."
    )

    @property
    def matches_engine(self) -> bool | None:
        """Whether the engine return tracks the simple hold return (±5%/±1pp)."""
        hold = self.audit.hold_return_pct
        if hold is None or self.engine_return_pct is None:
            return None
        return abs(self.engine_return_pct - hold) <= max(1.0, abs(hold) * 0.05)


def hold_return_pct(candles: Sequence[Candle]) -> float | None:
    """Return the simple buy-and-hold return over ``candles`` in percent.

    ``(last_close / first_close - 1) * 100`` on the chronologically first and
    last candles. ``None`` if there are fewer than two candles. This is the
    ground-truth benchmark return, computed with no sizing, leverage or costs.
    """
    if len(candles) < 2:
        return None
    ordered = sorted(candles, key=lambda c: c.timestamp)
    first = float(ordered[0].close)
    last = float(ordered[-1].close)
    if first <= 0:
        return None
    return round(100.0 * (last / first - 1.0), 2)


def suspected_adjustments(
    candles: Sequence[Candle], threshold_pct: float = DEFAULT_JUMP_THRESHOLD_PCT
) -> list[PriceJump]:
    """Flag overnight close-to-close jumps beyond ``threshold_pct`` (%).

    On unadjusted data a split or bonus appears as a large overnight move (a
    5:1 split reads as an ~80% crash). Genuine large-cap sessions rarely move
    this much, so these flags are strong evidence the feed is unadjusted.
    """
    ordered = sorted(candles, key=lambda c: c.timestamp)
    jumps: list[PriceJump] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        prev_close = float(previous.close)
        close = float(current.close)
        if prev_close <= 0:
            continue
        change = 100.0 * (close / prev_close - 1.0)
        if abs(change) >= threshold_pct:
            jumps.append(
                PriceJump(
                    on=current.timestamp.date(),
                    prev_close=round(prev_close, 4),
                    close=round(close, 4),
                    change_pct=round(change, 2),
                )
            )
    return jumps


def audit_candles(
    symbol: str,
    candles: Sequence[Candle],
    window_start: date,
    window_end: date,
    *,
    interval: Interval = Interval.ONE_DAY,
    jump_threshold_pct: float = DEFAULT_JUMP_THRESHOLD_PCT,
) -> CandleDateAudit:
    """Audit stored candles for ``symbol`` over ``[window_start, window_end]``.

    Reports missing sessions (weekends excluded), duplicate and out-of-order
    rows, candles dated outside the window (wrong-year leakage), and suspected
    split/bonus adjustments — plus the true simple hold return via the model's
    :attr:`~CandleDateAudit.hold_return_pct`.
    """
    ordered = sorted(candles, key=lambda c: c.timestamp)
    dates = [c.timestamp.date() for c in ordered]

    seen: set[date] = set()
    duplicates: list[date] = []
    for day in dates:
        if day in seen and day not in duplicates:
            duplicates.append(day)
        seen.add(day)

    # Out-of-order counts pairs in the ORIGINAL (as-stored) order, not sorted.
    original = [c.timestamp for c in candles]
    out_of_order = sum(
        1
        for earlier, later in zip(original, original[1:], strict=False)
        if later < earlier
    )

    out_of_window = tuple(
        day for day in dates if day < window_start or day > window_end
    )
    _, missing = ValidationEngine().detect_gaps(ordered, interval)

    return CandleDateAudit(
        symbol=symbol,
        window_start=window_start,
        window_end=window_end,
        candles=len(ordered),
        first_date=dates[0] if dates else None,
        last_date=dates[-1] if dates else None,
        first_close=float(ordered[0].close) if ordered else None,
        last_close=float(ordered[-1].close) if ordered else None,
        missing_sessions=missing,
        duplicate_dates=tuple(duplicates),
        out_of_order=out_of_order,
        out_of_window_dates=out_of_window,
        suspected_adjustments=tuple(suspected_adjustments(ordered, jump_threshold_pct)),
    )
