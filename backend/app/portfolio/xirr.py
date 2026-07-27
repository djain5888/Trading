"""XIRR (money-weighted, irregular-cashflow annualised return).

Deterministic Newton-Raphson with a bisection fallback so the same cashflows
always yield the same rate. Works on any dated signed-cashflow series — SIP
instalments, partial sells, dividends and charges are just more flows — plus a
terminal market value as the final positive flow.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from app.portfolio.models import Transaction

#: Days per year used to convert a date offset into a fractional year.
_DAYS_PER_YEAR = 365.0
#: Newton iteration controls.
_MAX_ITERATIONS = 200
_TOLERANCE = 1e-8
#: Valid domain for the rate. ``(1 + rate)`` must stay positive for the
#: fractional powers, so the rate is clamped just inside ``(-1, +inf)`` — a
#: -99.99% to +10000% annualised band that covers any real portfolio.
_RATE_LOW = -0.9999
_RATE_HIGH = 100.0
#: Initial bisection bracket; the upper bound expands if no sign change is found.
_BRACKET_LOW = -0.9999
_BRACKET_HIGH = 10.0


def _clamp(rate: float) -> float:
    """Clamp a rate into the valid ``(-0.9999, 100.0)`` domain."""
    return min(_RATE_HIGH, max(_RATE_LOW, rate))


def _npv(rate: float, flows: Sequence[tuple[date, float]], origin: date) -> float:
    """Return the NPV of ``flows`` at ``rate``, or ``nan`` on overflow.

    The discount base ``1 + rate`` is guaranteed positive (callers clamp the
    rate), but large exponents can still overflow; that is caught and surfaced
    as ``nan`` so the caller treats it as a failed evaluation rather than
    crashing (the bug this fix addresses).
    """
    base = 1.0 + rate
    if base <= 0.0:
        return math.nan
    total = 0.0
    try:
        for when, amount in flows:
            years = (when - origin).days / _DAYS_PER_YEAR
            total += amount / base**years
    except (OverflowError, ZeroDivisionError):
        return math.nan
    return total if math.isfinite(total) else math.nan


def _npv_derivative(
    rate: float, flows: Sequence[tuple[date, float]], origin: date
) -> float:
    """Return d(NPV)/d(rate), or ``nan`` on overflow."""
    base = 1.0 + rate
    if base <= 0.0:
        return math.nan
    total = 0.0
    try:
        for when, amount in flows:
            years = (when - origin).days / _DAYS_PER_YEAR
            if years == 0.0:
                continue
            total += -years * amount / base ** (years + 1.0)
    except (OverflowError, ZeroDivisionError):
        return math.nan
    return total if math.isfinite(total) else math.nan


def xirr(flows: Sequence[tuple[date, float]], *, guess: float = 0.1) -> float | None:
    """Return the annualised money-weighted rate, or ``None`` if undefined.

    ``flows`` are dated signed cashflows (outflows negative, inflows positive).
    A solution needs at least one of each sign; an all-one-sign series has no
    internal rate of return and yields ``None``. The result is a decimal rate
    (``0.17`` == 17%).

    The solver is numerically hardened: the rate is clamped to a valid domain
    every step, non-finite evaluations abandon the Newton step, and any
    divergence, overflow or exhaustion falls back to a bracketed bisection.
    """
    if len(flows) < 2:
        return None
    signs = {amount > 0 for _, amount in flows if amount != 0}
    if len(signs) < 2:
        return None  # need both an outflow and an inflow — otherwise undefined

    # Normalise the origin to the FIRST cashflow date so year offsets (and thus
    # the exponents) stay as small as possible.
    origin = min(when for when, _ in flows)

    newton = _newton(flows, origin, guess)
    if newton is not None:
        return newton
    return _bisect(flows, origin)


def _newton(
    flows: Sequence[tuple[date, float]], origin: date, guess: float
) -> float | None:
    """Run clamped Newton-Raphson; return a root or ``None`` to fall back."""
    rate = _clamp(guess)
    for _ in range(_MAX_ITERATIONS):
        value = _npv(rate, flows, origin)
        if not math.isfinite(value):
            return None  # overflow / non-finite — hand off to bisection
        if abs(value) < _TOLERANCE:
            return rate
        derivative = _npv_derivative(rate, flows, origin)
        if not math.isfinite(derivative) or derivative == 0.0:
            return None
        step = value / derivative
        if not math.isfinite(step):
            return None
        next_rate = _clamp(rate - step)
        if abs(next_rate - rate) < _TOLERANCE:
            return next_rate
        if next_rate == rate:
            return None  # clamped hard against a bound — bisection is safer
        rate = next_rate
    return None  # did not converge within the iteration budget


def _bisect(flows: Sequence[tuple[date, float]], origin: date) -> float | None:
    """Bracket the root by a sign change, expanding the bracket, then bisect."""
    low, high = _BRACKET_LOW, _BRACKET_HIGH
    low_val = _npv(low, flows, origin)

    high_val = _npv(high, flows, origin)
    # Expand the upper bound until the endpoints straddle a root (or we give up).
    while (
        math.isfinite(low_val)
        and math.isfinite(high_val)
        and (low_val > 0.0) == (high_val > 0.0)
        and high < _RATE_HIGH
    ):
        high = min(_RATE_HIGH, high * 2.0 if high > 0 else 1.0)
        high_val = _npv(high, flows, origin)

    if not math.isfinite(low_val) or not math.isfinite(high_val):
        return None
    if low_val == 0.0:
        return low
    if high_val == 0.0:
        return high
    if (low_val > 0.0) == (high_val > 0.0):
        return None  # no sign change across the bracket — XIRR is undefined

    for _ in range(_MAX_ITERATIONS):
        mid = (low + high) / 2.0
        mid_val = _npv(mid, flows, origin)
        if not math.isfinite(mid_val):
            return None
        if abs(mid_val) < _TOLERANCE or (high - low) / 2.0 < _TOLERANCE:
            return mid
        if (mid_val > 0.0) == (low_val > 0.0):
            low, low_val = mid, mid_val
        else:
            high = mid
    return (low + high) / 2.0


def cashflows_from(
    transactions: Sequence[Transaction],
    *,
    terminal_value: Decimal,
    terminal_date: date,
) -> list[tuple[date, float]]:
    """Build a dated signed-cashflow series from transactions plus a terminal MV.

    The terminal market value is appended as a final positive inflow (as if the
    holding were liquidated at valuation), which is what makes the XIRR reflect
    unrealised gains too.
    """
    flows: list[tuple[date, float]] = [
        (txn.date, float(txn.cashflow())) for txn in transactions
    ]
    if terminal_value != 0:
        flows.append((terminal_date, float(terminal_value)))
    return flows


def absolute_return_pct(invested: Decimal, current_plus_realised: Decimal) -> float:
    """Return the simple absolute return (%) on capital deployed.

    ``invested`` is the gross cost of all buys (including charges); the second
    argument is current market value plus realised proceeds and dividends.
    """
    if invested <= 0:
        return 0.0
    return float((current_plus_realised - invested) / invested * Decimal(100))
