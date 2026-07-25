"""XIRR (money-weighted, irregular-cashflow annualised return).

Deterministic Newton-Raphson with a bisection fallback so the same cashflows
always yield the same rate. Works on any dated signed-cashflow series — SIP
instalments, partial sells, dividends and charges are just more flows — plus a
terminal market value as the final positive flow.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from app.portfolio.models import Transaction

#: Days per year used to convert a date offset into a fractional year.
_DAYS_PER_YEAR = 365.0
#: Newton iteration controls.
_MAX_ITERATIONS = 200
_TOLERANCE = 1e-8
#: Bisection search bounds on the rate (>-100% .. +100000%).
_RATE_LOW = -0.999999
_RATE_HIGH = 1000.0


def _npv(rate: float, flows: Sequence[tuple[date, float]], origin: date) -> float:
    """Return the net present value of ``flows`` at ``rate`` (annual)."""
    total = 0.0
    for when, amount in flows:
        years = (when - origin).days / _DAYS_PER_YEAR
        total += amount / (1.0 + rate) ** years
    return total


def _npv_derivative(
    rate: float, flows: Sequence[tuple[date, float]], origin: date
) -> float:
    """Return d(NPV)/d(rate)."""
    total = 0.0
    for when, amount in flows:
        years = (when - origin).days / _DAYS_PER_YEAR
        if years == 0.0:
            continue
        total += -years * amount / (1.0 + rate) ** (years + 1.0)
    return total


def xirr(flows: Sequence[tuple[date, float]], *, guess: float = 0.1) -> float | None:
    """Return the annualised money-weighted rate, or ``None`` if undefined.

    ``flows`` are dated signed cashflows (outflows negative, inflows positive).
    A solution needs at least one of each sign; an all-one-sign series has no
    internal rate of return and yields ``None``. The result is a decimal rate
    (``0.17`` == 17%).
    """
    if len(flows) < 2:
        return None
    signs = {amount > 0 for _, amount in flows if amount != 0}
    if len(signs) < 2:
        return None  # need both an outflow and an inflow

    origin = min(when for when, _ in flows)

    rate = guess
    for _ in range(_MAX_ITERATIONS):
        value = _npv(rate, flows, origin)
        if abs(value) < _TOLERANCE:
            return rate
        derivative = _npv_derivative(rate, flows, origin)
        if derivative == 0.0:
            break
        step = value / derivative
        next_rate = rate - step
        if next_rate <= _RATE_LOW:
            break  # Newton walked off the domain — fall back to bisection
        if abs(step) < _TOLERANCE:
            return next_rate
        rate = next_rate

    return _bisect(flows, origin)


def _bisect(flows: Sequence[tuple[date, float]], origin: date) -> float | None:
    """Bracket the root by sign change, then bisect deterministically."""
    low, high = _RATE_LOW, _RATE_HIGH
    low_val = _npv(low, flows, origin)
    high_val = _npv(high, flows, origin)
    if low_val == 0.0:
        return low
    if high_val == 0.0:
        return high
    if (low_val > 0.0) == (high_val > 0.0):
        return None  # no sign change in the bracket — no locatable root
    for _ in range(_MAX_ITERATIONS):
        mid = (low + high) / 2.0
        mid_val = _npv(mid, flows, origin)
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
