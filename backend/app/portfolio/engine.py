"""The portfolio analysis engine: assemble the measurement report.

Pure and deterministic — same transactions, prices, NAVs and valuation date
always produce the same :class:`PortfolioReport`. No signals, no side effects:
this only measures an existing book (value, XIRR, allocation drift, LTCG
readiness, exit tax, broker cross-check).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal

from app.portfolio.models import (
    AssetType,
    Holding,
    PortfolioConfig,
    Sleeve,
    Transaction,
    TxnType,
)
from app.portfolio.nav import NavEntry, NavSnapshot
from app.portfolio.prices import HoldingMismatch, cross_check_holdings
from app.portfolio.report import (
    AssetClassReport,
    BandStatus,
    HoldingReport,
    PortfolioReport,
    SleeveReport,
)
from app.portfolio.tax import TaxConfig, build_lots, exit_tax, holding_tax
from app.portfolio.xirr import absolute_return_pct, cashflows_from, xirr

#: How many of the weakest holdings to name as return "drags".
_MAX_DRAGS = 3


def _to_pct(rate: float | None) -> float | None:
    """Convert a decimal XIRR rate to a percentage, preserving None."""
    return None if rate is None else round(rate * 100.0, 2)


def analyse_portfolio(
    config: PortfolioConfig,
    transactions: Sequence[Transaction],
    *,
    valuation_date: date,
    equity_prices: Mapping[str, Decimal],
    nav: NavSnapshot | None = None,
    tax_config: TaxConfig | None = None,
    broker_units: Mapping[str, Decimal] | None = None,
) -> PortfolioReport:
    """Measure the portfolio and return a :class:`PortfolioReport`."""
    taxes = tax_config or TaxConfig()
    grouped = _group(transactions)

    holding_reports: list[HoldingReport] = []
    unpriced: list[str] = []
    unmatched: list[str] = []
    imported_units: dict[str, Decimal] = {}

    for holding in config.holdings:
        txns = grouped.get(holding.key, [])
        price, matched_code = _price_for(holding, equity_prices, nav)
        report = _analyse_holding(
            holding, txns, valuation_date, price, matched_code, taxes
        )
        holding_reports.append(report)
        if not report.priced and report.units > 0:
            unpriced.append(holding.identifier)
            # An MF with a NAV file loaded but no resolvable code is a config
            # error the user must fix (BUG 1): flag it for the loud failure.
            if holding.asset_type is AssetType.MF and nav is not None:
                unmatched.append(holding.identifier)
        if holding.asset_type is AssetType.EQUITY and report.units > 0:
            imported_units[holding.identifier] = report.units

    total_value = sum((h.market_value for h in holding_reports), Decimal(0))
    mismatches = _cross_check(imported_units, broker_units)

    sleeves = _sleeve_reports(
        config, holding_reports, grouped, valuation_date, total_value
    )
    asset_classes = _asset_class_reports(holding_reports, grouped, valuation_date)
    portfolio_incomplete = _has_unpriced(holding_reports)
    portfolio_xirr = (
        None
        if portfolio_incomplete
        else _aggregate_xirr(transactions, total_value, valuation_date)
    )

    return _build_report(
        config=config,
        valuation_date=valuation_date,
        holding_reports=holding_reports,
        total_value=total_value,
        portfolio_xirr=portfolio_xirr,
        portfolio_incomplete=portfolio_incomplete,
        sleeves=sleeves,
        asset_classes=asset_classes,
        unpriced=unpriced,
        unmatched=unmatched,
        mismatches=mismatches,
    )


def _group(
    transactions: Sequence[Transaction],
) -> dict[tuple[AssetType, str], list[Transaction]]:
    """Group transactions by holding key."""
    grouped: dict[tuple[AssetType, str], list[Transaction]] = {}
    for txn in transactions:
        grouped.setdefault(txn.key, []).append(txn)
    return grouped


def _price_for(
    holding: Holding, equity_prices: Mapping[str, Decimal], nav: NavSnapshot | None
) -> tuple[Decimal | None, str | None]:
    """Return ``(price, matched_code)`` for a holding.

    Equities price by close (no code). MF holdings resolve their AMFI NAV via,
    in order: an explicit ``scheme_code`` override, the identifier treated as a
    code, then a fuzzy scheme-name match (BUG 1) — so ``portfolio.json`` may use
    plain scheme names. ``matched_code`` records which AMFI code was used.
    """
    if holding.asset_type is AssetType.EQUITY:
        price = equity_prices.get(holding.identifier.upper()) or equity_prices.get(
            holding.identifier
        )
        return price, None
    if nav is None:
        return None, None
    entry = _resolve_mf(holding, nav)
    return (entry.nav, entry.scheme_code) if entry is not None else (None, None)


def _resolve_mf(holding: Holding, nav: NavSnapshot) -> NavEntry | None:
    """Resolve an MF holding to an AMFI NAV entry (code override, code, or name)."""
    if holding.scheme_code:
        entry = nav.get(holding.scheme_code)
        if entry is not None:
            return entry
    direct = nav.get(holding.identifier)
    if direct is not None:
        return direct
    return nav.match_by_name(holding.name) or nav.match_by_name(holding.identifier)


def _has_unpriced(holding_reports: Sequence[HoldingReport]) -> bool:
    """Return whether any held (units > 0) holding lacks a valuation price."""
    return any(not r.priced and r.units > 0 for r in holding_reports)


def _analyse_holding(
    holding: Holding,
    txns: Sequence[Transaction],
    valuation_date: date,
    price: Decimal | None,
    matched_code: str | None,
    taxes: TaxConfig,
) -> HoldingReport:
    """Measure a single holding."""
    open_lots, realised = build_lots(txns, taxes)
    units = sum((lot.units for lot in open_lots), Decimal(0))
    cost_basis = sum((lot.units * lot.cost_per_unit for lot in open_lots), Decimal(0))
    priced = price is not None
    market_value = (price or Decimal(0)) * units
    invested = sum(
        (t.amount + t.charges for t in txns if t.txn_type is TxnType.BUY), Decimal(0)
    )
    realised_proceeds = sum((g.proceeds for g in realised), Decimal(0))
    realised_gain = sum((g.gain for g in realised), Decimal(0))
    dividends = sum(
        (t.amount - t.charges for t in txns if t.txn_type is TxnType.DIVIDEND),
        Decimal(0),
    )
    htax = holding_tax(open_lots, holding.asset_type, valuation_date, taxes)
    if price is not None:
        etax = exit_tax(
            open_lots, holding.asset_type, valuation_date, price, taxes
        ).total_tax
    else:
        etax = Decimal(0)
    # An unpriced holding that still holds units has no meaningful terminal
    # value, so its XIRR would be garbage — withhold it (BUG 2).
    if not priced and units > 0:
        xirr_rate = None
    else:
        xirr_rate = xirr(
            cashflows_from(
                txns, terminal_value=market_value, terminal_date=valuation_date
            )
        )
    abs_ret = absolute_return_pct(
        invested, market_value + realised_proceeds + dividends
    )

    ltcg_ready = htax.ltcg_units_now * (price or Decimal(0))
    ltcg_90d = (htax.ltcg_units_30d + htax.ltcg_units_60d + htax.ltcg_units_90d) * (
        price or Decimal(0)
    )

    return HoldingReport(
        identifier=holding.identifier,
        name=holding.name,
        asset_type=holding.asset_type,
        sleeve=holding.sleeve,
        units=units,
        price=price,
        market_value=market_value,
        cost_basis=cost_basis,
        invested=invested,
        realised_proceeds=realised_proceeds,
        realised_gain=realised_gain,
        dividends=dividends,
        unrealised_pnl=market_value - cost_basis,
        absolute_return_pct=round(abs_ret, 2),
        xirr_pct=_to_pct(xirr_rate),
        ltcg_ready_value=ltcg_ready,
        ltcg_value_90d=ltcg_90d,
        exit_tax=etax,
        priced=priced,
        matched_code=matched_code,
    )


def _aggregate_xirr(
    transactions: Sequence[Transaction], terminal_value: Decimal, on: date
) -> float | None:
    """Return the XIRR over a set of transactions plus a terminal value."""
    flows = cashflows_from(
        transactions, terminal_value=terminal_value, terminal_date=on
    )
    return xirr(flows)


def _sleeve_reports(
    config: PortfolioConfig,
    holding_reports: Sequence[HoldingReport],
    grouped: Mapping[tuple[AssetType, str], list[Transaction]],
    valuation_date: date,
    total_value: Decimal,
) -> tuple[SleeveReport, ...]:
    """Build per-sleeve allocation and XIRR reports."""
    reports: list[SleeveReport] = []
    for sleeve in Sleeve:
        members = [h for h in config.holdings if h.sleeve is sleeve]
        sleeve_reports = [r for r in holding_reports if r.sleeve is sleeve]
        value = sum((r.market_value for r in sleeve_reports), Decimal(0))
        txns: list[Transaction] = []
        for holding in members:
            txns.extend(grouped.get(holding.key, []))
        actual = float(value / total_value * 100) if total_value > 0 else 0.0
        incomplete = _has_unpriced(sleeve_reports)
        reports.append(
            SleeveReport(
                sleeve=sleeve,
                market_value=value,
                actual_pct=round(actual, 2),
                target_pct=float(config.targets.target_for(sleeve)),
                xirr_pct=(
                    None
                    if incomplete
                    else _to_pct(_aggregate_xirr(txns, value, valuation_date))
                ),
                xirr_incomplete=incomplete,
            )
        )
    return tuple(reports)


def _asset_class_reports(
    holding_reports: Sequence[HoldingReport],
    grouped: Mapping[tuple[AssetType, str], list[Transaction]],
    valuation_date: date,
) -> tuple[AssetClassReport, ...]:
    """Build per-asset-class roll-ups."""
    reports: list[AssetClassReport] = []
    for asset_type in AssetType:
        members = [r for r in holding_reports if r.asset_type is asset_type]
        value = sum((r.market_value for r in members), Decimal(0))
        txns = [
            txn for key, rows in grouped.items() if key[0] is asset_type for txn in rows
        ]
        if not txns:
            continue
        incomplete = _has_unpriced(members)
        reports.append(
            AssetClassReport(
                asset_type=asset_type,
                market_value=value,
                xirr_pct=(
                    None
                    if incomplete
                    else _to_pct(_aggregate_xirr(txns, value, valuation_date))
                ),
                xirr_incomplete=incomplete,
            )
        )
    return tuple(reports)


def _cross_check(
    imported_units: Mapping[str, Decimal],
    broker_units: Mapping[str, Decimal] | None,
) -> tuple[HoldingMismatch, ...]:
    """Run the broker holdings cross-check when broker data is available."""
    if broker_units is None:
        return ()
    return tuple(cross_check_holdings(imported_units, broker_units))


def _band_status(xirr_pct: float | None, low: float, high: float) -> BandStatus:
    """Classify a portfolio XIRR against the target band."""
    if xirr_pct is None or xirr_pct < low:
        return BandStatus.BELOW
    if xirr_pct > high:
        return BandStatus.ABOVE
    return BandStatus.WITHIN


def _drags(holding_reports: Sequence[HoldingReport]) -> tuple[str, ...]:
    """Name the weakest priced holdings dragging the return down."""
    priced = [h for h in holding_reports if h.priced and h.units > 0]
    ranked = sorted(
        priced, key=lambda h: (h.xirr_pct if h.xirr_pct is not None else 0.0)
    )
    return tuple(h.identifier for h in ranked[:_MAX_DRAGS])


def _build_report(
    *,
    config: PortfolioConfig,
    valuation_date: date,
    holding_reports: Sequence[HoldingReport],
    total_value: Decimal,
    portfolio_xirr: float | None,
    portfolio_incomplete: bool,
    sleeves: tuple[SleeveReport, ...],
    asset_classes: tuple[AssetClassReport, ...],
    unpriced: Sequence[str],
    unmatched: Sequence[str],
    mismatches: tuple[HoldingMismatch, ...],
) -> PortfolioReport:
    """Assemble the top-level portfolio report from the parts."""
    cost_basis = sum((h.cost_basis for h in holding_reports), Decimal(0))
    invested = sum((h.invested for h in holding_reports), Decimal(0))
    realised = sum((h.realised_proceeds for h in holding_reports), Decimal(0))
    dividends = sum((h.dividends for h in holding_reports), Decimal(0))
    ltcg_ready = sum((h.ltcg_ready_value for h in holding_reports), Decimal(0))
    exit_total = sum((h.exit_tax for h in holding_reports), Decimal(0))
    realised_gain = sum((h.realised_gain for h in holding_reports), Decimal(0))

    xirr_pct = _to_pct(portfolio_xirr)
    low = float(config.targets.xirr_low)
    high = float(config.targets.xirr_high)
    abs_ret = absolute_return_pct(invested, total_value + realised + dividends)

    largest = max(holding_reports, key=lambda h: h.market_value, default=None)
    concentration_id = largest.identifier if largest is not None else ""
    concentration_pct = (
        float(largest.market_value / total_value * 100)
        if largest is not None and total_value > 0
        else 0.0
    )

    return PortfolioReport(
        valuation_date=valuation_date,
        market_value=total_value,
        cost_basis=cost_basis,
        invested=invested,
        unrealised_pnl=total_value - cost_basis,
        realised_gain=realised_gain,
        absolute_return_pct=round(abs_ret, 2),
        xirr_pct=xirr_pct,
        xirr_incomplete=portfolio_incomplete,
        band_low=low,
        band_high=high,
        band_status=_band_status(xirr_pct, low, high),
        drags=_drags(holding_reports),
        concentration_id=concentration_id,
        concentration_pct=round(concentration_pct, 2),
        ltcg_ready_value=ltcg_ready,
        exit_tax=exit_total,
        sleeves=sleeves,
        asset_classes=asset_classes,
        holdings=tuple(holding_reports),
        unpriced=tuple(unpriced),
        unmatched_schemes=tuple(unmatched),
        mismatches=mismatches,
    )


def raise_for_unmatched(report: PortfolioReport) -> None:
    """Fail loudly if any MF scheme could not be matched to an AMFI code.

    An unmatched scheme means the portfolio config's name (or code) does not
    resolve against the NAV file, so those holdings cannot be valued. Rather
    than silently valuing them at zero, the caller stops and tells the user
    exactly which schemes to fix (add an explicit ``scheme_code`` or correct
    the name).

    Raises:
        ValueError: If ``report.unmatched_schemes`` is non-empty.
    """
    if report.unmatched_schemes:
        joined = ", ".join(report.unmatched_schemes)
        raise ValueError(
            "Could not match these mutual-fund holdings to an AMFI scheme code: "
            f"{joined}. Fix the scheme name in portfolio.json or add an explicit "
            '"scheme_code".'
        )
