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

from app.core.logging import get_logger
from app.portfolio.models import (
    AssetType,
    Holding,
    PortfolioConfig,
    Sleeve,
    Transaction,
    TxnType,
)
from app.portfolio.nav import (
    DEFAULT_NAME_MATCH_THRESHOLD,
    NavEntry,
    NavSnapshot,
    name_jaccard,
)
from app.portfolio.prices import HoldingMismatch, cross_check_holdings
from app.portfolio.report import (
    AssetClassReport,
    BandStatus,
    ClosedPosition,
    HoldingReport,
    PortfolioReport,
    SleeveReport,
)
from app.portfolio.tax import TaxConfig, build_lots, exit_tax, holding_tax
from app.portfolio.xirr import absolute_return_pct, cashflows_from, xirr

logger = get_logger(__name__)

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
    per_holding, unmatched_txns = _assign_transactions(config, transactions)

    holding_reports: list[HoldingReport] = []
    unpriced: list[str] = []
    unmatched: list[str] = []
    imported_units: dict[str, Decimal] = {}

    for holding in config.holdings:
        txns = per_holding[holding.key]
        price, matched_code = _price_for(holding, equity_prices, nav)
        report = _analyse_holding(
            holding, txns, valuation_date, price, matched_code, taxes
        )
        holding_reports.append(report)
        logger.info(
            "Holding %s (%s): %d transaction(s) joined, %s units.",
            holding.identifier,
            holding.asset_type.value,
            len(txns),
            report.units,
        )
        if not report.priced and report.units > 0:
            unpriced.append(holding.identifier)
            # An MF with a NAV file loaded but no resolvable code is a config
            # error the user must fix (BUG 1): flag it for the loud failure.
            if holding.asset_type is AssetType.MF and nav is not None:
                unmatched.append(holding.identifier)
        if holding.asset_type is AssetType.EQUITY and report.units > 0:
            imported_units[holding.identifier] = report.units

    # Join-integrity checks (BUG A). A config holding that matched no
    # transactions is a genuine error (you can't hold something you never
    # bought).
    zero_txn = [
        h.identifier
        for h in config.holdings
        if h.asset_type is AssetType.MF and not per_holding[h.key]
    ]
    for identifier in zero_txn:
        logger.error("MF holding %s matched ZERO transactions.", identifier)

    # Orphan transactions (no config holding): a net-zero group is a legitimately
    # CLOSED position; a non-zero group means the user holds something not in
    # portfolio.json — a genuine error.
    closed_positions, closed_txns, held_unconfigured = _classify_orphans(
        unmatched_txns, valuation_date
    )

    total_value = sum((h.market_value for h in holding_reports), Decimal(0))
    mismatches = _cross_check(imported_units, broker_units)

    sleeves = _sleeve_reports(
        config, holding_reports, per_holding, valuation_date, total_value
    )
    asset_classes = _asset_class_reports(holding_reports, per_holding, valuation_date)
    portfolio_incomplete = _aggregate_blocked(holding_reports)
    # Portfolio XIRR spans every real cashflow: held holdings AND closed
    # positions (their money-in/out is real), valued to the current book value.
    matched = [txn for txns in per_holding.values() for txn in txns]
    portfolio_xirr = (
        None
        if portfolio_incomplete
        else _aggregate_xirr(matched + closed_txns, total_value, valuation_date)
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
        zero_txn=zero_txn,
        closed_positions=closed_positions,
        held_unconfigured=held_unconfigured,
        mismatches=mismatches,
    )


def _classify_orphans(
    orphans: Sequence[Transaction], valuation_date: date
) -> tuple[tuple[ClosedPosition, ...], list[Transaction], list[str]]:
    """Split orphan transactions into closed positions vs held-but-unconfigured.

    Orphans are grouped by their own key. A group whose net units are zero is a
    legitimately closed position — reported with its realised P&L and XIRR, and
    its cashflows still count toward the portfolio return. A group with a
    non-zero (or negative/over-sold) net position means the user holds something
    absent from ``portfolio.json`` — a genuine error the caller fails on.
    """
    groups: dict[tuple[AssetType, str], list[Transaction]] = {}
    for txn in orphans:
        groups.setdefault(txn.key, []).append(txn)

    closed: list[ClosedPosition] = []
    closed_txns: list[Transaction] = []
    held: list[str] = []
    for (asset_type, identifier), txns in groups.items():
        bought = sum((t.units for t in txns if t.txn_type is TxnType.BUY), Decimal(0))
        sold = sum((t.units for t in txns if t.txn_type is TxnType.SELL), Decimal(0))
        label = next((t.name for t in txns if t.name), identifier)
        if bought == sold:  # fully exited (net zero)
            realised = sum((t.cashflow() for t in txns), Decimal(0))
            rate = xirr(
                cashflows_from(
                    txns, terminal_value=Decimal(0), terminal_date=valuation_date
                )
            )
            closed.append(
                ClosedPosition(
                    identifier=label,
                    asset_type=asset_type,
                    realised_gain=realised,
                    xirr_pct=_to_pct(rate),
                    trades=len(txns),
                )
            )
            closed_txns.extend(txns)
        else:  # still holding something not in the config
            held.append(label)
    return tuple(closed), closed_txns, sorted(held)


def _assign_transactions(
    config: PortfolioConfig, transactions: Sequence[Transaction]
) -> tuple[dict[tuple[AssetType, str], list[Transaction]], list[Transaction]]:
    """Join transactions to holdings; return per-holding lists and the orphans.

    Equities join on the ticker; MF transactions join on the *normalised* scheme
    name (the same normalisation used for AMFI matching), with an explicit
    ``scheme_code`` override taking precedence — because the CAS scheme-code
    column is often blank and the names differ in punctuation/suffixes between
    the statement and ``portfolio.json`` (BUG A).
    """
    per_holding: dict[tuple[AssetType, str], list[Transaction]] = {
        h.key: [] for h in config.holdings
    }
    equity_by_id = {
        h.identifier.upper(): h
        for h in config.holdings
        if h.asset_type is AssetType.EQUITY
    }
    mf_holdings = [h for h in config.holdings if h.asset_type is AssetType.MF]
    orphans: list[Transaction] = []
    for txn in transactions:
        if txn.asset_type is AssetType.EQUITY:
            holding = equity_by_id.get(txn.identifier.strip().upper())
        else:
            holding = _match_mf_holding(txn, mf_holdings)
        if holding is None:
            orphans.append(txn)
        else:
            per_holding[holding.key].append(txn)
    return per_holding, orphans


def _match_mf_holding(
    txn: Transaction, mf_holdings: Sequence[Holding]
) -> Holding | None:
    """Match one MF transaction to a holding by scheme_code or normalised name."""
    for holding in mf_holdings:
        if (
            holding.scheme_code
            and txn.identifier.strip() == holding.scheme_code.strip()
        ):
            return holding
    txn_names = [name for name in (txn.identifier, txn.name) if name]
    best: Holding | None = None
    best_score = 0.0
    for holding in mf_holdings:
        candidates = [holding.identifier, holding.name]
        if holding.scheme_code:
            candidates.append(holding.scheme_code)
        score = max(
            name_jaccard(txn_name, candidate)
            for txn_name in txn_names
            for candidate in candidates
        )
        if score > best_score + 1e-9:
            best_score, best = score, holding
    return best if best_score >= DEFAULT_NAME_MATCH_THRESHOLD else None


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


def _aggregate_blocked(holding_reports: Sequence[HoldingReport]) -> bool:
    """Return whether an aggregate's XIRR must be withheld.

    True only when a member is *held but unpriced* — a real valuation gap that
    would make the XIRR garbage. A fully-exited member (transactions but zero
    units) is NOT blocking: its cashflows are complete and its XIRR is well
    defined, so it counts normally.
    """
    return any(r.units > 0 and not r.priced for r in holding_reports)


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
    # A held-but-unpriced holding has no meaningful terminal value, so its XIRR
    # would be garbage — withhold it. A fully-exited holding (units 0) keeps its
    # XIRR: buys out and sells in are complete, real cashflows.
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
        matched_txns=len(txns),
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
        incomplete = _aggregate_blocked(sleeve_reports)
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
        incomplete = _aggregate_blocked(members)
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
    zero_txn: Sequence[str],
    closed_positions: tuple[ClosedPosition, ...],
    held_unconfigured: Sequence[str],
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
        zero_txn_holdings=tuple(zero_txn),
        closed_positions=closed_positions,
        held_unconfigured=tuple(held_unconfigured),
        mismatches=mismatches,
    )


def raise_for_join_errors(report: PortfolioReport) -> None:
    """Fail loudly on transaction↔holding join failures (BUG A).

    Raises if any MF holding matched zero transactions, or a NON-ZERO position
    exists for a scheme not in ``portfolio.json`` (the user holds something the
    config omits). Fully-exited (closed) positions are legitimate history and do
    NOT raise.

    Raises:
        ValueError: If there are unjoined holdings or unconfigured holdings.
    """
    problems: list[str] = []
    if report.zero_txn_holdings:
        problems.append(
            "MF holdings with ZERO matched transactions: "
            + ", ".join(report.zero_txn_holdings)
        )
    if report.held_unconfigured:
        problems.append(
            "open positions absent from portfolio.json: "
            + ", ".join(report.held_unconfigured)
        )
    if problems:
        raise ValueError(
            "Transaction join failed — fix scheme names in portfolio.json or add "
            "the missing holding. " + "; ".join(problems)
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
