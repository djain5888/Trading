"""Terminal rendering for the portfolio report (pure, unit-testable)."""

from __future__ import annotations

from collections.abc import Sequence

from app.portfolio.importers.base import ImportResult
from app.portfolio.report import PortfolioReport


def render_import(results: Sequence[ImportResult]) -> str:
    """Render a per-importer summary: rows parsed and rows skipped with reasons."""
    lines = ["===== Portfolio Import ====="]
    for result in results:
        lines.append(
            f"[{result.source}] parsed {result.parsed}, skipped {len(result.skipped)}"
        )
        if result.mapping:
            cols = ", ".join(f"{h}->{f}" for h, f in result.mapping.items())
            lines.append(f"  columns: {cols}")
        for row in result.skipped:
            detail = f" ({row.detail})" if row.detail else ""
            lines.append(f"  - row {row.row}: {row.reason}{detail}")
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    """Format an optional percentage."""
    return "n/a" if value is None else f"{value:+.2f}%"


def render_portfolio(report: PortfolioReport) -> str:
    """Render the full portfolio measurement report."""
    lines = [
        "===== Titan Portfolio =====",
        f"Valued:           {report.valuation_date.isoformat()}",
        f"Market Value:     {report.market_value:,.2f}",
        f"Cost Basis:       {report.cost_basis:,.2f}",
        f"Unrealised P&L:   {report.unrealised_pnl:+,.2f} "
        f"({report.absolute_return_pct:+.2f}% on invested)",
        f"Realised Gain:    {report.realised_gain:+,.2f}",
        f"Portfolio XIRR:   {_pct(report.xirr_pct)}  "
        f"[band {report.band_low:.0f}-{report.band_high:.0f}%: "
        f"{report.band_status.value}]",
    ]
    if report.band_status.value != "WITHIN" and report.drags:
        lines.append(f"Biggest drags:    {', '.join(report.drags)}")
    lines.append(
        f"Concentration:    {report.concentration_id} "
        f"{report.concentration_pct:.1f}% of book"
    )
    lines.append(
        f"LTCG ready:       {report.ltcg_ready_value:,.2f}  "
        f"est. exit tax {report.exit_tax:,.2f}"
    )

    lines.append("")
    lines.append("Sleeves (actual vs target):")
    for sleeve in report.sleeves:
        lines.append(
            f"  {sleeve.sleeve.value:<10} {sleeve.actual_pct:5.1f}% "
            f"(target {sleeve.target_pct:4.0f}%, drift {sleeve.drift_pct:+.1f}pp)  "
            f"XIRR {_pct(sleeve.xirr_pct)}  value {sleeve.market_value:,.0f}"
        )

    lines.append("")
    lines.append("Asset classes:")
    for ac in report.asset_classes:
        lines.append(
            f"  {ac.asset_type.value:<7} value {ac.market_value:,.0f}  "
            f"XIRR {_pct(ac.xirr_pct)}"
        )

    lines.append("")
    lines.append("Holdings:")
    header = (
        f"  {'holding':<14}{'sleeve':<10}{'value':>12}{'unreal':>12}"
        f"{'abs%':>8}{'xirr':>9}{'LTCGready':>12}"
    )
    lines.append(header)
    for h in sorted(report.holdings, key=lambda x: x.market_value, reverse=True):
        price_flag = "" if h.priced else " *unpriced"
        lines.append(
            f"  {h.identifier:<14}{h.sleeve.value:<10}{h.market_value:>12,.0f}"
            f"{h.unrealised_pnl:>+12,.0f}{h.absolute_return_pct:>7.1f}%"
            f"{_pct(h.xirr_pct):>9}{h.ltcg_ready_value:>12,.0f}{price_flag}"
        )

    na_xirr = [
        h.identifier for h in report.holdings if h.units > 0 and h.xirr_pct is None
    ]
    if na_xirr or report.xirr_pct is None:
        lines.append("")
        lines.append(
            "XIRR n/a: undefined cashflows (e.g. fully exited, or same-day "
            f"round trips with no elapsed time): {', '.join(na_xirr) or 'portfolio'}"
        )

    if report.unpriced:
        lines.append("")
        lines.append(f"UNPRICED (excluded from value): {', '.join(report.unpriced)}")
    if report.mismatches:
        lines.append("")
        lines.append("Broker cross-check mismatches:")
        for m in report.mismatches:
            lines.append(
                f"  {m.identifier:<14} imported {m.imported_units} "
                f"vs broker {m.broker_units} (diff {m.difference:+})"
            )
    return "\n".join(lines)
