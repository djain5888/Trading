"""Terminal rendering for CLI output.

Pure functions that turn diagnostic and workflow results into strings, kept
separate from the Typer commands so they are unit-testable.
"""

from __future__ import annotations

from app.market.regime.models import RegimeReport
from app.market.relative.models import RSReport
from app.market.sector.models import SectorReport
from app.scanner.models import ScannerResult
from app.workflow.diagnostics import HealthCheck, VersionInfo
from app.workflow.models import (
    CollectReport,
    ImportReport,
    IndicatorsReport,
    MorningReport,
    RegimeWorkflowReport,
    RSWorkflowReport,
    ScanReport,
    SectorWorkflowReport,
    WorkflowRun,
)


def render_version(info: VersionInfo) -> str:
    """Render version information."""
    return "\n".join(
        [
            "Titan",
            f"  Version:     {info.app_version}",
            f"  Git commit:  {info.git_commit}",
            f"  Python:      {info.python_version}",
            f"  Environment: {info.environment}",
        ]
    )


def render_health(checks: list[HealthCheck]) -> str:
    """Render a list of health checks."""
    lines = ["Titan health:"]
    for check in checks:
        mark = "OK  " if check.healthy else "FAIL"
        tag = "" if check.critical else " (non-critical)"
        lines.append(f"  [{mark}] {check.name}{tag}: {check.detail}")
    return "\n".join(lines)


def render_run(run: WorkflowRun) -> str:
    """Render a workflow run: header, steps, error and report."""
    status = "OK" if run.success else "FAILED"
    lines = [f"Workflow '{run.workflow}': {status} ({run.elapsed_seconds:.3f}s)"]
    for step in run.steps:
        mark = "✓" if step.ok else "✗"
        suffix = f" - {step.error}" if step.error else ""
        lines.append(f"  {mark} {step.name} ({step.duration_seconds:.3f}s){suffix}")
    if run.error:
        lines.append(f"Error: {run.error}")
    lines.extend(_render_report(run.report))
    return "\n".join(lines)


def _render_report(report: object) -> list[str]:
    """Render the workflow-specific report body."""
    if isinstance(report, MorningReport):
        return _render_morning(report)
    if isinstance(report, ScanReport):
        return ["", f"Stocks scanned: {report.stocks_scanned}"] + _render_results(
            report.results
        )
    if isinstance(report, ImportReport):
        return [
            "",
            f"Symbols processed: {report.symbols_processed}",
            f"Candles imported:  {report.candles_imported}",
            f"Failed requests:   {report.failed_requests}",
        ]
    if isinstance(report, RegimeWorkflowReport):
        return _render_regime(report.regime)
    if isinstance(report, SectorWorkflowReport):
        return _render_sectors(report.sectors)
    if isinstance(report, RSWorkflowReport):
        return _render_rs(report.relative)
    if isinstance(report, IndicatorsReport):
        return ["", f"Computed: {report.computed}  Skipped: {report.skipped}"]
    if isinstance(report, CollectReport):
        return [
            "",
            f"Cycles: {report.cycles}  "
            f"Updates: {report.successful_updates}  "
            f"Failed: {report.failed_updates}",
        ]
    return []


def _render_morning(report: MorningReport) -> list[str]:
    """Render the morning report."""
    lines = [
        "",
        "===== Titan Morning Report =====",
        f"Market Status:       {'OPEN' if report.market_open else 'CLOSED'}",
        f"Market State:        {report.market_state}",
        f"Stocks Scanned:      {report.stocks_scanned}",
        f"Import:              {_format_import(report)}",
        f"Indicators Refreshed:{report.indicators_refreshed}",
        f"Market Regime:       {_regime_summary(report.regime)}",
        f"Sector Strength:     {_sector_summary(report.sectors)}",
        f"Relative Strength:   {_rs_summary(report.relative)}",
    ]
    if report.scanner_summary:
        lines.append("Scanner Results:")
        for name, count in sorted(report.scanner_summary.items()):
            lines.append(f"  {name}: {count}")
    lines.append(f"Top {len(report.top_results)} Ranked Stocks:")
    lines.extend(_render_results(report.top_results, report.relative))
    lines.append(f"Generated At:        {report.generated_at.isoformat()}")
    return lines


def _regime_summary(report: RegimeReport | None) -> str:
    """Format a one-line regime summary for the morning report."""
    if report is None or not report.available or report.regime is None:
        return "unavailable"
    volatility = report.volatility.value if report.volatility else "?"
    return (
        f"{report.regime.value} (conf {report.confidence:.0f})  "
        f"vol={volatility}  breadth={report.breadth:.0f}%"
    )


def _render_regime(report: RegimeReport) -> list[str]:
    """Render a standalone market-regime report."""
    lines = [
        "",
        "===== Titan Market Regime =====",
        f"Index:      {report.index_symbol}",
    ]
    if not report.available or report.regime is None:
        lines.append("Regime:     unavailable")
        lines.append(f"Detail:     {report.detail}")
    else:
        volatility = report.volatility.value if report.volatility else "?"
        lines.extend(
            [
                f"Regime:     {report.regime.value} "
                f"(confidence {report.confidence:.0f})",
                f"Volatility: {volatility}",
                f"Breadth:    {report.breadth:.0f}% of watchlist above 50 EMA",
            ]
        )
    lines.append(f"Generated:  {report.generated_at.isoformat()}")
    return lines


def _sector_summary(report: SectorReport | None) -> str:
    """Format a one-line sector summary for the morning report."""
    if report is None or not report.available:
        return "unavailable"
    if not report.ranked:
        return "no sector had enough valid members"
    strongest = report.strongest[0]
    weakest = report.weakest[0]
    return (
        f"strongest {strongest.sector} ({strongest.score:.0f}, "
        f"{strongest.trend.value}), weakest {weakest.sector} "
        f"({weakest.score:.0f}, {weakest.trend.value})"
    )


def _render_sectors(report: SectorReport) -> list[str]:
    """Render a standalone sector-strength report."""
    lines = ["", "===== Titan Sector Strength ====="]
    if not report.available:
        lines.append("Sectors:    unavailable")
        lines.append(f"Detail:     {report.detail}")
    elif not report.ranked:
        lines.append("Sectors:    no sector had enough valid members")
    else:
        for index, sector in enumerate(report.ranked, start=1):
            lines.append(
                f"  {index:>2}. {sector.sector:<14} "
                f"score={sector.score:5.1f} "
                f"trend={sector.trend.value:<4} "
                f"above={sector.above_pct:.0f}% "
                f"({sector.members} members)"
            )
    lines.append(f"Generated:  {report.generated_at.isoformat()}")
    return lines


def _format_import(report: MorningReport) -> str:
    """Format import counts, e.g. ``24 ok / 3 failed (XYZ, ABC, PQR)``."""
    line = f"{report.imported_symbols} ok / {report.failed_symbols} failed"
    if report.failed_symbol_names:
        line += f" ({', '.join(report.failed_symbol_names)})"
    return line


def _render_results(
    results: tuple[ScannerResult, ...], relative: RSReport | None = None
) -> list[str]:
    """Render a ranked list of scanner results, annotating RS when available."""
    if not results:
        return ["  (no candidates)"]
    by_symbol = relative.by_symbol if relative and relative.available else {}
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        entry = by_symbol.get(result.symbol)
        rs = (
            f" RS={entry.composite:.0f}{'*' if entry.leader else ''}"
            if entry is not None
            else ""
        )
        lines.append(
            f"  {index:>2}. {result.symbol:<12} "
            f"score={result.score:6.1f} "
            f"conf={result.confidence:.2f} "
            f"[{result.scanner_name}]{rs}"
        )
    return lines


def _rs_summary(report: RSReport | None) -> str:
    """Format a one-line relative-strength summary for the morning report."""
    if report is None or not report.available:
        return "unavailable"
    market = "" if report.market_available else " (market neutral)"
    if not report.leaders:
        return f"no leaders{market}"
    leaders = ", ".join(entry.symbol for entry in report.leaders[:5])
    return f"{len(report.leaders)} leader(s): {leaders}{market}"


def _render_rs(report: RSReport) -> list[str]:
    """Render a standalone relative-strength report."""
    lines = ["", "===== Titan Relative Strength ====="]
    if not report.available:
        lines.append("Relative Strength: unavailable")
        lines.append(f"Detail:     {report.detail}")
    else:
        if not report.market_available:
            lines.append("(NIFTY absent — vs-market component is neutral)")
        for index, entry in enumerate(report.entries, start=1):
            flag = " LEADER" if entry.leader else ""
            lines.append(
                f"  {index:>2}. {entry.symbol:<12} "
                f"mkt={entry.rs_vs_market:5.1f} "
                f"sec={entry.rs_vs_sector:5.1f} "
                f"composite={entry.composite:5.1f}{flag}"
            )
    lines.append(f"Generated:  {report.generated_at.isoformat()}")
    return lines
