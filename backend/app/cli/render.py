"""Terminal rendering for CLI output.

Pure functions that turn diagnostic and workflow results into strings, kept
separate from the Typer commands so they are unit-testable.
"""

from __future__ import annotations

from app.scanner.models import ScannerResult
from app.workflow.diagnostics import HealthCheck, VersionInfo
from app.workflow.models import (
    CollectReport,
    ImportReport,
    IndicatorsReport,
    MorningReport,
    ScanReport,
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
        f"Indicators Refreshed:{report.indicators_refreshed}",
    ]
    if report.scanner_summary:
        lines.append("Scanner Results:")
        for name, count in sorted(report.scanner_summary.items()):
            lines.append(f"  {name}: {count}")
    lines.append(f"Top {len(report.top_results)} Ranked Stocks:")
    lines.extend(_render_results(report.top_results))
    lines.append(f"Generated At:        {report.generated_at.isoformat()}")
    return lines


def _render_results(results: tuple[ScannerResult, ...]) -> list[str]:
    """Render a ranked list of scanner results."""
    if not results:
        return ["  (no candidates)"]
    return [
        f"  {index:>2}. {result.symbol:<12} "
        f"score={result.score:6.1f} "
        f"conf={result.confidence:.2f} "
        f"[{result.scanner_name}]"
        for index, result in enumerate(results, start=1)
    ]
