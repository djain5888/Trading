"""Terminal rendering for CLI output.

Pure functions that turn diagnostic and workflow results into strings, kept
separate from the Typer commands so they are unit-testable.
"""

from __future__ import annotations

from app.backtest.models import (
    BacktestReport,
    ExecutionLeakage,
    ExitAnalysis,
    ExitBreakdown,
)
from app.market.regime.models import RegimeReport
from app.market.relative.models import RSReport
from app.market.sector.models import SectorReport
from app.paper.models import PaperReport, PaperTrade
from app.scanner.models import ScannerResult
from app.strategy.models import StrategyReport
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
    StrategyWorkflowReport,
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
    if isinstance(report, StrategyWorkflowReport):
        return _render_strategies(report.strategies)
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
    lines.extend(_strategy_section(report.strategies))
    lines.append(f"Paper Book:          {_paper_summary(report.paper)}")
    lines.append(f"Generated At:        {report.generated_at.isoformat()}")
    return lines


def _paper_summary(report: PaperReport | None) -> str:
    """Format a one-line paper-book summary for the morning report."""
    if report is None:
        return "unavailable"
    return (
        f"equity={report.equity:.0f}  open={len(report.open_positions)}  "
        f"closed={len(report.closed_trades)}  win={report.win_rate:.0f}%  "
        f"pnl={report.total_pnl:+.0f}  pf={report.profit_factor:.2f}"
    )


def render_paper(report: PaperReport, *, history: bool = False) -> str:
    """Render the paper-trading performance report (``--history`` adds closed)."""
    lines = [
        "===== Titan Paper Book =====",
        f"Starting Capital: {report.starting_capital:.0f}",
        f"Equity:           {report.equity:.2f}",
        f"Total P&L:        {report.total_pnl:+.2f} "
        f"(realized {report.realized_pnl:+.2f}, open {report.open_pnl:+.2f})",
        f"Win Rate:         {report.win_rate:.1f}%  "
        f"({report.wins}W / {report.losses}L)  PF={report.profit_factor:.2f}",
        f"Avg Win/Loss:     {report.avg_win:+.2f} / {report.avg_loss:+.2f}",
        f"Open Positions ({len(report.open_positions)}):",
    ]
    lines.extend(_render_positions(report.open_positions))
    if report.per_strategy:
        lines.append("Per-Strategy:")
        for stats in report.per_strategy:
            lines.append(
                f"  {stats.strategy:<10} trades={stats.trades} "
                f"win={stats.win_rate:.0f}% pnl={stats.total_pnl:+.2f}"
            )
    if history:
        lines.append(f"Closed Trades ({len(report.closed_trades)}):")
        lines.extend(_render_closed(report.closed_trades))
    lines.append(f"Generated: {report.generated_at.isoformat()}")
    return "\n".join(lines)


def _render_positions(trades: tuple[PaperTrade, ...]) -> list[str]:
    """Render open paper positions (mark-to-market)."""
    if not trades:
        return ["  (none)"]
    return [
        f"  {t.symbol:<12} {t.strategy:<9} entry={t.entry_price:.2f} "
        f"stop={t.stop_price:.2f} target={t.target_price:.2f} "
        f"mark={t.last_price:.2f} pnl={t.pnl:+.2f}"
        for t in trades
    ]


def _render_closed(trades: tuple[PaperTrade, ...]) -> list[str]:
    """Render closed paper trades with their exit reason."""
    if not trades:
        return ["  (none)"]
    return [
        f"  {t.symbol:<12} {t.strategy:<9} "
        f"{t.exit_reason.value if t.exit_reason else '?':<7} "
        f"entry={t.entry_price:.2f} exit={t.exit_price or 0:.2f} pnl={t.pnl:+.2f}"
        for t in trades
    ]


def _strategy_section(report: StrategyReport | None) -> list[str]:
    """Render the morning strategy block, always visible ("none" when empty)."""
    setups = report.setups if report is not None and report.available else ()
    if not setups:
        detail = f" ({report.detail})" if report is not None else ""
        return [f"Strategy Setups:     none{detail}"]
    return ["Strategy Setups:", *_render_setups(report)]


def _render_setups(report: StrategyReport | None, limit: int = 10) -> list[str]:
    """Render the top named strategy setups (symbol, strategy, confidence, RR)."""
    if report is None or not report.available or not report.setups:
        return ["  none"]
    return [
        f"  {index:>2}. {setup.symbol:<12} {setup.strategy:<9} "
        f"conf={setup.confidence:5.1f} RR={setup.reward_risk:.1f} "
        f"entry={setup.entry:.2f} stop={setup.stop:.2f} target={setup.target:.2f}"
        for index, setup in enumerate(report.setups[:limit], start=1)
    ]


def _render_strategies(report: StrategyReport) -> list[str]:
    """Render a standalone strategy report."""
    lines = ["", "===== Titan Strategy Setups ====="]
    if not report.available:
        lines.append("Strategies: unavailable")
        lines.append(f"Detail:     {report.detail}")
    else:
        lines.extend(_render_setups(report, limit=len(report.setups)))
    lines.append(f"Generated:  {report.generated_at.isoformat()}")
    return lines


def render_backtest(report: BacktestReport) -> str:
    """Render the backtest expectancy report to the terminal."""
    lines = [
        "===== Titan Backtest =====",
        f"Window:           {report.start.isoformat()} -> {report.end.isoformat()}",
        f"Universe:         {len(report.symbols)} symbol(s)",
        f"Starting Capital: {report.starting_capital:.0f}",
        f"Ending Equity:    {report.ending_equity:.2f} "
        f"({report.total_return_pct:+.2f}%)",
        f"Trades:           {report.trades}  " f"({report.wins}W / {report.losses}L)",
        f"Win Rate:         {report.win_rate:.1f}%",
        f"Expectancy:       {report.expectancy_r:+.3f} R/trade  "
        f"PF={report.profit_factor:.2f}",
        f"Avg Win/Loss:     {report.avg_win:+.2f} / {report.avg_loss:+.2f}",
        f"Max Drawdown:     {report.max_drawdown_pct:.2f}%  "
        f"Longest Losing Streak: {report.longest_losing_streak}",
        f"Total P&L:        {report.total_pnl:+.2f}",
    ]
    if report.per_strategy:
        lines.append("Per-Strategy:")
        for stats in report.per_strategy:
            lines.append(
                f"  {stats.strategy:<10} trades={stats.trades} "
                f"win={stats.win_rate:.0f}% exp={stats.expectancy_r:+.2f}R "
                f"pf={stats.profit_factor:.2f} pnl={stats.total_pnl:+.2f}"
            )
    if report.per_regime:
        lines.append("Per-Regime:")
        for regime in report.per_regime:
            lines.append(
                f"  {regime.regime:<10} trades={regime.trades} "
                f"win={regime.win_rate:.0f}% exp={regime.expectancy_r:+.2f}R "
                f"pnl={regime.total_pnl:+.2f}"
            )
    if report.monthly:
        lines.append("Monthly:")
        for month in report.monthly:
            lines.append(
                f"  {month.month} trades={month.trades} "
                f"pnl={month.net_pnl:+.2f} ({month.return_pct:+.2f}%)"
            )
    lines.extend(_render_exit_analysis(report.exit_analysis))
    if report.trades:
        lines.extend(_render_execution_leakage(report.execution_leakage))
    lines.append(f"Generated:        {report.generated_at.isoformat()}")
    return "\n".join(lines)


def _render_execution_leakage(leakage: ExecutionLeakage) -> list[str]:
    """Render the execution-leakage diagnostics."""
    stops, targets, rc, pnl = (
        leakage.stops,
        leakage.targets,
        leakage.r_consistency,
        leakage.pnl,
    )
    lines = ["Execution Leakage:"]
    if stops.trades:
        lines.append(
            f"  Stops ({stops.trades}): overshoot avg={stops.avg_overshoot_pct:+.3f}% "
            f"median={stops.median_overshoot_pct:+.3f}% "
            f"worst={stops.worst_overshoot_pct:+.3f}%"
        )
        lines.append(
            f"    beyond -1R: {stops.avg_overshoot_r:+.3f}R "
            f"(gap={stops.avg_gap_r:+.3f} slip={stops.avg_slippage_r:+.3f} "
            f"cost={stops.avg_cost_r:+.3f})"
        )
    if targets.trades:
        lines.append(
            f"  Targets ({targets.trades}): intended={targets.intended_r:.2f}R "
            f"realised={targets.avg_realised_r:.2f}R "
            f"(vs intended-risk {targets.avg_realised_r_intended_risk:.2f}R)  "
            f"entry drift={targets.avg_entry_displacement_pct:+.3f}%"
        )
    lines.append(
        f"  R audit: recorded={rc.avg_r_recorded:+.3f}R "
        f"intended-risk={rc.avg_r_intended_risk:+.3f}R "
        f"|diff|={rc.avg_abs_discrepancy_r:.3f}R "
        f"(actual-entry denom={rc.denominator_uses_actual_entry})"
    )
    lines.append(
        f"  P&L split: edge={pnl.strategy_edge:+.2f} "
        f"entry_gap={pnl.entry_gap:+.2f} entry_slip={pnl.entry_slippage:+.2f} "
        f"exit_slip={pnl.exit_slippage:+.2f} charges={pnl.charges:+.2f} "
        f"=> net={pnl.net_pnl:+.2f}"
    )
    lines.append(
        f"  Perfect-exec expectancy @ {leakage.win_rate:.1f}% win: "
        f"{leakage.perfect_expectancy_r:+.3f}R/trade"
    )
    return lines


def _render_exit_breakdown(rows: tuple[ExitBreakdown, ...]) -> list[str]:
    """Render one exit-reason breakdown block."""
    return [
        f"    {row.kind:<16} n={row.trades:<4} win={row.wins:<4} "
        f"avgR={row.avg_r:+.2f} hold={row.avg_holding_days:.1f}d "
        f"pnl={row.total_pnl:+.2f}"
        for row in rows
    ]


def _render_exit_analysis(analysis: ExitAnalysis) -> list[str]:
    """Render the exit diagnostics (breakdown, winner R, slippage, BULL)."""
    if not analysis.by_reason:
        return []
    dist = analysis.winner_r
    lines = ["Exit Analysis:", "  By exit reason:"]
    lines.extend(_render_exit_breakdown(analysis.by_reason))
    lines.append(
        f"  Winners' R:      {dist.winners} total  "
        f">=2R={dist.reached_2r}  1-2R={dist.between_1_and_2r}  "
        f"0-1R={dist.between_0_and_1r}"
    )
    lines.append(
        f"  Timeouts:        {analysis.timeout_trades} closed "
        f"({analysis.timeout_profitable} profitable at close)"
    )
    lines.append(
        f"  Entry slippage:  {analysis.avg_entry_slippage:+.4f}/share "
        f"({analysis.avg_entry_slippage_pct:+.3f}% vs signal)"
    )
    if analysis.bull_by_reason:
        lines.append("  BULL-regime exits:")
        lines.extend(_render_exit_breakdown(analysis.bull_by_reason))
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
