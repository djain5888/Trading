"""The Titan command-line interface.

A thin Typer layer over the workflow engine and diagnostics. Commands resolve
services through DI, run a workflow, render the result, and set a non-zero exit
code on failure. They contain no business logic.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import TYPE_CHECKING

import typer

from app.cli.render import (
    render_backtest,
    render_health,
    render_paper,
    render_run,
    render_version,
)
from app.config.settings import DEFAULT_HISTORY_DAYS
from app.config.watchlist import get_watchlist_config
from app.core.logging import configure_logging, get_logger
from app.market.enums import Exchange, Interval
from app.workflow.diagnostics import (
    is_healthy,
    run_health_checks,
    version_info,
)
from app.workflow.engine import build_workflow_engine
from app.workflow.models import WorkflowRequest, WorkflowRun
from app.workflow.services import WorkflowServices

if TYPE_CHECKING:
    from app.backtest.adjust import SplitSummary

app = typer.Typer(help="Titan — AI quant trading platform CLI.", no_args_is_help=True)
logger = get_logger(__name__)


class TierChoice(StrEnum):
    """The ``--tier`` selection for ``titan validate``."""

    ALL = "all"
    LARGE = "large"
    MID = "mid"
    SMALL = "small"


#: Loggers that emit per-symbol / per-request noise during a wide validation.
_NOISY_IMPORT_LOGGERS = (
    "app.providers.groww.sdk_provider",
    "app.market.historical.importer.engine",
    "app.backtest.dependencies",
)


@contextmanager
def _quiet_import_logs(enabled: bool) -> Iterator[None]:
    """Temporarily raise noisy import loggers to WARNING when ``enabled``."""
    if not enabled:
        yield
        return
    saved = {name: logging.getLogger(name).level for name in _NOISY_IMPORT_LOGGERS}
    for name in _NOISY_IMPORT_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        yield
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)


def _resolve_services() -> WorkflowServices:
    """Resolve services through DI (overridable in tests)."""
    return WorkflowServices.resolve()


def _watchlist(symbol: list[str]) -> tuple[str, ...]:
    """Return the given symbols, or the configured default watchlist if none."""
    return tuple(symbol) if symbol else get_watchlist_config().symbols


def _report_adjustments(summaries: list[SplitSummary]) -> set[str]:
    """Echo the split/bonus adjustment summary; return unusable symbol names."""
    from app.cli.render import render_adjustments

    if summaries:
        typer.echo(render_adjustments(summaries))
    return {s.symbol.strip().upper() for s in summaries if not s.usable}


def _run_workflow(name: str, request: WorkflowRequest) -> WorkflowRun:
    """Run a workflow via the engine and return its outcome."""
    services = _resolve_services()
    configure_logging(services.settings)
    engine = build_workflow_engine(services=services)
    return asyncio.run(engine.run(name, request))


def _emit(run: WorkflowRun) -> None:
    """Print a workflow run and exit non-zero on failure."""
    typer.echo(render_run(run))
    if not run.success:
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Show application version, git commit, Python and environment."""
    typer.echo(render_version(version_info()))


@app.command()
def health() -> None:
    """Verify configuration, database, provider, calendar and dependencies."""
    checks = run_health_checks()
    typer.echo(render_health(checks))
    if not is_healthy(checks):
        raise typer.Exit(code=1)


@app.command()
def morning(
    symbol: list[str] = typer.Option([], "--symbol", "-s", help="Symbol to include."),
    exchange: Exchange = typer.Option(Exchange.NSE, help="Listing exchange."),
    interval: Interval = typer.Option(Interval.ONE_DAY, help="Candle interval."),
    history_days: int = typer.Option(
        DEFAULT_HISTORY_DAYS, help="Days of history to consider."
    ),
) -> None:
    """Run the full pre-market pipeline and print the morning report."""
    request = WorkflowRequest(
        symbols=_watchlist(symbol),
        exchange=exchange,
        interval=interval,
        history_days=history_days,
    )
    _emit(_run_workflow("morning", request))


@app.command(name="import")
def import_(
    symbol: list[str] = typer.Option([], "--symbol", "-s", help="Symbol to import."),
    exchange: Exchange = typer.Option(Exchange.NSE, help="Listing exchange."),
    interval: Interval = typer.Option(Interval.ONE_DAY, help="Candle interval."),
    history_days: int = typer.Option(
        DEFAULT_HISTORY_DAYS, help="Days of history to import."
    ),
) -> None:
    """Update historical candles for the given symbols."""
    request = WorkflowRequest(
        symbols=_watchlist(symbol),
        exchange=exchange,
        interval=interval,
        history_days=history_days,
    )
    _emit(_run_workflow("import", request))


@app.command()
def indicators(
    symbol: list[str] = typer.Option([], "--symbol", "-s", help="Symbol to compute."),
    exchange: Exchange = typer.Option(Exchange.NSE, help="Listing exchange."),
    interval: Interval = typer.Option(Interval.ONE_DAY, help="Candle interval."),
    history_days: int = typer.Option(
        DEFAULT_HISTORY_DAYS, help="Days of history to use."
    ),
) -> None:
    """Refresh indicators for the given symbols."""
    request = WorkflowRequest(
        symbols=_watchlist(symbol),
        exchange=exchange,
        interval=interval,
        history_days=history_days,
    )
    _emit(_run_workflow("indicators", request))


@app.command()
def scan(
    symbol: list[str] = typer.Option([], "--symbol", "-s", help="Symbol to scan."),
    scanner: list[str] = typer.Option([], "--scanner", help="Scanner to run."),
    exchange: Exchange = typer.Option(Exchange.NSE, help="Listing exchange."),
    interval: Interval = typer.Option(Interval.ONE_DAY, help="Candle interval."),
    history_days: int = typer.Option(
        DEFAULT_HISTORY_DAYS, help="Days of history to use."
    ),
    top: int = typer.Option(20, help="Number of ranked results to show."),
) -> None:
    """Run scanners and print ranked candidates."""
    request = WorkflowRequest(
        symbols=_watchlist(symbol),
        scanners=tuple(scanner),
        exchange=exchange,
        interval=interval,
        history_days=history_days,
        top_n=top,
    )
    _emit(_run_workflow("scan", request))


@app.command()
def regime(
    symbol: list[str] = typer.Option([], "--symbol", "-s", help="Watchlist symbol."),
    exchange: Exchange = typer.Option(Exchange.NSE, help="Listing exchange."),
    interval: Interval = typer.Option(Interval.ONE_DAY, help="Candle interval."),
    history_days: int = typer.Option(500, help="Days of history to consider."),
) -> None:
    """Classify the market regime (trend, volatility and breadth)."""
    request = WorkflowRequest(
        symbols=_watchlist(symbol),
        exchange=exchange,
        interval=interval,
        history_days=history_days,
    )
    _emit(_run_workflow("regime", request))


@app.command()
def sectors(
    symbol: list[str] = typer.Option([], "--symbol", "-s", help="Watchlist symbol."),
    exchange: Exchange = typer.Option(Exchange.NSE, help="Listing exchange."),
    interval: Interval = typer.Option(Interval.ONE_DAY, help="Candle interval."),
    history_days: int = typer.Option(500, help="Days of history to consider."),
) -> None:
    """Rank sectors by strength across the watchlist."""
    request = WorkflowRequest(
        symbols=_watchlist(symbol),
        exchange=exchange,
        interval=interval,
        history_days=history_days,
    )
    _emit(_run_workflow("sectors", request))


@app.command()
def rs(
    symbol: list[str] = typer.Option([], "--symbol", "-s", help="Watchlist symbol."),
    exchange: Exchange = typer.Option(Exchange.NSE, help="Listing exchange."),
    interval: Interval = typer.Option(Interval.ONE_DAY, help="Candle interval."),
    history_days: int = typer.Option(500, help="Days of history to consider."),
) -> None:
    """Score relative strength versus the market and each stock's sector."""
    request = WorkflowRequest(
        symbols=_watchlist(symbol),
        exchange=exchange,
        interval=interval,
        history_days=history_days,
    )
    _emit(_run_workflow("rs", request))


@app.command()
def strategies(
    symbol: list[str] = typer.Option([], "--symbol", "-s", help="Watchlist symbol."),
    exchange: Exchange = typer.Option(Exchange.NSE, help="Listing exchange."),
    interval: Interval = typer.Option(Interval.ONE_DAY, help="Candle interval."),
    history_days: int = typer.Option(400, help="Days of history to consider."),
) -> None:
    """Name strategy setups (breakout, pullback, momentum) with confidence and RR."""
    request = WorkflowRequest(
        symbols=_watchlist(symbol),
        exchange=exchange,
        interval=interval,
        history_days=history_days,
    )
    _emit(_run_workflow("strategies", request))


@app.command()
def collect(
    symbol: list[str] = typer.Option([], "--symbol", "-s", help="Symbol to collect."),
    exchange: Exchange = typer.Option(Exchange.NSE, help="Listing exchange."),
    interval: Interval = typer.Option(Interval.ONE_MINUTE, help="Snapshot interval."),
    cycles: int = typer.Option(1, help="Number of poll cycles."),
) -> None:
    """Run a bounded number of live-collection cycles."""
    request = WorkflowRequest(
        symbols=tuple(symbol),
        exchange=exchange,
        interval=interval,
        collect_cycles=cycles,
    )
    _emit(_run_workflow("collect", request))


@app.command()
def paper(
    history: bool = typer.Option(
        False, "--history", help="Include the closed-trade history."
    ),
) -> None:
    """Show paper-trading performance (open positions, win rate, P&L)."""
    services = _resolve_services()
    configure_logging(services.settings)
    report = asyncio.run(services.paper_engine.report())
    typer.echo(render_paper(report, history=history))


@app.command()
def backtest(
    from_: str = typer.Option(..., "--from", help="Replay start date (YYYY-MM-DD)."),
    to: str = typer.Option(..., "--to", help="Replay end date (YYYY-MM-DD)."),
    symbol: list[str] = typer.Option([], "--symbol", "-s", help="Symbol to include."),
    exchange: Exchange = typer.Option(Exchange.NSE, help="Listing exchange."),
    interval: Interval = typer.Option(Interval.ONE_DAY, help="Candle interval."),
) -> None:
    """Replay strategy setups over history and print an expectancy report."""
    from datetime import date

    from app.backtest.dependencies import build_backtest_engine, load_backtest_history
    from app.backtest.models import BacktestConfig, BacktestReport

    services = _resolve_services()
    configure_logging(services.settings)
    universe = list(_watchlist(symbol))
    config = BacktestConfig(exchange=exchange, interval=interval)
    start, end = date.fromisoformat(from_), date.fromisoformat(to)

    async def _run() -> BacktestReport:
        run_universe = list(universe)
        try:
            # Load candles first (like the morning workflow) so the replay sees
            # the same data; the store is per-process and starts empty.
            summaries = await load_backtest_history(
                services, universe, start, end, config
            )
            unusable = _report_adjustments(summaries)
            run_universe = [s for s in universe if s.strip().upper() not in unusable]
        except Exception as exc:  # noqa: BLE001 - best-effort; the engine warns loudly
            logger.warning("Backtest history load failed: %s", exc)
        return await build_backtest_engine(services, config).run(
            run_universe, start, end
        )

    typer.echo(render_backtest(asyncio.run(_run())))


@app.command()
def baseline(
    from_: str = typer.Option(..., "--from", help="Replay start date (YYYY-MM-DD)."),
    to: str = typer.Option(..., "--to", help="Replay end date (YYYY-MM-DD)."),
    strategy: str = typer.Option(
        "", "--strategy", help="Baseline name (omit with --all)."
    ),
    all_baselines: bool = typer.Option(
        False, "--all", help="Run every baseline and print a comparison table."
    ),
    symbol: list[str] = typer.Option([], "--symbol", "-s", help="Symbol to include."),
    exchange: Exchange = typer.Option(Exchange.NSE, help="Listing exchange."),
    interval: Interval = typer.Option(Interval.ONE_DAY, help="Candle interval."),
) -> None:
    """Search well-known baselines for any positive-expectancy edge."""
    from datetime import date

    from app.backtest.baselines.dependencies import build_baseline_engine
    from app.backtest.baselines.models import BaselineName, BaselineResult
    from app.backtest.dependencies import load_backtest_history
    from app.backtest.models import BacktestConfig, BacktestReport
    from app.cli.render import render_baseline_comparison

    services = _resolve_services()
    configure_logging(services.settings)
    universe = list(_watchlist(symbol))
    config = BacktestConfig(exchange=exchange, interval=interval)
    start, end = date.fromisoformat(from_), date.fromisoformat(to)

    if not all_baselines and not strategy:
        raise typer.BadParameter("Provide --strategy <name> or --all.")
    try:
        names = (
            list(BaselineName)
            if all_baselines
            else [BaselineName(strategy.strip().lower())]
        )
    except ValueError as exc:
        raise typer.BadParameter(f"Unknown baseline '{strategy}'.") from exc

    async def _run() -> list[tuple[BaselineName, BacktestReport]]:
        run_universe = list(universe)
        try:
            summaries = await load_backtest_history(
                services, universe, start, end, config
            )
            unusable = _report_adjustments(summaries)
            run_universe = [s for s in universe if s.strip().upper() not in unusable]
        except Exception as exc:  # noqa: BLE001 - best-effort; each run warns loudly
            logger.warning("Baseline history load failed: %s", exc)
        reports: list[tuple[BaselineName, BacktestReport]] = []
        for name in names:
            report = await build_baseline_engine(services, name, config=config).run(
                run_universe, start, end
            )
            reports.append((name, report))
        return reports

    try:
        reports = asyncio.run(_run())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if all_baselines:
        results = [BaselineResult.from_report(n.value, r) for n, r in reports]
        typer.echo(render_baseline_comparison(results))
    else:
        typer.echo(render_backtest(reports[0][1]))


@app.command()
def validate(
    windows: int = typer.Option(3, help="Number of non-overlapping windows."),
    window_months: int = typer.Option(12, help="Length of each window, in months."),
    symbol: list[str] = typer.Option([], "--symbol", "-s", help="Symbol to include."),
    wide: bool = typer.Option(
        False,
        "--wide",
        help="Use the bundled wide NSE universe and report a per-cap-tier split.",
    ),
    tier: TierChoice = typer.Option(
        TierChoice.ALL,
        "--tier",
        help="Run one cap tier alone (large/mid/small) to cut memory; 'all' = full.",
    ),
    symbols_limit: int = typer.Option(
        0,
        "--symbols-limit",
        help="Cap the universe to N symbols (balanced across tiers); 0 = no cap.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", help="Suppress per-symbol import logs during the run."
    ),
    exchange: Exchange = typer.Option(Exchange.NSE, help="Listing exchange."),
    interval: Interval = typer.Option(Interval.ONE_DAY, help="Candle interval."),
) -> None:
    """Validate the exploration candidates across non-overlapping history windows."""
    from datetime import date, timedelta

    from app.backtest import execution
    from app.backtest.baselines.models import EXPLORATION_CANDIDATES, BaselineConfig
    from app.backtest.dependencies import load_backtest_history
    from app.backtest.guard import LookaheadGuard
    from app.backtest.models import BacktestConfig
    from app.backtest.validation import (
        TieredValidationReport,
        ValidationReport,
        ValidationRunner,
        check_history,
        limit_by_tier,
        longest_span,
        majority_span,
        partition_by_tier,
    )
    from app.cli.render import render_tiered_validation, render_validation
    from app.config.watchlist import load_wide_universe

    services = _resolve_services()
    configure_logging(services.settings)
    single_tier = tier is not TierChoice.ALL
    # A single-tier run implies the wide universe (that is where tiers live) but
    # loads ONLY that tier's symbols, so peak memory scales with the tier, not 60.
    use_wide = wide or single_tier
    if use_wide:
        watchlist = load_wide_universe()
        tiers = dict(watchlist.tiers)
        universe = list(watchlist.symbols)
        if single_tier:
            universe = partition_by_tier(universe, tiers).get(tier.value, [])
    else:
        tiers = {}
        universe = list(_watchlist(symbol))
    if symbols_limit > 0:
        universe = (
            limit_by_tier(universe, tiers, symbols_limit)
            if tiers and not single_tier
            else universe[:symbols_limit]
        )
    label = tier.value if single_tier else ("wide" if wide else "default")
    typer.echo(f"Universe: {len(universe)} symbol(s) [{label}]")
    if not universe:
        typer.echo("Empty universe — nothing to validate.")
        raise typer.Exit(code=1)
    config = BacktestConfig(exchange=exchange, interval=interval)
    tuning = BaselineConfig()
    runner = ValidationRunner(
        data_engine=services.data_engine,
        indicators=services.indicator_engine,
        calendar=services.calendar,
        config=config,
        baseline_config=tuning,
    )

    async def _run() -> ValidationReport | TieredValidationReport | None:
        # Pull the maximum history the provider serves, then use all of it.
        wide_end = date.today()
        wide_start = wide_end - timedelta(days=8 * 365)
        try:
            summaries = await load_backtest_history(
                services, universe, wide_start, wide_end, config
            )
            unusable = _report_adjustments(summaries)
            # Exclude symbols whose prices could not be reliably split-adjusted.
            universe[:] = [s for s in universe if s.strip().upper() not in unusable]
        except Exception as exc:  # noqa: BLE001 - best-effort; span check reports it
            logger.warning("Validation history load failed: %s", exc)
        guard = LookaheadGuard(services.data_engine)
        spans = await execution.stored_spans(guard, universe, exchange, interval)
        typer.echo("Confirmed available history (true earliest per symbol):")
        for name, first, last, count in spans:
            typer.echo(f"  {name:<12} {first}..{last}  ({count} candles)")
        if not spans:
            typer.echo("No stored candles for the universe — cannot validate.")
            return None

        # Per-window eligibility (TASK-027): lay windows over the span a MAJORITY
        # of symbols support, not the single shortest one, and refuse only if
        # even the longest-history symbol cannot cover the request.
        symbol_spans = {name: (first, last) for name, first, last, _ in spans}
        majority = majority_span(spans)
        longest = longest_span(spans)
        assert majority is not None and longest is not None  # spans is non-empty
        typer.echo(
            f"Available history — majority span {majority[0]}..{majority[1]}, "
            f"longest span {longest[0]}..{longest[1]}."
        )
        chosen: tuple[date, date] | None = None
        required_days = 0
        for candidate in (majority, longest):
            check = check_history(
                candidate[0],
                candidate[1],
                lookback_days=tuning.history_days,
                windows=windows,
                window_months=window_months,
            )
            if check.ok:
                chosen, required_days = candidate, check.required_days
                break
        if chosen is None:
            # Even the longest-history symbol cannot cover the request — refuse.
            typer.echo(check.message)
            return None
        # Lay windows over the most recent, sufficiently-long slice of the span.
        start = chosen[1] - timedelta(days=required_days)
        typer.echo(
            f"Validating {start}..{chosen[1]} over {windows} window(s); symbols "
            "lacking history for a given window are excluded from that window only."
        )
        if wide and not single_tier:
            return await runner.validate_tiers(
                universe,
                EXPLORATION_CANDIDATES,
                start,
                chosen[1],
                window_count=windows,
                tiers=tiers,
                symbol_spans=symbol_spans,
                span_start=majority[0],
                span_end=majority[1],
            )
        return await runner.validate(
            universe,
            EXPLORATION_CANDIDATES,
            start,
            chosen[1],
            window_count=windows,
            symbol_spans=symbol_spans,
            span_start=majority[0],
            span_end=majority[1],
        )

    with _quiet_import_logs(quiet):
        report = asyncio.run(_run())
    if report is None:
        raise typer.Exit(code=1)
    if isinstance(report, TieredValidationReport):
        typer.echo(render_tiered_validation(report))
    else:
        typer.echo(render_validation(report))


@app.command()
def diagnose(
    from_: str = typer.Option(..., "--from", help="Window start (YYYY-MM-DD)."),
    to: str = typer.Option(..., "--to", help="Window end (YYYY-MM-DD)."),
    windows: int = typer.Option(1, help="Split the range into N contiguous windows."),
    symbol: list[str] = typer.Option(
        [], "--symbol", "-s", help="Symbols to audit (default: RELIANCE, NIFTY)."
    ),
    jump_threshold: float = typer.Option(
        25.0, help="Overnight %% move flagged as a suspected split/bonus."
    ),
    exchange: Exchange = typer.Option(Exchange.NSE, help="Listing exchange."),
    interval: Interval = typer.Option(Interval.ONE_DAY, help="Candle interval."),
) -> None:
    """Audit stored candles and the buy_and_hold return math (data integrity)."""
    from datetime import date, datetime

    from app.backtest.baselines.dependencies import build_baseline_engine
    from app.backtest.baselines.models import BaselineName
    from app.backtest.dependencies import load_backtest_history
    from app.backtest.diagnostics import ReturnCheck, audit_candles
    from app.backtest.models import BacktestConfig
    from app.backtest.validation import split_windows
    from app.cli.render import render_returns_diagnosis
    from app.core.timezone import INDIA_TZ
    from app.market.historical.models import SeriesKey

    services = _resolve_services()
    configure_logging(services.settings)
    symbols = list(symbol) if symbol else ["RELIANCE", "NIFTY"]
    config = BacktestConfig(exchange=exchange, interval=interval)
    start, end = date.fromisoformat(from_), date.fromisoformat(to)
    windows_spec = split_windows(start, end, windows)

    async def _run() -> dict[str, list[ReturnCheck]]:
        try:
            summaries = await load_backtest_history(
                services, symbols, start, end, config
            )
            _report_adjustments(summaries)
        except Exception as exc:  # noqa: BLE001 - best-effort; the audit reports gaps
            logger.warning("Diagnose history load failed: %s", exc)
        results: dict[str, list[ReturnCheck]] = {}
        for name in symbols:
            key = SeriesKey(symbol=name, exchange=exchange, interval=interval)
            checks: list[ReturnCheck] = []
            for window in windows_spec:
                lo = datetime(
                    window.start.year,
                    window.start.month,
                    window.start.day,
                    tzinfo=INDIA_TZ,
                )
                hi = datetime(
                    window.end.year,
                    window.end.month,
                    window.end.day,
                    23,
                    59,
                    59,
                    tzinfo=INDIA_TZ,
                )
                candles = await services.data_engine.get_candles(key, lo, hi)
                audit = audit_candles(
                    name,
                    candles,
                    window.start,
                    window.end,
                    interval=interval,
                    jump_threshold_pct=jump_threshold,
                )
                engine = build_baseline_engine(
                    services, BaselineName.BUY_AND_HOLD, config=config
                )
                report = await engine.run([name], window.start, window.end)
                checks.append(
                    ReturnCheck(
                        window_label=window.label,
                        audit=audit,
                        engine_return_pct=report.total_return_pct,
                    )
                )
            results[name] = checks
        return results

    results = asyncio.run(_run())
    for name, checks in results.items():
        typer.echo(render_returns_diagnosis(name, checks))
        typer.echo("")


@app.command()
def serve() -> None:
    """Start the FastAPI server (Uvicorn)."""
    from app.main import run as run_server

    run_server()


if __name__ == "__main__":
    app()
