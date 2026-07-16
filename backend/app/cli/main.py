"""The Titan command-line interface.

A thin Typer layer over the workflow engine and diagnostics. Commands resolve
services through DI, run a workflow, render the result, and set a non-zero exit
code on failure. They contain no business logic.
"""

from __future__ import annotations

import asyncio

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

app = typer.Typer(help="Titan — AI quant trading platform CLI.", no_args_is_help=True)
logger = get_logger(__name__)


def _resolve_services() -> WorkflowServices:
    """Resolve services through DI (overridable in tests)."""
    return WorkflowServices.resolve()


def _watchlist(symbol: list[str]) -> tuple[str, ...]:
    """Return the given symbols, or the configured default watchlist if none."""
    return tuple(symbol) if symbol else get_watchlist_config().symbols


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
        try:
            # Load candles first (like the morning workflow) so the replay sees
            # the same data; the store is per-process and starts empty.
            await load_backtest_history(services, universe, start, end, config)
        except Exception as exc:  # noqa: BLE001 - best-effort; the engine warns loudly
            logger.warning("Backtest history load failed: %s", exc)
        return await build_backtest_engine(services, config).run(universe, start, end)

    typer.echo(render_backtest(asyncio.run(_run())))


@app.command()
def serve() -> None:
    """Start the FastAPI server (Uvicorn)."""
    from app.main import run as run_server

    run_server()


if __name__ == "__main__":
    app()
