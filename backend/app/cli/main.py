"""The Titan command-line interface.

A thin Typer layer over the workflow engine and diagnostics. Commands resolve
services through DI, run a workflow, render the result, and set a non-zero exit
code on failure. They contain no business logic.
"""

from __future__ import annotations

import asyncio

import typer

from app.cli.render import render_health, render_run, render_version
from app.core.logging import configure_logging
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


def _resolve_services() -> WorkflowServices:
    """Resolve services through DI (overridable in tests)."""
    return WorkflowServices.resolve()


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
    history_days: int = typer.Option(60, help="Days of history to consider."),
) -> None:
    """Run the full pre-market pipeline and print the morning report."""
    request = WorkflowRequest(
        symbols=tuple(symbol),
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
    history_days: int = typer.Option(60, help="Days of history to import."),
) -> None:
    """Update historical candles for the given symbols."""
    request = WorkflowRequest(
        symbols=tuple(symbol),
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
    history_days: int = typer.Option(60, help="Days of history to use."),
) -> None:
    """Refresh indicators for the given symbols."""
    request = WorkflowRequest(
        symbols=tuple(symbol),
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
    history_days: int = typer.Option(60, help="Days of history to use."),
    top: int = typer.Option(20, help="Number of ranked results to show."),
) -> None:
    """Run scanners and print ranked candidates."""
    request = WorkflowRequest(
        symbols=tuple(symbol),
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
        symbols=tuple(symbol),
        exchange=exchange,
        interval=interval,
        history_days=history_days,
    )
    _emit(_run_workflow("regime", request))


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
def serve() -> None:
    """Start the FastAPI server (Uvicorn)."""
    from app.main import run as run_server

    run_server()


if __name__ == "__main__":
    app()
