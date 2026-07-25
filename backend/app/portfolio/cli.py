"""``titan portfolio`` commands: import statements and show the measurement.

Network access (equity quotes, AMFI NAV, broker holdings) is best-effort and
cache-backed: a fetch refreshes the local cache, and ``--offline`` (or a failed
fetch) runs entirely from cache, failing loudly only when data is missing or
stale. The measurement engine itself is pure, so the numbers are deterministic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path

import typer

from app.core.logging import configure_logging, get_logger
from app.portfolio.engine import analyse_portfolio
from app.portfolio.importers.base import ImportResult
from app.portfolio.importers.cas_mf import parse_cas_csv
from app.portfolio.importers.groww_equity import parse_groww_equity_csv
from app.portfolio.importers.manual_json import parse_manual_json
from app.portfolio.models import AssetType, load_portfolio_config
from app.portfolio.nav import (
    AMFI_NAVALL_URL,
    NavSnapshot,
    default_cache_path,
    load_navs,
)
from app.portfolio.prices import default_price_cache_path, load_equity_prices
from app.portfolio.render import render_import, render_portfolio
from app.portfolio.store import (
    config_path,
    load_transactions,
    merge_transactions,
    save_transactions,
    transactions_path,
)

logger = get_logger(__name__)

portfolio_app = typer.Typer(help="Portfolio companion: import, value and measure.")


@portfolio_app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """Show the portfolio measurement when no sub-command is given."""
    if ctx.invoked_subcommand is None:
        _show(offline=False)


@portfolio_app.command(name="import")
def import_(
    groww: list[Path] = typer.Option(
        [], "--groww", help="Groww equity transaction CSV(s)."
    ),
    cas: list[Path] = typer.Option([], "--cas", help="CAMS/KFintech CAS CSV(s)."),
    manual: list[Path] = typer.Option([], "--manual", help="Manual JSON file(s)."),
) -> None:
    """Import transaction statements into the canonical store."""
    configure_logging_safe()
    results: list[ImportResult] = []
    for path in groww:
        results.append(parse_groww_equity_csv(path.read_text(encoding="utf-8")))
    for path in cas:
        results.append(parse_cas_csv(path.read_text(encoding="utf-8")))
    for path in manual:
        results.append(parse_manual_json(path.read_text(encoding="utf-8")))

    if not results:
        raise typer.BadParameter("Provide at least one of --groww / --cas / --manual.")

    imported = [txn for result in results for txn in result.transactions]
    existing = load_transactions(transactions_path())
    merged = merge_transactions(existing, imported)
    save_transactions(transactions_path(), merged)
    typer.echo(render_import(results))
    typer.echo(f"\nStored {len(merged)} transactions at {transactions_path()}.")


@portfolio_app.command(name="show")
def show(
    offline: bool = typer.Option(
        False, "--offline", help="Value from cache only (no network fetch)."
    ),
) -> None:
    """Show the portfolio measurement report."""
    _show(offline=offline)


def configure_logging_safe() -> None:
    """Configure logging from settings, tolerating an unconfigured environment."""
    try:
        from app.config.settings import get_settings

        configure_logging(get_settings())
    except Exception as exc:  # noqa: BLE001 - logging must never block a CLI command
        logger.debug("Falling back to default logging: %s", exc)


def _show(*, offline: bool) -> None:
    """Load config + transactions, value the book, and render the report."""
    configure_logging_safe()
    config = load_portfolio_config(config_path())
    transactions = load_transactions(transactions_path())
    valuation_date = date.today()

    equity_symbols = [
        h.identifier for h in config.holdings if h.asset_type is AssetType.EQUITY
    ]
    prices = load_equity_prices(
        default_price_cache_path(),
        equity_symbols,
        on=valuation_date,
        fetcher=None if offline else _equity_fetcher(),
        offline=offline,
    )
    equity_prices: dict[str, Decimal] = {s: p.close for s, p in prices.items()}

    has_mf = any(h.asset_type is AssetType.MF for h in config.holdings)
    nav: NavSnapshot | None = None
    if has_mf:
        nav = load_navs(
            default_cache_path(),
            on=valuation_date,
            fetcher=None if offline else _amfi_fetcher(),
            offline=offline,
        )

    broker_units = None if offline else _broker_holdings()
    report = analyse_portfolio(
        config,
        transactions,
        valuation_date=valuation_date,
        equity_prices=equity_prices,
        nav=nav,
        broker_units=broker_units,
    )
    typer.echo(render_portfolio(report))


def _equity_fetcher() -> Callable[[str], Decimal | None]:
    """Return a callable ``symbol -> Decimal|None`` backed by the market provider."""
    from app.providers.dependencies import get_market_data_provider

    provider = get_market_data_provider()

    def fetch(symbol: str) -> Decimal | None:
        try:
            quote = asyncio.run(provider.get_quote(symbol))
            return Decimal(str(quote.last_price))
        except Exception as exc:  # noqa: BLE001 - cache covers unfetched symbols
            logger.warning("Quote fetch failed for %s: %s", symbol, exc)
            return None

    return fetch


def _amfi_fetcher() -> Callable[[], str]:
    """Return a callable fetching AMFI NAVAll.txt text over HTTPS."""

    def fetch() -> str:
        import httpx

        response = httpx.get(AMFI_NAVALL_URL, timeout=30.0)
        response.raise_for_status()
        return response.text

    return fetch


def _broker_holdings() -> dict[str, Decimal] | None:
    """Attempt groww.get_holdings_for_user(); return units by symbol, or None.

    INVESTIGATION: the growwapi SDK's holdings endpoint requires a trading
    (not just market-data) session and is not part of the market-data provider
    port. When it is unavailable — no trading session, method absent, or any
    error — we return None and the cross-check is simply skipped and reported.
    """
    try:
        from app.providers.dependencies import get_market_data_provider

        provider = get_market_data_provider()
        client = getattr(provider, "_sdk", None)
        holdings_fn = getattr(
            client() if callable(client) else client, "get_holdings_for_user", None
        )
        if holdings_fn is None:
            logger.info("Broker holdings API unavailable; skipping cross-check.")
            return None
        payload = holdings_fn()
        return _parse_broker_holdings(payload)
    except Exception as exc:  # noqa: BLE001 - cross-check is optional
        logger.info("Broker holdings cross-check unavailable: %s", exc)
        return None


def _parse_broker_holdings(payload: object) -> dict[str, Decimal] | None:
    """Best-effort parse of a broker holdings payload into units by symbol."""
    if not isinstance(payload, dict):
        return None
    rows = payload.get("holdings") or payload.get("data") or []
    result: dict[str, Decimal] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        symbol = row.get("trading_symbol") or row.get("symbol")
        quantity = row.get("quantity") or row.get("net_quantity")
        if symbol is None or quantity is None:
            continue
        try:
            result[str(symbol).upper()] = Decimal(str(quantity))
        except (ValueError, ArithmeticError):
            continue
    return result or None
