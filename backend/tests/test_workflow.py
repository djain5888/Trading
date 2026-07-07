"""Unit tests for the workflow engine and CLI."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from app.cli import main as cli_main
from app.core.clock import FakeClock
from app.core.timezone import INDIA_TZ
from app.indicators.dependencies import get_indicator_registry
from app.indicators.engine import IndicatorEngine
from app.market.calendar.config import InMemoryCalendarConfigProvider
from app.market.calendar.holidays import StaticHolidayProvider
from app.market.calendar.service import MarketCalendarService
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.importer.engine import HistoricalImportEngine
from app.market.historical.importer.models import ImportSummary
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import Candle
from app.market.historical.validation import ValidationEngine
from app.market.live.collector import LiveMarketCollector
from app.market.live.models import CollectorConfig
from app.market.models import (
    HistoricalData,
    MarketDepth,
    MarketStatusInfo,
    Quote,
    SymbolSearchResponse,
)
from app.market.regime.engine import MarketRegimeEngine
from app.market.sector.engine import SectorStrengthEngine
from app.providers.base import MarketDataProvider
from app.scanner.dependencies import get_scanner_registry
from app.scanner.engine import ScannerEngine
from app.workflow.engine import WorkflowEngine, build_workflow_engine
from app.workflow.errors import UnknownWorkflowError
from app.workflow.models import MorningReport, WorkflowRequest
from app.workflow.services import WorkflowServices
from typer.testing import CliRunner

_OPEN = datetime(2025, 1, 6, 10, 0, tzinfo=INDIA_TZ)  # Monday, market open


class _StubProvider(MarketDataProvider):
    """A provider stub (unused by the workflow tests directly)."""

    async def get_market_status(
        self, exchange: Exchange = Exchange.NSE
    ) -> MarketStatusInfo:
        raise NotImplementedError

    async def get_quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Quote:
        raise NotImplementedError

    async def get_quotes(
        self, symbols: Sequence[str], exchange: Exchange = Exchange.NSE
    ) -> list[Quote]:
        return [
            Quote(
                symbol=s,
                exchange=exchange,
                last_price=Decimal("100"),
                volume=100,
                timestamp=_OPEN,
            )
            for s in symbols
        ]

    async def get_historical_data(
        self,
        symbol: str,
        interval: Interval,
        start_date: datetime,
        end_date: datetime,
        exchange: Exchange = Exchange.NSE,
    ) -> HistoricalData:
        raise NotImplementedError

    async def search_symbol(
        self, query: str, exchange: Exchange | None = None
    ) -> SymbolSearchResponse:
        raise NotImplementedError

    async def get_market_depth(
        self, symbol: str, exchange: Exchange = Exchange.NSE
    ) -> MarketDepth:
        raise NotImplementedError


class _FakeImportEngine(HistoricalImportEngine):
    """Import engine that seeds candles instead of calling a provider."""

    def __init__(
        self,
        data_engine: HistoricalDataEngine,
        fail: bool = False,
        fail_symbols: frozenset[str] = frozenset(),
        auth_fail: bool = False,
    ) -> None:
        self._data_engine = data_engine
        self._fail = fail
        self._fail_symbols = fail_symbols
        self._auth_fail = auth_fail

    async def import_history(
        self,
        symbols: Sequence[str],
        interval: Interval,
        start: datetime,
        end: datetime,
        exchange: Exchange = Exchange.NSE,
        *,
        mode: object = None,
        on_progress: object = None,
    ) -> ImportSummary:
        if self._auth_fail:
            from app.providers.exceptions import AuthenticationError

            raise AuthenticationError("Groww rejected credentials.")
        if self._fail:
            raise RuntimeError("provider unavailable")
        if any(symbol in self._fail_symbols for symbol in symbols):
            return ImportSummary(
                symbols_processed=len(symbols), candles_imported=0, failed_requests=1
            )
        imported = 0
        count = 260
        for symbol in symbols:
            candles = [
                Candle(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    timestamp=end - timedelta(minutes=(count - 1 - i)),
                    open=Decimal(f"{100 + i:.4f}"),
                    high=Decimal(f"{100 + i + 0.5:.4f}"),
                    low=Decimal(f"{100 + i - 0.5:.4f}"),
                    close=Decimal(f"{100 + i:.4f}"),
                    volume=1000,
                )
                for i in range(count)
            ]
            report = await self._data_engine.import_candles(candles)
            imported += report.imported
        return ImportSummary(symbols_processed=len(symbols), candles_imported=imported)


def _services(
    *,
    import_fail: bool = False,
    fail_symbols: frozenset[str] = frozenset(),
    auth_fail: bool = False,
) -> WorkflowServices:
    """Build a fully-faked service container."""
    clock = FakeClock(_OPEN)
    repo = InMemoryCandleRepository()
    data_engine = HistoricalDataEngine(repo, ValidationEngine())
    calendar = MarketCalendarService(
        clock=clock,
        config_provider=InMemoryCalendarConfigProvider.with_defaults(),
        holiday_provider=StaticHolidayProvider({}),
    )
    indicator_engine = IndicatorEngine(data_engine, clock, get_indicator_registry())
    scanner_engine = ScannerEngine(
        data_engine, indicator_engine, calendar, clock, get_scanner_registry()
    )
    regime_engine = MarketRegimeEngine(data_engine, indicator_engine, clock)
    sector_engine = SectorStrengthEngine(data_engine, indicator_engine, clock)

    def _collector_factory(config: CollectorConfig) -> LiveMarketCollector:
        return LiveMarketCollector(
            provider=_StubProvider(),
            calendar=calendar,
            clock=clock,
            repository=repo,
            config=config,
        )

    from app.config.settings import Settings

    return WorkflowServices(
        settings=Settings(),
        clock=clock,
        calendar=calendar,
        data_engine=data_engine,
        import_engine=_FakeImportEngine(
            data_engine,
            fail=import_fail,
            fail_symbols=fail_symbols,
            auth_fail=auth_fail,
        ),
        indicator_engine=indicator_engine,
        scanner_engine=scanner_engine,
        regime_engine=regime_engine,
        sector_engine=sector_engine,
        provider=_StubProvider(),
        collector_factory=_collector_factory,
    )


# -- Workflow engine -------------------------------------------------------


async def test_morning_workflow_success() -> None:
    """The morning workflow runs all steps and produces a report."""
    engine = build_workflow_engine(services=_services())
    request = WorkflowRequest(
        symbols=("AAA", "BBB"), interval=Interval.ONE_MINUTE, history_days=1
    )
    run = await engine.run("morning", request)

    assert run.success
    assert run.error is None
    assert len(run.steps) == 8
    assert all(step.ok for step in run.steps)
    assert isinstance(run.report, MorningReport)
    assert run.report.stocks_scanned == 2
    assert run.report.indicators_refreshed > 0
    assert run.report.market_open is True
    assert run.report.imported_symbols == 2
    assert run.report.failed_symbols == 0


async def test_morning_report_surfaces_failed_symbols() -> None:
    """A per-symbol import failure is counted and named in the report."""
    engine = build_workflow_engine(services=_services(fail_symbols=frozenset({"BBB"})))
    request = WorkflowRequest(
        symbols=("AAA", "BBB"), interval=Interval.ONE_MINUTE, history_days=1
    )
    run = await engine.run("morning", request)

    assert run.success
    assert isinstance(run.report, MorningReport)
    assert run.report.imported_symbols == 1
    assert run.report.failed_symbols == 1
    assert run.report.failed_symbol_names == ("BBB",)


async def test_morning_workflow_hard_auth_failure_surfaces() -> None:
    """A hard auth failure aborts the run and is surfaced clearly."""
    engine = build_workflow_engine(services=_services(auth_fail=True))
    request = WorkflowRequest(symbols=("AAA",), interval=Interval.ONE_MINUTE)
    run = await engine.run("morning", request)

    assert run.success is False
    assert run.report is None
    assert run.error is not None
    assert "credential" in run.error.lower() or "auth" in run.error.lower()
    assert run.steps[-1].ok is False


def test_render_morning_shows_import_counts() -> None:
    """The terminal render surfaces imported/failed symbols and names."""
    from app.cli.render import render_run
    from app.workflow.models import StepResult, WorkflowRun

    report = MorningReport(
        market_open=True,
        market_state="open",
        stocks_scanned=2,
        imported_symbols=1,
        failed_symbols=1,
        failed_symbol_names=("BBB",),
        indicators_refreshed=3,
        generated_at=_OPEN,
    )
    run = WorkflowRun(
        workflow="morning",
        success=True,
        elapsed_seconds=0.1,
        steps=(StepResult(name="x", ok=True, duration_seconds=0.0),),
        error=None,
        report=report,
    )
    text = render_run(run)
    assert "Import:              1 ok / 1 failed (BBB)" in text


async def test_morning_workflow_step_failure() -> None:
    """A failing step aborts gracefully with no report and an error."""
    engine = build_workflow_engine(services=_services(import_fail=True))
    request = WorkflowRequest(symbols=("AAA",), interval=Interval.ONE_MINUTE)
    run = await engine.run("morning", request)

    assert run.success is False
    assert run.report is None
    assert run.error is not None
    # The failed step is recorded; steps after it are not.
    assert run.steps[-1].ok is False
    assert run.steps[-1].name == "Update historical candles"


async def test_unknown_workflow() -> None:
    """An unknown workflow name is rejected."""
    engine = build_workflow_engine(services=_services())
    with pytest.raises(UnknownWorkflowError):
        await engine.run("nope", WorkflowRequest())


async def test_scan_workflow() -> None:
    """The scan workflow ranks candidates from stored data."""
    services = _services()
    # Seed candles via the fake import engine first.
    engine = build_workflow_engine(services=services)
    await engine.run(
        "import",
        WorkflowRequest(symbols=("AAA",), interval=Interval.ONE_MINUTE),
    )
    run = await engine.run(
        "scan", WorkflowRequest(symbols=("AAA",), interval=Interval.ONE_MINUTE)
    )
    assert run.success
    from app.workflow.models import ScanReport

    assert isinstance(run.report, ScanReport)
    assert run.report.stocks_scanned == 1


def test_registry_lists_workflows() -> None:
    """All five workflows are registered."""
    engine = build_workflow_engine(services=_services())
    assert set(engine.registry.available()) == {
        "morning",
        "import",
        "indicators",
        "scan",
        "regime",
        "sectors",
        "collect",
    }


def test_services_resolve_uses_di() -> None:
    """WorkflowServices.resolve wires real DI singletons without error."""
    services = WorkflowServices.resolve()
    assert isinstance(services, WorkflowServices)
    assert isinstance(services.scanner_engine, ScannerEngine)
    assert isinstance(services.clock.now(), datetime)


def test_engine_uses_injected_services() -> None:
    """The engine coordinates without constructing services itself."""
    services = _services()
    engine = WorkflowEngine(services=services, clock=services.clock)
    assert engine.registry.available()


# -- CLI -------------------------------------------------------------------


def test_cli_version() -> None:
    """`titan version` prints version information."""
    result = CliRunner().invoke(cli_main.app, ["version"])
    assert result.exit_code == 0
    assert "Version:" in result.stdout
    assert "Python:" in result.stdout


def test_cli_health() -> None:
    """`titan health` prints checks and exits 0 (provider is non-critical)."""
    result = CliRunner().invoke(cli_main.app, ["health"])
    assert result.exit_code == 0
    assert "Configuration" in result.stdout
    assert "Database" in result.stdout


def test_health_provider_surfaces_hard_auth_failure() -> None:
    """A hard auth failure is a critical, unhealthy provider check."""
    from app.config.broker import GrowwSettings
    from app.config.settings import Settings
    from app.providers.exceptions import AuthenticationError
    from app.workflow.diagnostics import run_health_checks

    async def _fail_probe(_groww: GrowwSettings) -> None:
        raise AuthenticationError("Groww rejected credentials.")

    settings = Settings(groww=GrowwSettings(api_key="k", totp_seed="AAAA"))
    checks = run_health_checks(settings, auth_probe=_fail_probe)
    provider = next(check for check in checks if check.name == "Provider")

    assert provider.healthy is False
    assert provider.critical is True
    assert "auth" in provider.detail.lower()


def test_health_provider_reports_authenticated() -> None:
    """A successful auth probe yields a healthy provider check."""
    from app.config.broker import GrowwSettings
    from app.config.settings import Settings
    from app.workflow.diagnostics import run_health_checks

    async def _ok_probe(_groww: GrowwSettings) -> None:
        return None

    settings = Settings(groww=GrowwSettings(api_key="k", totp_seed="AAAA"))
    checks = run_health_checks(settings, auth_probe=_ok_probe)
    provider = next(check for check in checks if check.name == "Provider")

    assert provider.healthy is True
    assert "authenticated" in provider.detail.lower()


def test_cli_morning(monkeypatch: pytest.MonkeyPatch) -> None:
    """`titan morning` runs the workflow through injected services."""
    monkeypatch.setattr(cli_main, "_resolve_services", lambda: _services())
    result = CliRunner().invoke(
        cli_main.app,
        ["morning", "-s", "AAA", "-s", "BBB", "--interval", "1m"],
    )
    assert result.exit_code == 0
    assert "Titan Morning Report" in result.stdout
    assert "Stocks Scanned:" in result.stdout


def test_cli_morning_failure_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """A workflow failure yields a non-zero CLI exit code."""
    monkeypatch.setattr(
        cli_main, "_resolve_services", lambda: _services(import_fail=True)
    )
    result = CliRunner().invoke(
        cli_main.app, ["morning", "-s", "AAA", "--interval", "1m"]
    )
    assert result.exit_code == 1
    assert "FAILED" in result.stdout
