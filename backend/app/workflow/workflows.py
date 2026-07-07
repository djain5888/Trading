"""Concrete workflows.

Each workflow coordinates existing engines through the context's step runner
and returns a typed report. Workflows hold no business logic of their own.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import ClassVar

from app.market.historical.models import SeriesKey
from app.market.live.models import CollectorConfig
from app.market.regime.models import RegimeReport
from app.market.relative.models import NEUTRAL_RS, RSReport
from app.market.sector.models import SectorReport
from app.scanner.engine import ScannerEngine
from app.scanner.models import ScannerResult
from app.workflow.base import Workflow
from app.workflow.context import WorkflowContext
from app.workflow.models import (
    CollectReport,
    ImportReport,
    IndicatorsReport,
    MorningReport,
    RegimeWorkflowReport,
    RSWorkflowReport,
    ScanReport,
    SectorWorkflowReport,
)
from app.workflow.services import WorkflowServices


def _resolve_range(
    services: WorkflowServices,
    start: datetime | None,
    end: datetime | None,
    history_days: int,
) -> tuple[datetime, datetime]:
    """Return an explicit ``(start, end)`` range, defaulting via the clock."""
    resolved_end = end or services.clock.now()
    resolved_start = start or resolved_end - timedelta(days=history_days)
    return resolved_start, resolved_end


class MorningWorkflow(Workflow):
    """The daily pre-market pipeline: calendar → import → indicators → scan."""

    name: ClassVar[str] = "morning"

    async def run(self, context: WorkflowContext) -> MorningReport:
        """Run the full morning pipeline and produce a terminal report."""
        services = context.services
        request = context.request
        start, end = _resolve_range(
            services, request.start, request.end, request.history_days
        )
        keys = [
            SeriesKey(
                symbol=symbol, exchange=request.exchange, interval=request.interval
            )
            for symbol in request.symbols
        ]

        state = await context.run_step(
            "Check market state", self._market_state(context)
        )
        imported = await context.run_step(
            "Update historical candles", self._import(context, start, end)
        )
        refreshed = await context.run_step(
            "Update indicators", self._indicators(context, keys, start, end)
        )
        regime = await context.run_step(
            "Assess market regime", self._regime(context, end)
        )
        sectors = await context.run_step(
            "Rank sector strength", self._sectors(context, end)
        )
        relative = await context.run_step(
            "Score relative strength", self._relative(context, end)
        )
        raw = await context.run_step(
            "Run scanners", self._scan(context, keys, start, end)
        )
        ranked = await context.run_step(
            "Rank results", self._rank(raw, request.top_n, relative)
        )
        return await context.run_step(
            "Generate report",
            self._report(
                context,
                state,
                len(keys),
                imported,
                refreshed,
                regime,
                sectors,
                relative,
                raw,
                ranked,
            ),
        )

    async def _report(
        self,
        context: WorkflowContext,
        state: tuple[bool, str],
        stocks_scanned: int,
        imported: tuple[list[str], list[str]],
        refreshed: int,
        regime: RegimeReport,
        sectors: SectorReport,
        relative: RSReport,
        raw: list[ScannerResult],
        ranked: list[ScannerResult],
    ) -> MorningReport:
        summary: dict[str, int] = {}
        for result in raw:
            summary[result.scanner_name] = summary.get(result.scanner_name, 0) + 1
        imported_symbols, failed_symbols = imported
        return MorningReport(
            market_open=state[0],
            market_state=state[1],
            stocks_scanned=stocks_scanned,
            imported_symbols=len(imported_symbols),
            failed_symbols=len(failed_symbols),
            failed_symbol_names=tuple(failed_symbols),
            indicators_refreshed=refreshed,
            regime=regime,
            sectors=sectors,
            relative=relative,
            scanner_summary=summary,
            top_results=tuple(ranked),
            generated_at=context.services.clock.now(),
        )

    async def _regime(self, context: WorkflowContext, end: datetime) -> RegimeReport:
        """Classify the market regime; degrades gracefully if data is missing."""
        request = context.request
        return await context.services.regime_engine.analyze(
            request.symbols,
            exchange=request.exchange,
            interval=request.interval,
            end=end,
        )

    async def _sectors(self, context: WorkflowContext, end: datetime) -> SectorReport:
        """Rank sector strength; degrades gracefully without metadata."""
        request = context.request
        return await context.services.sector_engine.analyze(
            request.symbols,
            exchange=request.exchange,
            interval=request.interval,
            end=end,
        )

    async def _relative(self, context: WorkflowContext, end: datetime) -> RSReport:
        """Score relative strength; degrades to neutral without index/sector."""
        request = context.request
        return await context.services.relative_engine.analyze(
            request.symbols,
            exchange=request.exchange,
            interval=request.interval,
            end=end,
        )

    async def _market_state(self, context: WorkflowContext) -> tuple[bool, str]:
        services = context.services
        exchange = context.request.exchange
        state = services.calendar.get_market_state(exchange)
        is_open = services.calendar.is_market_open(exchange)
        return is_open, state.value

    async def _import(
        self, context: WorkflowContext, start: datetime, end: datetime
    ) -> tuple[list[str], list[str]]:
        """Import each symbol independently, returning (imported, failed).

        Symbols are imported one at a time so a per-symbol failure is isolated
        and named. An authentication failure is not per-symbol recoverable and
        propagates, aborting the workflow.
        """
        request = context.request
        engine = context.services.import_engine
        imported: list[str] = []
        failed: list[str] = []
        for symbol in request.symbols:
            summary = await engine.import_history(
                [symbol],
                request.interval,
                start,
                end,
                request.exchange,
                mode=request.import_mode,
            )
            if summary.failed_requests > 0:
                failed.append(symbol)
            else:
                imported.append(symbol)
        return imported, failed

    async def _indicators(
        self,
        context: WorkflowContext,
        keys: Sequence[SeriesKey],
        start: datetime,
        end: datetime,
    ) -> int:
        engine = context.services.indicator_engine
        computed = 0
        for key in keys:
            for indicator in context.request.indicators:
                try:
                    await engine.compute(key, indicator, start, end)
                    computed += 1
                except Exception:  # noqa: BLE001 - best-effort warm-up per indicator
                    continue
        return computed

    async def _scan(
        self,
        context: WorkflowContext,
        keys: Sequence[SeriesKey],
        start: datetime,
        end: datetime,
    ) -> list[ScannerResult]:
        engine = context.services.scanner_engine
        names = context.request.scanners
        scanners = (
            [engine.registry.create(name) for name in names]
            if names
            else engine.registry.create_all()
        )
        return await engine.scan(scanners, keys, start, end, dedupe=False)

    async def _rank(
        self, raw: list[ScannerResult], top_n: int, relative: RSReport
    ) -> list[ScannerResult]:
        """Rank scanner results, using relative strength as a ranking input.

        Scanner logic is untouched: the deduped scanner ranking is re-ordered so
        leaders float up and a symbol's composite relative strength blends into
        the score. Symbols without an RS score fall back to neutral.
        """
        ranked = ScannerEngine.rank(raw, dedupe=True)
        if not relative.available:
            return ranked[:top_n]
        by_symbol = relative.by_symbol

        def key(result: ScannerResult) -> tuple[int, float, float]:
            entry = by_symbol.get(result.symbol)
            composite = entry.composite if entry is not None else NEUTRAL_RS
            leader = 1 if entry is not None and entry.leader else 0
            blended = 0.6 * result.score + 0.4 * composite
            return leader, blended, result.confidence

        return sorted(ranked, key=key, reverse=True)[:top_n]


class ImportWorkflow(Workflow):
    """Updates historical candles for the requested symbols."""

    name: ClassVar[str] = "import"

    async def run(self, context: WorkflowContext) -> ImportReport:
        """Run a historical import and report the summary."""
        services = context.services
        request = context.request
        start, end = _resolve_range(
            services, request.start, request.end, request.history_days
        )
        summary = await context.run_step(
            "Import historical candles",
            services.import_engine.import_history(
                list(request.symbols),
                request.interval,
                start,
                end,
                request.exchange,
                mode=request.import_mode,
            ),
        )
        return ImportReport(
            symbols_processed=summary.symbols_processed,
            candles_imported=summary.candles_imported,
            failed_requests=summary.failed_requests,
        )


class IndicatorsWorkflow(Workflow):
    """Refreshes indicators for the requested symbols."""

    name: ClassVar[str] = "indicators"

    async def run(self, context: WorkflowContext) -> IndicatorsReport:
        """Compute the requested indicators for every symbol."""
        services = context.services
        request = context.request
        start, end = _resolve_range(
            services, request.start, request.end, request.history_days
        )
        keys = [
            SeriesKey(
                symbol=symbol, exchange=request.exchange, interval=request.interval
            )
            for symbol in request.symbols
        ]

        async def _compute() -> tuple[int, int]:
            computed = 0
            skipped = 0
            for key in keys:
                for indicator in request.indicators:
                    try:
                        await services.indicator_engine.compute(
                            key, indicator, start, end
                        )
                        computed += 1
                    except Exception:  # noqa: BLE001 - report as skipped
                        skipped += 1
            return computed, skipped

        computed, skipped = await context.run_step("Compute indicators", _compute())
        return IndicatorsReport(computed=computed, skipped=skipped)


class ScanWorkflow(Workflow):
    """Runs scanners and ranks the results."""

    name: ClassVar[str] = "scan"

    async def run(self, context: WorkflowContext) -> ScanReport:
        """Run the requested scanners and rank candidates."""
        services = context.services
        request = context.request
        start, end = _resolve_range(
            services, request.start, request.end, request.history_days
        )
        keys = [
            SeriesKey(
                symbol=symbol, exchange=request.exchange, interval=request.interval
            )
            for symbol in request.symbols
        ]
        engine = services.scanner_engine
        scanners = (
            [engine.registry.create(name) for name in request.scanners]
            if request.scanners
            else engine.registry.create_all()
        )
        results = await context.run_step(
            "Run scanners", engine.scan(scanners, keys, start, end, dedupe=True)
        )
        return ScanReport(
            stocks_scanned=len(keys), results=tuple(results[: request.top_n])
        )


class RegimeWorkflow(Workflow):
    """Classifies the current market regime from the index and watchlist."""

    name: ClassVar[str] = "regime"

    async def run(self, context: WorkflowContext) -> RegimeWorkflowReport:
        """Assess the market regime and wrap it in a workflow report."""
        services = context.services
        request = context.request
        _, end = _resolve_range(
            services, request.start, request.end, request.history_days
        )
        report = await context.run_step(
            "Assess market regime",
            services.regime_engine.analyze(
                request.symbols,
                exchange=request.exchange,
                interval=request.interval,
                end=end,
            ),
        )
        return RegimeWorkflowReport(regime=report)


class SectorWorkflow(Workflow):
    """Ranks sector strength across the watchlist."""

    name: ClassVar[str] = "sectors"

    async def run(self, context: WorkflowContext) -> SectorWorkflowReport:
        """Rank sector strength and wrap it in a workflow report."""
        services = context.services
        request = context.request
        _, end = _resolve_range(
            services, request.start, request.end, request.history_days
        )
        report = await context.run_step(
            "Rank sector strength",
            services.sector_engine.analyze(
                request.symbols,
                exchange=request.exchange,
                interval=request.interval,
                end=end,
            ),
        )
        return SectorWorkflowReport(sectors=report)


class RelativeStrengthWorkflow(Workflow):
    """Scores relative strength across the watchlist."""

    name: ClassVar[str] = "rs"

    async def run(self, context: WorkflowContext) -> RSWorkflowReport:
        """Score relative strength and wrap it in a workflow report."""
        services = context.services
        request = context.request
        _, end = _resolve_range(
            services, request.start, request.end, request.history_days
        )
        report = await context.run_step(
            "Score relative strength",
            services.relative_engine.analyze(
                request.symbols,
                exchange=request.exchange,
                interval=request.interval,
                end=end,
            ),
        )
        return RSWorkflowReport(relative=report)


class CollectWorkflow(Workflow):
    """Runs a bounded number of live collection cycles."""

    name: ClassVar[str] = "collect"

    async def run(self, context: WorkflowContext) -> CollectReport:
        """Poll the market for the configured number of cycles."""
        services = context.services
        request = context.request
        config = CollectorConfig(
            watchlist=request.symbols or ("RELIANCE",),
            exchange=request.exchange,
            interval=request.interval,
        )
        collector = services.collector_factory(config)

        async def _poll() -> None:
            for _ in range(request.collect_cycles):
                await collector.poll_once()

        await context.run_step("Collect live snapshots", _poll())
        metrics = collector.metrics()
        return CollectReport(
            cycles=request.collect_cycles,
            successful_updates=metrics.successful_updates,
            failed_updates=metrics.failed_updates,
        )
