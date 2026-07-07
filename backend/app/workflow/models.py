"""Workflow request, step and result models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.market.enums import Exchange, Interval
from app.market.historical.importer.models import ImportMode
from app.market.regime.models import RegimeReport
from app.market.sector.models import SectorReport
from app.scanner.models import ScannerResult

#: Default indicators refreshed by the morning workflow.
_DEFAULT_INDICATORS = ("sma", "ema", "rsi", "macd", "atr")


class WorkflowRequest(BaseModel):
    """Parameters shared by all workflows."""

    model_config = ConfigDict(frozen=True)

    symbols: tuple[str, ...] = Field(default=(), description="Symbols to operate on.")
    exchange: Exchange = Field(default=Exchange.NSE, description="Listing exchange.")
    interval: Interval = Field(default=Interval.ONE_DAY, description="Candle interval.")
    start: datetime | None = Field(default=None, description="Explicit range start.")
    end: datetime | None = Field(default=None, description="Explicit range end.")
    history_days: int = Field(
        default=60, ge=1, description="Days of history when no range is given."
    )
    import_mode: ImportMode = Field(
        default=ImportMode.INCREMENTAL, description="Import mode."
    )
    indicators: tuple[str, ...] = Field(
        default=_DEFAULT_INDICATORS, description="Indicators to refresh."
    )
    scanners: tuple[str, ...] = Field(
        default=(), description="Scanner names to run (empty = all)."
    )
    collect_cycles: int = Field(
        default=1, ge=1, description="Poll cycles for the collect workflow."
    )
    top_n: int = Field(default=20, ge=1, description="How many ranked results to keep.")

    @field_validator("symbols")
    @classmethod
    def _normalize(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Trim, upper-case and de-duplicate symbols preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for raw in value:
            symbol = raw.strip().upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                result.append(symbol)
        return tuple(result)


class StepResult(BaseModel):
    """The outcome of a single workflow step."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Step name.")
    ok: bool = Field(description="Whether the step succeeded.")
    duration_seconds: float = Field(ge=0, description="Step wall time.")
    detail: str | None = Field(default=None, description="Optional success detail.")
    error: str | None = Field(default=None, description="Error message on failure.")


class WorkflowReport(BaseModel):
    """Base class for workflow-specific reports."""

    model_config = ConfigDict(frozen=True)


class MorningReport(WorkflowReport):
    """The morning workflow's terminal report payload."""

    market_open: bool = Field(description="Whether the market is currently open.")
    market_state: str = Field(description="Current market state.")
    stocks_scanned: int = Field(ge=0, description="Number of symbols scanned.")
    imported_symbols: int = Field(
        default=0, ge=0, description="Symbols whose history imported cleanly."
    )
    failed_symbols: int = Field(
        default=0, ge=0, description="Symbols whose history import failed."
    )
    failed_symbol_names: tuple[str, ...] = Field(
        default_factory=tuple, description="Names of symbols that failed to import."
    )
    indicators_refreshed: int = Field(ge=0, description="Indicator computations run.")
    regime: RegimeReport | None = Field(
        default=None, description="Market-regime classification, if produced."
    )
    sectors: SectorReport | None = Field(
        default=None, description="Sector-strength ranking, if produced."
    )
    scanner_summary: dict[str, int] = Field(
        default_factory=dict, description="Candidate count per scanner."
    )
    top_results: tuple[ScannerResult, ...] = Field(
        default_factory=tuple, description="Top ranked candidates."
    )
    generated_at: datetime = Field(description="When the report was produced.")


class ScanReport(WorkflowReport):
    """A ranked scan report."""

    stocks_scanned: int = Field(ge=0)
    results: tuple[ScannerResult, ...] = Field(default_factory=tuple)


class RegimeWorkflowReport(WorkflowReport):
    """Standalone market-regime report for the ``regime`` workflow."""

    regime: RegimeReport = Field(description="The market-regime classification.")


class SectorWorkflowReport(WorkflowReport):
    """Standalone sector-strength report for the ``sectors`` workflow."""

    sectors: SectorReport = Field(description="The sector-strength ranking.")


class ImportReport(WorkflowReport):
    """A historical-import report."""

    symbols_processed: int = Field(ge=0)
    candles_imported: int = Field(ge=0)
    failed_requests: int = Field(ge=0)


class IndicatorsReport(WorkflowReport):
    """An indicator-refresh report."""

    computed: int = Field(ge=0)
    skipped: int = Field(ge=0)


class CollectReport(WorkflowReport):
    """A live-collection report."""

    cycles: int = Field(ge=0)
    successful_updates: int = Field(ge=0)
    failed_updates: int = Field(ge=0)


@dataclass(frozen=True)
class WorkflowRun:
    """The full outcome of running a workflow.

    A dataclass (not a Pydantic model) so the polymorphic ``report`` keeps its
    concrete subclass rather than being coerced to the base type.
    """

    workflow: str
    success: bool
    elapsed_seconds: float
    steps: tuple[StepResult, ...]
    error: str | None
    report: WorkflowReport | None
