"""Walk-forward validation: run strategies over non-overlapping windows.

The point of this module is statistical discipline, not more backtests. It
splits the full available history into non-overlapping windows, runs each
strategy in each window against a ``buy_and_hold`` benchmark, flags any window
with too few trades as NOT EVALUABLE, and passes a strategy only if it beats the
benchmark in the *majority* of windows — so a single lucky window cannot promote
noise.
"""

from __future__ import annotations

import gc
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict, Field

from app.backtest import execution
from app.backtest.baselines.dependencies import create_baseline
from app.backtest.baselines.engine import BaselineEngine
from app.backtest.baselines.models import BaselineConfig, BaselineName
from app.backtest.guard import LookaheadGuard
from app.backtest.models import BacktestConfig, BacktestReport
from app.config.watchlist import CAP_TIERS
from app.core.logging import get_logger
from app.indicators.engine import IndicatorEngine
from app.market.calendar.service import MarketCalendarService
from app.market.historical.engine import HistoricalDataEngine

logger = get_logger(__name__)

#: A window with fewer trades than this is statistically NOT EVALUABLE.
DEFAULT_MIN_TRADES = 30
#: Approximate calendar days per month, for translating a window span to days.
_DAYS_PER_MONTH = 30
#: Plausible band (%) for an unleveraged buy-and-hold window return. A value
#: outside this is almost certainly a data problem (unadjusted splits), not a
#: real move, so the runner logs it loudly.
PLAUSIBLE_RETURN_BAND = (-60.0, 150.0)


def is_plausible_return(return_pct: float) -> bool:
    """Return whether a benchmark window return sits in the plausible band."""
    low, high = PLAUSIBLE_RETURN_BAND
    return low <= return_pct <= high


@dataclass(frozen=True, slots=True)
class _WindowMetrics:
    """The only figures validation keeps from a per-window backtest report."""

    trades: int
    expectancy_r: float
    total_return_pct: float
    max_drawdown_pct: float


class HistoryCheck(BaseModel):
    """Whether the confirmed available history can support the requested span."""

    model_config = ConfigDict(frozen=True)

    ok: bool = Field(description="Whether there is enough confirmed history.")
    confirmed_start: date = Field(description="Earliest confirmed available date.")
    confirmed_end: date = Field(description="Latest confirmed available date.")
    usable_start: date = Field(description="First usable date after the lookback.")
    usable_days: int = Field(description="Usable calendar days after the lookback.")
    required_days: int = Field(description="Calendar days the request needs.")
    message: str = Field(description="Human-readable explanation.")


def check_history(
    confirmed_start: date,
    confirmed_end: date,
    *,
    lookback_days: int,
    windows: int,
    window_months: int,
) -> HistoryCheck:
    """Confirm the available history can cover ``windows`` x ``window_months``.

    The lookback the strategies need is reserved before the first window, so the
    usable span is what remains. A request that exceeds the confirmed history is
    refused rather than silently run on fabricated or too-short data.
    """
    usable_start = confirmed_start + timedelta(days=lookback_days)
    usable_days = max(0, (confirmed_end - usable_start).days)
    required_days = windows * window_months * _DAYS_PER_MONTH
    ok = required_days > 0 and usable_days >= required_days
    if ok:
        message = (
            f"Confirmed history {confirmed_start}..{confirmed_end}: "
            f"{usable_days} usable days cover {windows} x {window_months}-month "
            "windows."
        )
    else:
        message = (
            f"Refusing to validate: confirmed available history is "
            f"{confirmed_start}..{confirmed_end} ({usable_days} usable days after a "
            f"{lookback_days}-day lookback), but {windows} x {window_months}-month "
            f"windows need ~{required_days} days. Import more history, or request a "
            "shorter span."
        )
    return HistoryCheck(
        ok=ok,
        confirmed_start=confirmed_start,
        confirmed_end=confirmed_end,
        usable_start=usable_start,
        usable_days=usable_days,
        required_days=required_days,
        message=message,
    )


class WindowSpec(BaseModel):
    """One non-overlapping validation window."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="Zero-based window index.")
    start: date = Field(description="Inclusive window start.")
    end: date = Field(description="Inclusive window end.")
    label: str = Field(description="Human-readable window label.")


class WindowEligibility(BaseModel):
    """Which symbols had sufficient history for one window, and which did not."""

    model_config = ConfigDict(frozen=True)

    window: int = Field(ge=0, description="Window index.")
    label: str = Field(description="Human-readable window label.")
    eligible: tuple[str, ...] = Field(description="Symbols with enough history.")
    excluded: tuple[str, ...] = Field(
        description="Symbols excluded (IPO'd late / insufficient history)."
    )

    @property
    def eligible_count(self) -> int:
        """Return the number of eligible symbols for this window."""
        return len(self.eligible)


class ValidationCell(BaseModel):
    """One strategy's result in one window, versus the benchmark."""

    model_config = ConfigDict(frozen=True)

    strategy: str = Field(description="Strategy name.")
    window: int = Field(ge=0, description="Window index.")
    trades: int = Field(ge=0, description="Closed trades in the window.")
    expectancy_r: float = Field(description="Average R per trade.")
    total_return_pct: float = Field(description="Net return in the window (%).")
    benchmark_return_pct: float = Field(description="Buy-and-hold return (%).")
    max_drawdown_pct: float = Field(ge=0, description="Deepest equity drop (%).")
    evaluable: bool = Field(description="Whether trade count met the minimum.")
    beats_benchmark: bool = Field(description="Evaluable and beat buy-and-hold.")


class StrategyVerdict(BaseModel):
    """The majority-test verdict for one strategy across all windows."""

    model_config = ConfigDict(frozen=True)

    strategy: str = Field(description="Strategy name.")
    windows: int = Field(ge=0, description="Total windows tested.")
    evaluable_windows: int = Field(ge=0, description="Windows with enough trades.")
    beats_count: int = Field(ge=0, description="Windows that beat the benchmark.")
    passed: bool = Field(description="Beat the benchmark in a majority of windows.")
    note: str = Field(description="Human-readable verdict.")


class ValidationReport(BaseModel):
    """The full strategy-by-window matrix plus the per-strategy verdicts."""

    model_config = ConfigDict(frozen=True)

    span_start: date = Field(description="First stored date available.")
    span_end: date = Field(description="Last stored date available.")
    start: date = Field(description="First validated date (after the lookback).")
    end: date = Field(description="Last validated date.")
    benchmark: str = Field(description="Benchmark strategy name.")
    min_trades: int = Field(ge=1, description="Minimum trades for a window to count.")
    windows: tuple[WindowSpec, ...] = Field(description="The validation windows.")
    cells: tuple[ValidationCell, ...] = Field(description="Strategy x window results.")
    verdicts: tuple[StrategyVerdict, ...] = Field(description="Per-strategy verdicts.")
    eligibility: tuple[WindowEligibility, ...] = Field(
        default=(), description="Per-window symbol eligibility (late-IPO exclusions)."
    )

    @property
    def passed(self) -> tuple[str, ...]:
        """Return the names of strategies that passed the majority test."""
        return tuple(v.strategy for v in self.verdicts if v.passed)


def split_windows(start: date, end: date, count: int) -> list[WindowSpec]:
    """Split ``[start, end]`` into ``count`` non-overlapping, contiguous windows.

    Raises:
        ValueError: If the range is empty or ``count`` is not positive.
    """
    if count < 1:
        raise ValueError("Window count must be positive.")
    total_days = (end - start).days
    if total_days < count:
        raise ValueError("Range too short to split into the requested windows.")
    span = total_days // count
    windows: list[WindowSpec] = []
    for index in range(count):
        w_start = start + timedelta(days=index * span)
        w_end = end if index == count - 1 else w_start + timedelta(days=span - 1)
        windows.append(
            WindowSpec(
                index=index,
                start=w_start,
                end=w_end,
                label=f"W{index + 1} {w_start:%Y-%m}..{w_end:%Y-%m}",
            )
        )
    return windows


#: A per-symbol stored span as returned by :func:`execution.stored_spans`.
SymbolSpan = tuple[str, date, date, int]


def majority_span(spans: Sequence[SymbolSpan]) -> tuple[date, date] | None:
    """Return the date span a *majority* of symbols support, or ``None``.

    Unlike the strict intersection (which a single late-IPO symbol collapses),
    this takes the median first-date and median last-date: at least half the
    symbols have data covering ``[start, end]``. This is the span to lay windows
    over — late-IPO symbols are excluded per-window, not from the whole run.
    """
    if not spans:
        return None
    firsts = sorted(span[1] for span in spans)
    lasts = sorted(span[2] for span in spans)
    count = len(spans)
    # start: a majority start on or before the median first-date.
    start = firsts[count // 2]
    # end: a majority have data through at least this date.
    end = lasts[(count - 1) // 2]
    return start, end


def longest_span(spans: Sequence[SymbolSpan]) -> tuple[date, date] | None:
    """Return the widest span any single symbol offers (earliest..latest)."""
    if not spans:
        return None
    return min(span[1] for span in spans), max(span[2] for span in spans)


def eligible_for_window(
    symbols: Sequence[str],
    spans: dict[str, tuple[date, date]],
    window: WindowSpec,
    lookback_days: int,
) -> tuple[list[str], list[str]]:
    """Split ``symbols`` into (eligible, excluded) for one window.

    A symbol is eligible only if its stored history covers
    ``[window.start - lookback_days, window.end]`` — so the strategy has its full
    warm-up before the window and data through the window's end. Symbols that
    IPO'd after that point are excluded from *this* window only.
    """
    need_from = window.start - timedelta(days=lookback_days)
    eligible: list[str] = []
    excluded: list[str] = []
    for symbol in symbols:
        span = spans.get(symbol)
        if span is None or span[0] > need_from or span[1] < window.end:
            excluded.append(symbol)
        else:
            eligible.append(symbol)
    return eligible, excluded


def partition_by_tier(
    symbols: Sequence[str], tiers: dict[str, str]
) -> dict[str, list[str]]:
    """Group ``symbols`` by their market-cap tier, preserving order.

    Symbols with no known tier are dropped. The returned mapping is ordered with
    the recognised tiers (:data:`CAP_TIERS`) first, then any extra tiers seen.
    """
    groups: dict[str, list[str]] = {}
    for symbol in symbols:
        tier = tiers.get(symbol, "").strip().lower()
        if tier:
            groups.setdefault(tier, []).append(symbol)
    ordered: dict[str, list[str]] = {t: groups[t] for t in CAP_TIERS if t in groups}
    for tier, members in groups.items():
        if tier not in ordered:
            ordered[tier] = members
    return ordered


def limit_by_tier(
    symbols: Sequence[str], tiers: dict[str, str], limit: int
) -> list[str]:
    """Truncate ``symbols`` to at most ``limit``, balanced across tiers.

    Selection is round-robin across the tiers so each cap tier keeps roughly the
    same representation. Original order is preserved in the returned list. Any
    untiered symbols backfill only once every tier is exhausted.
    """
    if limit <= 0:
        return []
    if len(symbols) <= limit:
        return list(symbols)
    groups = partition_by_tier(symbols, tiers)
    queues = {tier: list(members) for tier, members in groups.items()}
    chosen: set[str] = set()
    while len(chosen) < limit and any(queues.values()):
        for queue in queues.values():
            if queue and len(chosen) < limit:
                chosen.add(queue.pop(0))
    if len(chosen) < limit:
        for symbol in symbols:
            if symbol not in chosen:
                chosen.add(symbol)
                if len(chosen) >= limit:
                    break
    return [symbol for symbol in symbols if symbol in chosen]


class TierValidation(BaseModel):
    """One market-cap tier's sub-universe and its validation report."""

    model_config = ConfigDict(frozen=True)

    tier: str = Field(description="Market-cap tier label (large/mid/small).")
    symbols: tuple[str, ...] = Field(description="Symbols in this tier.")
    report: ValidationReport = Field(description="Validation over this tier only.")


class TieredValidationReport(BaseModel):
    """The overall validation plus per-tier breakdowns over the same windows."""

    model_config = ConfigDict(frozen=True)

    universe_size: int = Field(ge=0, description="Total symbols in the universe.")
    symbols: tuple[str, ...] = Field(description="The full validated universe.")
    overall: ValidationReport = Field(description="Validation over all symbols.")
    tiers: tuple[TierValidation, ...] = Field(description="Per-tier validations.")


class ValidationRunner:
    """Runs a set of strategies over non-overlapping windows with a benchmark."""

    def __init__(
        self,
        *,
        data_engine: HistoricalDataEngine,
        indicators: IndicatorEngine,
        calendar: MarketCalendarService,
        config: BacktestConfig,
        baseline_config: BaselineConfig,
        min_trades: int = DEFAULT_MIN_TRADES,
        benchmark: BaselineName = BaselineName.BUY_AND_HOLD,
    ) -> None:
        """Initialise the runner with the shared execution collaborators."""
        self._data = data_engine
        self._indicators = indicators
        self._calendar = calendar
        self._config = config
        self._baseline_config = baseline_config
        self._min_trades = min_trades
        self._benchmark = benchmark

    async def stored_range(self, symbols: Sequence[str]) -> tuple[date, date] | None:
        """Return the common stored date span for ``symbols``, or ``None``."""
        return await execution.stored_span(
            LookaheadGuard(self._data),
            symbols,
            self._config.exchange,
            self._config.interval,
        )

    async def _run_one(
        self, name: BaselineName, symbols: Sequence[str], start: date, end: date
    ) -> BacktestReport:
        """Run one strategy over one window with a fresh guard and instance."""
        engine = BaselineEngine(
            data=LookaheadGuard(self._data),
            indicators=self._indicators,
            calendar=self._calendar,
            baseline=create_baseline(name, self._baseline_config),
            config=self._config,
            history_days=self._baseline_config.history_days,
        )
        return await engine.run(symbols, start, end)

    async def _run_metrics(
        self, name: BaselineName, symbols: Sequence[str], start: date, end: date
    ) -> _WindowMetrics:
        """Run one strategy/window and keep ONLY the summary metrics.

        The full :class:`BacktestReport` (with every closed trade, exit analysis
        and leakage decomposition) is discarded when this returns, so peak memory
        stays flat across a many-window, many-tier validation instead of holding
        every window's report at once.
        """
        report = await self._run_one(name, symbols, start, end)
        return _WindowMetrics(
            trades=report.trades,
            expectancy_r=report.expectancy_r,
            total_return_pct=report.total_return_pct,
            max_drawdown_pct=report.max_drawdown_pct,
        )

    async def validate(
        self,
        symbols: Sequence[str],
        strategies: Sequence[BaselineName],
        start: date,
        end: date,
        *,
        window_count: int,
        symbol_spans: dict[str, tuple[date, date]] | None = None,
        span_start: date | None = None,
        span_end: date | None = None,
    ) -> ValidationReport:
        """Build the strategy-by-window matrix and the majority-test verdicts.

        When ``symbol_spans`` is supplied, each window uses only the symbols with
        sufficient history for *that* window (see :func:`eligible_for_window`):
        late-IPO symbols are excluded from early windows rather than from the
        whole run. With ``symbol_spans`` omitted, every symbol is used in every
        window (the original behaviour).
        """
        windows = split_windows(start, end, window_count)
        lookback = self._baseline_config.history_days
        eligibility: list[WindowEligibility] = []
        window_symbols: dict[int, list[str]] = {}
        for window in windows:
            if symbol_spans is None:
                usable, dropped = list(symbols), list[str]()
            else:
                usable, dropped = eligible_for_window(
                    symbols, symbol_spans, window, lookback
                )
            window_symbols[window.index] = usable
            eligibility.append(
                WindowEligibility(
                    window=window.index,
                    label=window.label,
                    eligible=tuple(usable),
                    excluded=tuple(dropped),
                )
            )

        benchmark: dict[int, float] = {}
        for window in windows:
            usable = window_symbols[window.index]
            if not usable:
                benchmark[window.index] = 0.0
                continue
            bench = await self._run_metrics(
                self._benchmark, usable, window.start, window.end
            )
            benchmark[window.index] = bench.total_return_pct
            if not is_plausible_return(bench.total_return_pct):
                logger.error(
                    "IMPLAUSIBLE benchmark return %.1f%% for %s over %s..%s "
                    "(%d symbols) — outside %s. Suspect unadjusted split/bonus "
                    "prices or a sizing/leverage bug; do not trust this window.",
                    bench.total_return_pct,
                    self._benchmark.value,
                    window.start,
                    window.end,
                    len(usable),
                    PLAUSIBLE_RETURN_BAND,
                )

        cells: list[ValidationCell] = []
        for name in strategies:
            for window in windows:
                usable = window_symbols[window.index]
                run = (
                    await self._run_metrics(name, usable, window.start, window.end)
                    if usable
                    else None
                )
                trades = run.trades if run is not None else 0
                evaluable = trades >= self._min_trades
                total_return = run.total_return_pct if run is not None else 0.0
                beats = evaluable and (total_return > benchmark[window.index])
                cells.append(
                    ValidationCell(
                        strategy=name.value,
                        window=window.index,
                        trades=trades,
                        expectancy_r=run.expectancy_r if run is not None else 0.0,
                        total_return_pct=total_return,
                        benchmark_return_pct=benchmark[window.index],
                        max_drawdown_pct=(
                            run.max_drawdown_pct if run is not None else 0.0
                        ),
                        evaluable=evaluable,
                        beats_benchmark=beats,
                    )
                )

        verdicts = tuple(
            build_verdict(name.value, cells, len(windows), self._min_trades)
            for name in strategies
        )
        return ValidationReport(
            span_start=span_start or start,
            span_end=span_end or end,
            start=start,
            end=end,
            benchmark=self._benchmark.value,
            min_trades=self._min_trades,
            windows=tuple(windows),
            cells=tuple(cells),
            verdicts=verdicts,
            eligibility=tuple(eligibility),
        )

    async def validate_tiers(
        self,
        symbols: Sequence[str],
        strategies: Sequence[BaselineName],
        start: date,
        end: date,
        *,
        window_count: int,
        tiers: dict[str, str],
        symbol_spans: dict[str, tuple[date, date]] | None = None,
        span_start: date | None = None,
        span_end: date | None = None,
    ) -> TieredValidationReport:
        """Validate the whole universe, then each cap tier over the same windows.

        The overall run and every per-tier run share the identical ``[start,
        end]`` windows and benchmark, so each tier is compared to its own
        buy-and-hold — making tier-level edge directly comparable. Per-window
        eligibility (``symbol_spans``) applies within each tier too.
        """
        overall = await self.validate(
            symbols,
            strategies,
            start,
            end,
            window_count=window_count,
            symbol_spans=symbol_spans,
            span_start=span_start,
            span_end=span_end,
        )
        groups = partition_by_tier(symbols, tiers)
        tier_results: list[TierValidation] = []
        for tier in CAP_TIERS:
            members = groups.get(tier)
            if not members:
                continue
            report = await self.validate(
                members,
                strategies,
                start,
                end,
                window_count=window_count,
                symbol_spans=symbol_spans,
                span_start=span_start,
                span_end=span_end,
            )
            tier_results.append(
                TierValidation(tier=tier, symbols=tuple(members), report=report)
            )
            # Reclaim the tier's transient backtest objects before the next tier,
            # so peak memory does not grow with the number of tiers on a small box.
            gc.collect()
        return TieredValidationReport(
            universe_size=len(symbols),
            symbols=tuple(symbols),
            overall=overall,
            tiers=tuple(tier_results),
        )


def build_verdict(
    strategy: str,
    cells: Sequence[ValidationCell],
    windows: int,
    min_trades: int,
) -> StrategyVerdict:
    """Pass a strategy only if it beats the benchmark in a strict majority.

    A window where the strategy is NOT EVALUABLE (too few trades) counts as a
    non-beat, so a strategy cannot pass on a single lucky, thinly-traded window.
    """
    mine = [cell for cell in cells if cell.strategy == strategy]
    evaluable = sum(1 for cell in mine if cell.evaluable)
    beats = sum(1 for cell in mine if cell.beats_benchmark)
    passed = beats * 2 > windows  # strict majority of ALL windows
    if evaluable == 0:
        note = f"NOT EVALUABLE — every window had < {min_trades} trades"
    elif passed:
        note = f"PASS — beat buy_and_hold in {beats}/{windows} windows"
    else:
        note = f"FAIL — beat buy_and_hold in only {beats}/{windows} windows"
    return StrategyVerdict(
        strategy=strategy,
        windows=windows,
        evaluable_windows=evaluable,
        beats_count=beats,
        passed=passed,
        note=note,
    )
