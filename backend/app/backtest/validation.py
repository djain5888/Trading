"""Walk-forward validation: run strategies over non-overlapping windows.

The point of this module is statistical discipline, not more backtests. It
splits the full available history into non-overlapping windows, runs each
strategy in each window against a ``buy_and_hold`` benchmark, flags any window
with too few trades as NOT EVALUABLE, and passes a strategy only if it beats the
benchmark in the *majority* of windows — so a single lucky window cannot promote
noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

from pydantic import BaseModel, ConfigDict, Field

from app.backtest import execution
from app.backtest.baselines.dependencies import create_baseline
from app.backtest.baselines.engine import BaselineEngine
from app.backtest.baselines.models import BaselineConfig, BaselineName
from app.backtest.guard import LookaheadGuard
from app.backtest.models import BacktestConfig, BacktestReport
from app.indicators.engine import IndicatorEngine
from app.market.calendar.service import MarketCalendarService
from app.market.historical.engine import HistoricalDataEngine

#: A window with fewer trades than this is statistically NOT EVALUABLE.
DEFAULT_MIN_TRADES = 30


class WindowSpec(BaseModel):
    """One non-overlapping validation window."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="Zero-based window index.")
    start: date = Field(description="Inclusive window start.")
    end: date = Field(description="Inclusive window end.")
    label: str = Field(description="Human-readable window label.")


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

    async def validate(
        self,
        symbols: Sequence[str],
        strategies: Sequence[BaselineName],
        start: date,
        end: date,
        *,
        window_count: int,
        span_start: date | None = None,
        span_end: date | None = None,
    ) -> ValidationReport:
        """Build the strategy-by-window matrix and the majority-test verdicts."""
        windows = split_windows(start, end, window_count)
        benchmark: dict[int, float] = {}
        for window in windows:
            report = await self._run_one(
                self._benchmark, symbols, window.start, window.end
            )
            benchmark[window.index] = report.total_return_pct

        cells: list[ValidationCell] = []
        for name in strategies:
            for window in windows:
                report = await self._run_one(name, symbols, window.start, window.end)
                evaluable = report.trades >= self._min_trades
                beats = evaluable and (
                    report.total_return_pct > benchmark[window.index]
                )
                cells.append(
                    ValidationCell(
                        strategy=name.value,
                        window=window.index,
                        trades=report.trades,
                        expectancy_r=report.expectancy_r,
                        total_return_pct=report.total_return_pct,
                        benchmark_return_pct=benchmark[window.index],
                        max_drawdown_pct=report.max_drawdown_pct,
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
