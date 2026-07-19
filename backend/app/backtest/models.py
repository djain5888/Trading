"""Value objects for the backtesting engine.

These describe a *simulated* walk-forward run: hypothetical trades filled at the
next day's open, charged realistic costs, and the expectancy metrics derived
from them. No real orders are ever placed. Risk sizing reuses the paper book's
:class:`PaperConfig`, and exits reuse the paper :class:`ExitReason` vocabulary,
so the backtest and the live paper book speak the same language.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.market.enums import Exchange, Interval
from app.paper.models import ExitReason, PaperConfig


class ExitKind(StrEnum):
    """A fine-grained exit category for diagnostics.

    Distinguishes cases the coarse :class:`ExitReason` conflates: a clean level
    hit vs a gap through it, and a holding-period timeout vs the forced close at
    the end of the backtest window.
    """

    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    TIMEOUT = "timeout"
    GAP_EXIT = "gap_exit"
    END_OF_BACKTEST = "end_of_backtest"


class BacktestCosts(BaseModel):
    """Per-trade friction: slippage on each fill plus percentage charges.

    All values are percentages. ``slippage_pct`` worsens every fill price (a buy
    fills higher, a sell lower); ``brokerage_pct`` and ``stt_pct`` are levied on
    each side's notional (brokerage, Securities Transaction Tax and the like).
    """

    model_config = ConfigDict(frozen=True)

    slippage_pct: float = Field(
        default=0.15, ge=0, description="Slippage applied to each fill price (%)."
    )
    brokerage_pct: float = Field(
        default=0.03, ge=0, description="Brokerage per side, on notional (%)."
    )
    stt_pct: float = Field(
        default=0.10, ge=0, description="STT/other charges per side, on notional (%)."
    )

    @property
    def charge_pct(self) -> float:
        """Percentage charge levied on each side's notional (excludes slippage)."""
        return self.brokerage_pct + self.stt_pct


class BacktestConfig(BaseModel):
    """Configuration for a backtest run: market, risk and cost model."""

    model_config = ConfigDict(frozen=True)

    exchange: Exchange = Field(default=Exchange.NSE, description="Listing exchange.")
    interval: Interval = Field(
        default=Interval.ONE_DAY, description="Candle interval to replay."
    )
    paper: PaperConfig = Field(
        default_factory=PaperConfig,
        description="Portfolio and risk rules (capital, 1% risk, max positions).",
    )
    costs: BacktestCosts = Field(
        default_factory=BacktestCosts, description="Slippage and charge model."
    )


class BacktestTrade(BaseModel):
    """One closed simulated trade, net of costs (immutable record)."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Traded symbol.")
    strategy: str = Field(description="Strategy that produced the setup.")
    regime: str | None = Field(
        default=None, description="Market regime at entry, if classified."
    )
    entry_date: date = Field(description="Fill date (the day after the signal).")
    exit_date: date = Field(description="Exit date.")
    signal_price: float = Field(gt=0, description="Setup reference price at signal.")
    entry_price: float = Field(gt=0, description="Fill price, incl. entry slippage.")
    exit_price: float = Field(gt=0, description="Exit fill price, incl. slippage.")
    stop_price: float = Field(gt=0, description="Stop level at entry.")
    target_price: float = Field(gt=0, description="Target level at entry.")
    size: float = Field(gt=0, description="Position size (shares).")
    exit_reason: ExitReason = Field(description="Why the trade closed (coarse).")
    exit_kind: ExitKind = Field(description="Fine-grained exit category.")
    gross_pnl: float = Field(description="P&L before charges.")
    costs: float = Field(ge=0, description="Charges deducted (brokerage/STT).")
    net_pnl: float = Field(description="P&L after charges.")
    r_multiple: float = Field(description="Net P&L in units of the risk taken.")


class StrategyPerformance(BaseModel):
    """Per-strategy backtest breakdown."""

    model_config = ConfigDict(frozen=True)

    strategy: str = Field(description="Strategy name.")
    trades: int = Field(ge=0, description="Closed trades.")
    wins: int = Field(ge=0, description="Winning trades.")
    win_rate: float = Field(ge=0, le=100, description="Win rate (%).")
    expectancy_r: float = Field(description="Average R per trade.")
    profit_factor: float = Field(ge=0, description="Gross profit / gross loss.")
    total_pnl: float = Field(description="Net P&L for this strategy.")


class RegimePerformance(BaseModel):
    """Backtest results segmented by market regime at entry."""

    model_config = ConfigDict(frozen=True)

    regime: str = Field(description="Regime label (BULL/BEAR/SIDEWAYS/UNKNOWN).")
    trades: int = Field(ge=0, description="Closed trades opened in this regime.")
    win_rate: float = Field(ge=0, le=100, description="Win rate (%).")
    expectancy_r: float = Field(description="Average R per trade.")
    total_pnl: float = Field(description="Net P&L in this regime.")


class MonthlyReturn(BaseModel):
    """Net P&L attributed to the month a trade closed in."""

    model_config = ConfigDict(frozen=True)

    month: str = Field(description="Calendar month, ``YYYY-MM``.")
    trades: int = Field(ge=0, description="Trades closed in the month.")
    net_pnl: float = Field(description="Net P&L realized in the month.")
    return_pct: float = Field(description="Net P&L as a % of starting capital.")


class ExitBreakdown(BaseModel):
    """Count, expectancy and holding time for one exit category."""

    model_config = ConfigDict(frozen=True)

    kind: str = Field(description="Exit category (an ``ExitKind`` value).")
    trades: int = Field(ge=0, description="Trades closed by this exit.")
    wins: int = Field(ge=0, description="Profitable trades in this bucket.")
    avg_r: float = Field(description="Average R multiple for this exit.")
    avg_holding_days: float = Field(
        ge=0, description="Average calendar days held before this exit."
    )
    total_pnl: float = Field(description="Net P&L for this exit category.")


class WinnerRDistribution(BaseModel):
    """How far winners actually ran, in R, versus the 2R target."""

    model_config = ConfigDict(frozen=True)

    winners: int = Field(default=0, ge=0, description="Winning trades (net > 0).")
    reached_2r: int = Field(default=0, ge=0, description="Winners with R >= 2.")
    between_1_and_2r: int = Field(
        default=0, ge=0, description="Winners with 1 <= R < 2."
    )
    between_0_and_1r: int = Field(
        default=0, ge=0, description="Winners with 0 < R < 1."
    )


class ExitAnalysis(BaseModel):
    """Diagnostics that explain where a strategy's expectancy leaks out."""

    model_config = ConfigDict(frozen=True)

    by_reason: tuple[ExitBreakdown, ...] = Field(
        default_factory=tuple, description="Exit breakdown across all trades."
    )
    winner_r: WinnerRDistribution = Field(
        default_factory=WinnerRDistribution, description="R distribution of winners."
    )
    timeout_trades: int = Field(
        default=0, ge=0, description="Trades closed by the holding-period timeout."
    )
    timeout_profitable: int = Field(
        default=0, ge=0, description="Timeout trades profitable at the close."
    )
    avg_entry_slippage_pct: float = Field(
        default=0.0, description="Average entry fill vs signal price (%)."
    )
    avg_entry_slippage: float = Field(
        default=0.0, description="Average entry fill vs signal price (per share)."
    )
    bull_by_reason: tuple[ExitBreakdown, ...] = Field(
        default_factory=tuple, description="Exit breakdown for BULL-regime trades."
    )


class BacktestReport(BaseModel):
    """The measured expectancy of the strategy path over a historical window."""

    model_config = ConfigDict(frozen=True)

    start: date = Field(description="First replayed calendar date.")
    end: date = Field(description="Last replayed calendar date.")
    symbols: tuple[str, ...] = Field(description="Symbols in the universe.")
    starting_capital: float = Field(description="Configured starting capital.")
    ending_equity: float = Field(description="Capital + realized net P&L.")
    total_return_pct: float = Field(description="Net P&L as a % of starting capital.")
    trades: int = Field(ge=0, description="Closed trades.")
    wins: int = Field(ge=0, description="Winning trades.")
    losses: int = Field(ge=0, description="Losing trades.")
    win_rate: float = Field(ge=0, le=100, description="Win rate (%).")
    avg_win: float = Field(description="Average winning trade (net).")
    avg_loss: float = Field(description="Average losing trade (net, <= 0).")
    expectancy_r: float = Field(description="Average R per trade (expectancy).")
    profit_factor: float = Field(ge=0, description="Gross profit / gross loss.")
    max_drawdown_pct: float = Field(
        ge=0, description="Deepest peak-to-trough equity drop (%)."
    )
    longest_losing_streak: int = Field(
        ge=0, description="Longest run of consecutive losing trades."
    )
    total_pnl: float = Field(description="Total net P&L.")
    per_strategy: tuple[StrategyPerformance, ...] = Field(
        default_factory=tuple, description="Per-strategy breakdown."
    )
    per_regime: tuple[RegimePerformance, ...] = Field(
        default_factory=tuple, description="Per-regime breakdown."
    )
    monthly: tuple[MonthlyReturn, ...] = Field(
        default_factory=tuple, description="Monthly net-P&L distribution."
    )
    closed_trades: tuple[BacktestTrade, ...] = Field(
        default_factory=tuple, description="All closed trades, oldest first."
    )
    exit_analysis: ExitAnalysis = Field(
        default_factory=ExitAnalysis, description="Exit-reason diagnostics."
    )
    generated_at: datetime = Field(description="Deterministic report timestamp.")

    @model_validator(mode="after")
    def _validate_range(self) -> BacktestReport:
        """Ensure the replay window is not inverted."""
        if self.end < self.start:
            raise ValueError("Backtest 'end' must not precede 'start'.")
        return self
