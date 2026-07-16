"""Value objects for the backtesting engine.

These describe a *simulated* walk-forward run: hypothetical trades filled at the
next day's open, charged realistic costs, and the expectancy metrics derived
from them. No real orders are ever placed. Risk sizing reuses the paper book's
:class:`PaperConfig`, and exits reuse the paper :class:`ExitReason` vocabulary,
so the backtest and the live paper book speak the same language.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.market.enums import Exchange, Interval
from app.paper.models import ExitReason, PaperConfig


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
    entry_price: float = Field(gt=0, description="Fill price, incl. entry slippage.")
    exit_price: float = Field(gt=0, description="Exit fill price, incl. slippage.")
    stop_price: float = Field(gt=0, description="Stop level at entry.")
    target_price: float = Field(gt=0, description="Target level at entry.")
    size: float = Field(gt=0, description="Position size (shares).")
    exit_reason: ExitReason = Field(description="Why the trade closed.")
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
    generated_at: datetime = Field(description="Deterministic report timestamp.")

    @model_validator(mode="after")
    def _validate_range(self) -> BacktestReport:
        """Ensure the replay window is not inverted."""
        if self.end < self.start:
            raise ValueError("Backtest 'end' must not precede 'start'.")
        return self
