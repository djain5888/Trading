"""Value objects for the baseline edge search.

Baselines are deliberately simple, fixed-parameter strategies run through the
same execution realism as the main backtest. We are searching for *any* edge,
not tuning knobs, so every parameter here has a sensible fixed default and is
never swept.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.backtest.models import BacktestReport


class BaselineName(StrEnum):
    """The selectable baselines."""

    BUY_AND_HOLD = "buy_and_hold"
    TOP_RS_WEEKLY = "top_rs_weekly"
    TOP_RS_TRAILING = "top_rs_trailing"
    MOMENTUM_12_1 = "momentum_12_1"
    MEAN_REVERT = "mean_revert"
    # -- TASK-024 exploration candidates --
    RS_MOMENTUM_TRAIL = "rs_momentum_trail"
    RS_MOMENTUM_CASH = "rs_momentum_cash"
    DUAL_MOMENTUM = "dual_momentum"
    TREND_FOLLOW = "trend_follow"
    VOLATILITY_BREAK = "volatility_break"
    RS_ROTATION_MONTHLY = "rs_rotation_monthly"


#: The six candidates explored (and validated) in TASK-024.
EXPLORATION_CANDIDATES: tuple[BaselineName, ...] = (
    BaselineName.RS_MOMENTUM_TRAIL,
    BaselineName.RS_MOMENTUM_CASH,
    BaselineName.DUAL_MOMENTUM,
    BaselineName.TREND_FOLLOW,
    BaselineName.VOLATILITY_BREAK,
    BaselineName.RS_ROTATION_MONTHLY,
)


class BaselineConfig(BaseModel):
    """Fixed, sensible baseline parameters (never swept)."""

    model_config = ConfigDict(frozen=True)

    top_n: int = Field(
        default=3, ge=1, description="Positions held by top-N baselines."
    )
    hold_days: int = Field(
        default=20, ge=1, description="Fixed holding period for TOP_RS_WEEKLY."
    )
    history_days: int = Field(
        default=420, ge=1, description="Calendar days of history each plan reads."
    )
    rs_lookback: int = Field(
        default=63, ge=1, description="Relative-strength return lookback (bars)."
    )
    momentum_lookback: int = Field(
        default=252, ge=2, description="12-month return lookback (bars)."
    )
    momentum_skip: int = Field(
        default=21, ge=0, description="Bars skipped at the end of the momentum window."
    )
    atr_period: int = Field(default=14, ge=1, description="ATR period for sizing.")
    stop_atr_mult: float = Field(
        default=1.5, gt=0, description="Risk distance = N x ATR (sizing)."
    )
    trail_atr_mult: float = Field(
        default=3.0, gt=0, description="Trailing-stop distance = N x ATR."
    )
    rsi_period: int = Field(
        default=14, ge=1, description="RSI period for mean reversion."
    )
    rsi_entry: float = Field(
        default=30.0, ge=0, le=100, description="Buy when RSI is below this."
    )
    rsi_exit: float = Field(
        default=50.0, ge=0, le=100, description="Sell when RSI is above this."
    )
    trend_ema: int = Field(
        default=200, ge=1, description="Only buy mean reversion above this EMA."
    )
    mean_revert_hold: int = Field(
        default=10, ge=1, description="Max holding period for mean reversion."
    )
    index_symbol: str = Field(
        default="NIFTY", min_length=1, description="Benchmark index for the trend gate."
    )
    vol_breakout_lookback: int = Field(
        default=20, ge=1, description="Lookback high a volatility breakout must clear."
    )
    vol_expansion_lookback: int = Field(
        default=10, ge=1, description="Bars back the ATR must exceed to be 'expanding'."
    )


class BaselineResult(BaseModel):
    """A comparable, one-line summary of a baseline run."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Baseline name.")
    trades: int = Field(ge=0, description="Closed trades.")
    expectancy_r: float = Field(description="Average R per trade.")
    profit_factor: float = Field(ge=0, description="Gross profit / gross loss.")
    win_rate: float = Field(ge=0, le=100, description="Win rate (%).")
    max_drawdown_pct: float = Field(ge=0, description="Deepest equity drop (%).")
    total_return_pct: float = Field(description="Net P&L as a % of starting capital.")
    perfect_expectancy_r: float = Field(
        description="Expectancy at the observed win rate with perfect +2R/-1R."
    )

    @classmethod
    def from_report(cls, name: str, report: BacktestReport) -> BaselineResult:
        """Summarise a full :class:`BacktestReport` into a comparison row."""
        return cls(
            name=name,
            trades=report.trades,
            expectancy_r=report.expectancy_r,
            profit_factor=report.profit_factor,
            win_rate=report.win_rate,
            max_drawdown_pct=report.max_drawdown_pct,
            total_return_pct=report.total_return_pct,
            perfect_expectancy_r=report.execution_leakage.perfect_expectancy_r,
        )
