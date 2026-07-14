"""Value objects for the paper-trading engine.

These track *hypothetical* trades opened from strategy setups so Titan can
measure whether its setups work. No real orders are ever placed.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.market.enums import Exchange, Interval


class TradeStatus(StrEnum):
    """Whether a paper trade is still open or has been closed."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


class ExitReason(StrEnum):
    """Why a paper trade was closed."""

    STOP = "STOP"
    TARGET = "TARGET"
    TIMEOUT = "TIMEOUT"


class PaperConfig(BaseModel):
    """Portfolio and risk configuration for the paper book."""

    model_config = ConfigDict(frozen=True)

    starting_capital: float = Field(
        default=100_000.0, gt=0, description="Starting book capital."
    )
    risk_pct: float = Field(
        default=1.0, gt=0, le=100, description="Percent of equity risked per trade."
    )
    max_positions: int = Field(
        default=5, ge=1, description="Maximum concurrent open positions."
    )
    max_holding_days: int = Field(
        default=10, ge=1, description="Force-close after this many days held."
    )
    min_confidence: float = Field(
        default=0.0, ge=0, le=100, description="Only open setups at/above this."
    )


class PaperTrade(BaseModel):
    """One hypothetical long trade (immutable; updates return a copy)."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Deterministic trade id (symbol-strategy-entrytime).")
    symbol: str = Field(description="Traded symbol.")
    strategy: str = Field(description="Strategy that produced the setup.")
    exchange: Exchange = Field(description="Listing exchange.")
    interval: Interval = Field(description="Candle interval the trade is marked on.")
    entry_price: float = Field(gt=0, description="Entry price.")
    stop_price: float = Field(gt=0, description="Stop price (below entry).")
    target_price: float = Field(gt=0, description="Target price (above entry).")
    size: float = Field(ge=0, description="Position size (shares).")
    opened_at: datetime = Field(description="Timestamp of the entry candle.")
    status: TradeStatus = Field(default=TradeStatus.OPEN, description="Trade status.")
    last_price: float = Field(gt=0, description="Latest mark (or exit) price.")
    pnl: float = Field(
        default=0.0, description="P&L: unrealized if open, realized if closed."
    )
    exit_price: float | None = Field(default=None, description="Exit price, if closed.")
    exit_reason: ExitReason | None = Field(default=None, description="Exit reason.")
    closed_at: datetime | None = Field(default=None, description="Close timestamp.")

    @model_validator(mode="after")
    def _validate_levels(self) -> PaperTrade:
        """Ensure a coherent long setup: stop < entry < target."""
        if not (self.stop_price < self.entry_price < self.target_price):
            raise ValueError("Require stop < entry < target for a long paper trade.")
        return self

    def marked(self, price: float) -> PaperTrade:
        """Return a copy marked to ``price`` with updated unrealized P&L."""
        return self.model_copy(
            update={"last_price": price, "pnl": (price - self.entry_price) * self.size}
        )

    def closed_with(self, price: float, reason: ExitReason, at: datetime) -> PaperTrade:
        """Return a closed copy with realized P&L at the exit ``price``."""
        return self.model_copy(
            update={
                "status": TradeStatus.CLOSED,
                "exit_price": price,
                "exit_reason": reason,
                "closed_at": at,
                "last_price": price,
                "pnl": (price - self.entry_price) * self.size,
            }
        )


class StrategyStats(BaseModel):
    """Per-strategy performance over closed trades."""

    model_config = ConfigDict(frozen=True)

    strategy: str = Field(description="Strategy name.")
    trades: int = Field(ge=0, description="Closed trades for this strategy.")
    wins: int = Field(ge=0, description="Winning closed trades.")
    win_rate: float = Field(ge=0, le=100, description="Win rate percent.")
    total_pnl: float = Field(description="Realized P&L for this strategy.")


class PaperReport(BaseModel):
    """The paper book's performance snapshot."""

    model_config = ConfigDict(frozen=True)

    generated_at: datetime = Field(description="When the report was produced.")
    starting_capital: float = Field(description="Configured starting capital.")
    equity: float = Field(description="Starting capital + realized P&L.")
    open_positions: tuple[PaperTrade, ...] = Field(
        default_factory=tuple, description="Currently open trades (mark-to-market)."
    )
    closed_trades: tuple[PaperTrade, ...] = Field(
        default_factory=tuple, description="Closed trades, newest first."
    )
    total_pnl: float = Field(description="Realized + unrealized P&L.")
    realized_pnl: float = Field(description="Realized P&L from closed trades.")
    open_pnl: float = Field(description="Unrealized P&L across open trades.")
    win_rate: float = Field(ge=0, le=100, description="Closed-trade win rate percent.")
    avg_win: float = Field(description="Average winning trade P&L.")
    avg_loss: float = Field(description="Average losing trade P&L (<= 0).")
    profit_factor: float = Field(
        ge=0, description="Gross profit / gross loss (0 when no losses)."
    )
    wins: int = Field(ge=0, description="Number of winning closed trades.")
    losses: int = Field(ge=0, description="Number of losing closed trades.")
    per_strategy: tuple[StrategyStats, ...] = Field(
        default_factory=tuple, description="Per-strategy breakdown."
    )
