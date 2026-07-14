"""Value objects for the strategy engine.

A strategy *names* what a setup is (breakout, pullback, momentum) and attaches a
confidence, an ATR-derived entry/stop/target and a reward-risk ratio. It is
explicitly not a decision: no buy/sell call, no position sizing.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrategyConfig(BaseModel):
    """Tunable parameters shared by the strategies and the confidence blend."""

    model_config = ConfigDict(frozen=True)

    # -- Momentum: EMA alignment + MACD + RSI band --
    ema_fast: int = Field(default=9, ge=1)
    ema_mid: int = Field(default=21, ge=1)
    ema_slow: int = Field(default=50, ge=1)
    macd_fast: int = Field(default=12, ge=1)
    macd_slow: int = Field(default=26, ge=1)
    macd_signal: int = Field(default=9, ge=1)
    rsi_period: int = Field(default=14, ge=1)
    rsi_momentum_low: float = Field(default=55.0, ge=0, le=100)
    rsi_momentum_high: float = Field(default=75.0, ge=0, le=100)

    # -- Pullback: uptrend + retrace to a moving average + reversal --
    pullback_fast: int = Field(default=20, ge=1)
    pullback_slow: int = Field(default=50, ge=1)
    pullback_touch_pct: float = Field(
        default=1.0, ge=0, description="Low within this % above the fast EMA."
    )

    # -- Breakout: break of the recent high + volume expansion --
    breakout_lookback: int = Field(default=20, ge=1)
    volume_period: int = Field(default=20, ge=1)
    breakout_volume_factor: float = Field(default=1.5, gt=0)

    # -- Risk: ATR-derived stop and target --
    atr_period: int = Field(default=14, ge=1)
    stop_atr_mult: float = Field(default=1.5, gt=0, description="Stop = N x ATR.")
    reward_risk: float = Field(default=2.0, gt=0, description="Target = RR x risk.")

    # -- Confidence blend weights (must sum to 1.0) --
    signal_weight: float = Field(default=0.4, ge=0, le=1)
    regime_weight: float = Field(default=0.2, ge=0, le=1)
    sector_weight: float = Field(default=0.2, ge=0, le=1)
    rs_weight: float = Field(default=0.2, ge=0, le=1)

    history_days: int = Field(default=400, ge=1)
    sector_map: dict[str, str] = Field(
        default_factory=dict, description="Symbol -> sector, for the sector blend."
    )

    @field_validator("sector_map", mode="before")
    @classmethod
    def _normalise_sector_map(cls, value: object) -> dict[str, str]:
        """Upper-case symbol keys and trim sector names."""
        if not isinstance(value, dict):
            raise ValueError("sector_map must be a mapping.")
        return {
            str(key).strip().upper(): str(sector).strip()
            for key, sector in value.items()
            if str(key).strip() and str(sector).strip()
        }

    @model_validator(mode="after")
    def _validate(self) -> StrategyConfig:
        """Enforce unit blend weights and a coherent RSI band."""
        total = (
            self.signal_weight
            + self.regime_weight
            + self.sector_weight
            + self.rs_weight
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Confidence weights must sum to 1.0.")
        if self.rsi_momentum_high <= self.rsi_momentum_low:
            raise ValueError("rsi_momentum_high must exceed rsi_momentum_low.")
        return self

    @property
    def min_candles(self) -> int:
        """Minimum candles a symbol needs to evaluate every strategy."""
        return max(
            self.ema_slow,
            self.pullback_slow,
            self.macd_slow,
            self.breakout_lookback + 1,
            self.volume_period,
            self.atr_period + 1,
            self.rsi_period + 1,
        )


class StrategySetup(BaseModel):
    """A named setup for one symbol (a description, never a recommendation)."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="The symbol the setup was found on.")
    strategy: str = Field(description="The strategy that named the setup.")
    confidence: float = Field(
        ge=0, le=100, description="Blended confidence (signal + context)."
    )
    signal_strength: float = Field(
        ge=0, le=100, description="Raw pre-blend signal strength."
    )
    entry: float = Field(gt=0, description="Reference entry price.")
    stop: float = Field(gt=0, description="ATR-derived stop price.")
    target: float = Field(gt=0, description="Reward-risk-derived target price.")
    reward_risk: float = Field(ge=0, description="Target-to-stop reward-risk ratio.")


class StrategyReport(BaseModel):
    """The strategy engine's output: a ranked list of named setups."""

    model_config = ConfigDict(frozen=True)

    available: bool = Field(description="Whether any symbol could be evaluated.")
    setups: tuple[StrategySetup, ...] = Field(
        default_factory=tuple, description="Setups ranked by confidence, best first."
    )
    detail: str = Field(description="Human-readable summary.")
    generated_at: datetime = Field(description="When the report was produced.")

    @classmethod
    def unavailable(cls, *, generated_at: datetime, detail: str) -> StrategyReport:
        """Build a graceful "no setups" report."""
        return cls(available=False, detail=detail, generated_at=generated_at)
