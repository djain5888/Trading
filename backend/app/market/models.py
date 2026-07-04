"""Request and response models for the market-data contract.

All models are immutable Pydantic v2 value objects. They form the stable
boundary every :class:`app.providers.base.MarketDataProvider` implementation
must speak, keeping the rest of Titan independent of any specific broker.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.market.enums import Exchange, Interval, MarketStatus


class MarketModel(BaseModel):
    """Base class for all market-data models (immutable)."""

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class QuoteRequest(MarketModel):
    """Request for a single instrument quote."""

    symbol: str = Field(min_length=1, description="Trading symbol, e.g. 'RELIANCE'.")
    exchange: Exchange = Field(default=Exchange.NSE, description="Listing exchange.")


class QuotesRequest(MarketModel):
    """Request for quotes of several instruments on one exchange."""

    symbols: tuple[str, ...] = Field(min_length=1, description="Trading symbols.")
    exchange: Exchange = Field(default=Exchange.NSE, description="Listing exchange.")


class HistoricalDataRequest(MarketModel):
    """Request for historical OHLCV candles."""

    symbol: str = Field(min_length=1, description="Trading symbol.")
    exchange: Exchange = Field(default=Exchange.NSE, description="Listing exchange.")
    interval: Interval = Field(description="Candle interval.")
    start: datetime = Field(description="Inclusive range start (UTC).")
    end: datetime = Field(description="Inclusive range end (UTC).")

    @model_validator(mode="after")
    def _validate_range(self) -> HistoricalDataRequest:
        """Ensure the requested time range is well ordered.

        Raises:
            ValueError: If ``end`` is not strictly after ``start``.
        """
        if self.end <= self.start:
            raise ValueError("'end' must be strictly after 'start'.")
        return self


class SymbolSearchRequest(MarketModel):
    """Request to search the instrument master by free text."""

    query: str = Field(min_length=1, description="Search query.")
    exchange: Exchange | None = Field(
        default=None, description="Optional exchange filter."
    )


class MarketDepthRequest(MarketModel):
    """Request for the order-book depth of an instrument."""

    symbol: str = Field(min_length=1, description="Trading symbol.")
    exchange: Exchange = Field(default=Exchange.NSE, description="Listing exchange.")


class MarketStatusRequest(MarketModel):
    """Request for the trading-session status of an exchange."""

    exchange: Exchange = Field(default=Exchange.NSE, description="Listing exchange.")


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class Quote(MarketModel):
    """A point-in-time price snapshot for one instrument."""

    symbol: str = Field(description="Trading symbol.")
    exchange: Exchange = Field(description="Listing exchange.")
    last_price: Decimal = Field(description="Last traded price.")
    open: Decimal | None = Field(default=None, description="Session open price.")
    high: Decimal | None = Field(default=None, description="Session high price.")
    low: Decimal | None = Field(default=None, description="Session low price.")
    previous_close: Decimal | None = Field(
        default=None, description="Previous session close price."
    )
    volume: int | None = Field(default=None, ge=0, description="Traded volume.")
    timestamp: datetime = Field(description="Quote timestamp (UTC).")


class Candle(MarketModel):
    """A single OHLCV bar."""

    timestamp: datetime = Field(description="Bar open time (UTC).")
    open: Decimal = Field(description="Open price.")
    high: Decimal = Field(description="High price.")
    low: Decimal = Field(description="Low price.")
    close: Decimal = Field(description="Close price.")
    volume: int = Field(ge=0, description="Traded volume.")


class HistoricalData(MarketModel):
    """Ordered series of candles for one instrument and interval."""

    symbol: str = Field(description="Trading symbol.")
    exchange: Exchange = Field(description="Listing exchange.")
    interval: Interval = Field(description="Candle interval.")
    candles: tuple[Candle, ...] = Field(
        default_factory=tuple, description="Chronologically ordered candles."
    )


class DepthLevel(MarketModel):
    """A single price level in the order book."""

    price: Decimal = Field(description="Price at this level.")
    quantity: int = Field(ge=0, description="Aggregate quantity at this level.")
    orders: int | None = Field(
        default=None, ge=0, description="Number of orders at this level."
    )


class MarketDepth(MarketModel):
    """Bid/ask order-book snapshot for one instrument."""

    symbol: str = Field(description="Trading symbol.")
    exchange: Exchange = Field(description="Listing exchange.")
    bids: tuple[DepthLevel, ...] = Field(
        default_factory=tuple, description="Buy side, best first."
    )
    asks: tuple[DepthLevel, ...] = Field(
        default_factory=tuple, description="Sell side, best first."
    )
    timestamp: datetime = Field(description="Snapshot timestamp (UTC).")


class SymbolSearchResult(MarketModel):
    """A single instrument returned by a symbol search."""

    symbol: str = Field(description="Trading symbol.")
    name: str = Field(description="Human-readable instrument name.")
    exchange: Exchange = Field(description="Listing exchange.")
    instrument_type: str = Field(
        default="EQ", description="Instrument type code (e.g. 'EQ', 'FUT', 'OPT')."
    )


class SymbolSearchResponse(MarketModel):
    """Results of a symbol search."""

    query: str = Field(description="Original search query.")
    results: tuple[SymbolSearchResult, ...] = Field(
        default_factory=tuple, description="Matching instruments."
    )


class MarketStatusInfo(MarketModel):
    """Trading-session status of an exchange at a point in time."""

    exchange: Exchange = Field(description="Listing exchange.")
    status: MarketStatus = Field(description="Current session status.")
    timestamp: datetime = Field(description="Status timestamp (UTC).")
    message: str | None = Field(default=None, description="Optional status detail.")
