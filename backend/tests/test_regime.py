"""Unit tests for the market-regime engine."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from app.core.clock import FakeClock
from app.core.timezone import INDIA_TZ
from app.indicators.dependencies import get_indicator_registry
from app.indicators.engine import IndicatorEngine
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import Candle
from app.market.historical.validation import ValidationEngine
from app.market.regime.engine import MarketRegimeEngine
from app.market.regime.models import Regime, RegimeConfig

_END = datetime(2025, 1, 6, 15, 30, tzinfo=INDIA_TZ)

# Short periods so a handful of candles exercises the full logic.
_CONFIG = RegimeConfig(
    index_symbol="NIFTY",
    trend_fast_period=3,
    trend_slow_period=5,
    slope_lookback=2,
    breadth_period=3,
    atr_period=2,
    volatility_lookback=10,
    history_days=100,
)


def _engine() -> tuple[HistoricalDataEngine, MarketRegimeEngine]:
    """Build a data engine and a regime engine sharing it."""
    repo = InMemoryCandleRepository()
    data_engine = HistoricalDataEngine(repo, ValidationEngine())
    indicator_engine = IndicatorEngine(
        data_engine, FakeClock(_END), get_indicator_registry()
    )
    regime_engine = MarketRegimeEngine(
        data_engine, indicator_engine, FakeClock(_END), _CONFIG
    )
    return data_engine, regime_engine


async def _seed(
    engine: HistoricalDataEngine, symbol: str, closes: Sequence[float]
) -> None:
    """Seed a daily series whose closes follow ``closes`` (oldest first)."""
    count = len(closes)
    candles = [
        Candle(
            symbol=symbol,
            exchange=Exchange.NSE,
            interval=Interval.ONE_DAY,
            timestamp=_END - timedelta(days=count - 1 - i),
            open=Decimal(f"{close:.4f}"),
            high=Decimal(f"{close * 1.01:.4f}"),
            low=Decimal(f"{close * 0.99:.4f}"),
            close=Decimal(f"{close:.4f}"),
            volume=1000,
        )
        for i, close in enumerate(closes)
    ]
    await engine.import_candles(candles)


# -- Classification --------------------------------------------------------


async def test_bull_regime() -> None:
    """A rising index with broad participation classifies as BULL."""
    data, regime = _engine()
    await _seed(data, "NIFTY", [100 + i for i in range(12)])
    await _seed(data, "AAA", [50 + i for i in range(12)])
    await _seed(data, "BBB", [70 + i for i in range(12)])

    report = await regime.analyze(["AAA", "BBB"], interval=Interval.ONE_DAY, end=_END)

    assert report.available
    assert report.regime is Regime.BULL
    assert report.confidence == 100.0
    assert report.breadth == 100.0


async def test_bear_regime() -> None:
    """A falling index with weak participation classifies as BEAR."""
    data, regime = _engine()
    await _seed(data, "NIFTY", [120 - i for i in range(12)])
    await _seed(data, "AAA", [80 - i for i in range(12)])
    await _seed(data, "BBB", [90 - i for i in range(12)])

    report = await regime.analyze(["AAA", "BBB"], interval=Interval.ONE_DAY, end=_END)

    assert report.available
    assert report.regime is Regime.BEAR
    assert report.confidence == 100.0
    assert report.breadth == 0.0


async def test_sideways_regime() -> None:
    """A flat index with split breadth classifies as SIDEWAYS."""
    data, regime = _engine()
    await _seed(data, "NIFTY", [100.0] * 12)
    await _seed(data, "AAA", [50 + i for i in range(12)])  # above its EMA
    await _seed(data, "BBB", [90 - i for i in range(12)])  # below its EMA

    report = await regime.analyze(["AAA", "BBB"], interval=Interval.ONE_DAY, end=_END)

    assert report.available
    assert report.regime is Regime.SIDEWAYS
    assert report.breadth == 50.0


# -- Graceful degradation --------------------------------------------------


async def test_missing_index_data_is_unavailable() -> None:
    """With no index data the report degrades to 'unavailable', not an error."""
    data, regime = _engine()
    await _seed(data, "AAA", [50 + i for i in range(12)])

    report = await regime.analyze(["AAA"], interval=Interval.ONE_DAY, end=_END)

    assert report.available is False
    assert report.regime is None
    assert report.confidence == 0.0
    assert "unavailable" in report.detail.lower()


async def test_short_index_history_is_unavailable() -> None:
    """Too few index candles for the slow EMA degrades gracefully."""
    data, regime = _engine()
    await _seed(data, "NIFTY", [100 + i for i in range(3)])  # < min_index_candles

    report = await regime.analyze([], interval=Interval.ONE_DAY, end=_END)

    assert report.available is False
    assert report.regime is None


async def test_breadth_counts_only_symbols_with_data() -> None:
    """Breadth ignores members lacking enough history for the EMA."""
    data, regime = _engine()
    await _seed(data, "NIFTY", [100 + i for i in range(12)])
    await _seed(data, "AAA", [50 + i for i in range(12)])  # above EMA
    await _seed(data, "BBB", [1.0])  # too short: excluded from breadth

    report = await regime.analyze(["AAA", "BBB"], interval=Interval.ONE_DAY, end=_END)

    assert report.available
    assert report.breadth == 100.0  # only AAA counted, and it is above
