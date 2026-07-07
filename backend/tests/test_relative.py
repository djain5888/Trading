"""Unit tests for the relative-strength engine."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from app.core.clock import FakeClock
from app.core.timezone import INDIA_TZ
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import Candle
from app.market.historical.validation import ValidationEngine
from app.market.relative.engine import RelativeStrengthEngine
from app.market.relative.models import RSConfig

_END = datetime(2025, 1, 6, 15, 30, tzinfo=INDIA_TZ)


def _config(sector_map: dict[str, str], **overrides: object) -> RSConfig:
    """Build a short-lookback config over the given metadata."""
    base: dict[str, object] = {
        "index_symbol": "NIFTY",
        "lookback_periods": 3,
        "leader_percentile": 80.0,
        "sector_map": sector_map,
        "history_days": 100,
    }
    base.update(overrides)
    return RSConfig(**base)


def _engine(config: RSConfig) -> tuple[HistoricalDataEngine, RelativeStrengthEngine]:
    """Build a data engine and a relative-strength engine sharing it."""
    repo = InMemoryCandleRepository()
    data_engine = HistoricalDataEngine(repo, ValidationEngine())
    rs_engine = RelativeStrengthEngine(data_engine, FakeClock(_END), config)
    return data_engine, rs_engine


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


# -- Ranking and leaders ---------------------------------------------------


async def test_ranking_and_leader_detection() -> None:
    """The strongest symbol leads on both axes and ranks first."""
    config = _config({"AAA": "TECH", "BBB": "TECH", "CCC": "BANK"})
    data, rs = _engine(config)
    await _seed(data, "NIFTY", [100, 100, 100, 103])  # +3%
    await _seed(data, "AAA", [100, 100, 100, 115])  # +15%
    await _seed(data, "BBB", [100, 100, 100, 101.5])  # +1.5%
    await _seed(data, "CCC", [100, 100, 100, 97])  # -3%

    report = await rs.analyze(
        ["AAA", "BBB", "CCC"], interval=Interval.ONE_DAY, end=_END
    )

    assert report.available
    assert report.market_available
    assert report.entries[0].symbol == "AAA"
    leader = report.by_symbol["AAA"]
    assert leader.leader
    assert leader.rs_vs_market == 100.0
    assert leader.rs_vs_sector == 100.0
    assert leader.composite == 100.0
    assert tuple(e.symbol for e in report.leaders) == ("AAA",)
    assert report.by_symbol["BBB"].leader is False
    assert report.by_symbol["CCC"].leader is False


# -- Graceful degradation --------------------------------------------------


async def test_missing_market_neutralises_component() -> None:
    """Without NIFTY the vs-market component is neutral, not a failure."""
    config = _config({"AAA": "TECH", "BBB": "TECH"})
    data, rs = _engine(config)
    await _seed(data, "AAA", [100, 100, 100, 115])
    await _seed(data, "BBB", [100, 100, 100, 90])

    report = await rs.analyze(["AAA", "BBB"], interval=Interval.ONE_DAY, end=_END)

    assert report.available
    assert report.market_available is False
    assert all(entry.rs_vs_market == 50.0 for entry in report.entries)
    assert report.leaders == ()  # neutral market blocks leadership


async def test_missing_sector_neutralises_component() -> None:
    """Without sector metadata the vs-sector component is neutral."""
    config = _config({})  # no sector metadata
    data, rs = _engine(config)
    await _seed(data, "NIFTY", [100, 100, 100, 103])
    await _seed(data, "AAA", [100, 100, 100, 115])
    await _seed(data, "BBB", [100, 100, 100, 101])

    report = await rs.analyze(["AAA", "BBB"], interval=Interval.ONE_DAY, end=_END)

    assert report.available
    assert all(entry.rs_vs_sector == 50.0 for entry in report.entries)


async def test_short_history_symbol_excluded() -> None:
    """A symbol with too little history is excluded, not failed."""
    config = _config({"AAA": "TECH", "BBB": "TECH"})
    data, rs = _engine(config)
    await _seed(data, "NIFTY", [100, 100, 100, 103])
    await _seed(data, "AAA", [100, 100, 100, 115])
    await _seed(data, "BBB", [100, 100])  # too short for a 3-period return

    report = await rs.analyze(["AAA", "BBB"], interval=Interval.ONE_DAY, end=_END)

    assert report.available
    assert "BBB" not in report.by_symbol
    assert "AAA" in report.by_symbol


async def test_no_valid_symbols_is_unavailable() -> None:
    """With no symbol having enough history the report degrades gracefully."""
    config = _config({"AAA": "TECH"})
    data, rs = _engine(config)
    await _seed(data, "AAA", [100, 100])  # too short

    report = await rs.analyze(["AAA"], interval=Interval.ONE_DAY, end=_END)

    assert report.available is False
    assert report.entries == ()
    assert "unavailable" in report.detail.lower()
