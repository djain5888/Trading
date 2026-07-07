"""Unit tests for the sector-strength engine."""

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
from app.market.sector.engine import SectorStrengthEngine
from app.market.sector.models import SectorConfig, SectorTrend

_END = datetime(2025, 1, 6, 15, 30, tzinfo=INDIA_TZ)


def _config(sector_map: dict[str, str], **overrides: object) -> SectorConfig:
    """Build a short-period config over the given metadata."""
    base: dict[str, object] = {
        "sector_map": sector_map,
        "ema_period": 3,
        "momentum_period": 2,
        "slope_lookback": 2,
        "min_members": 2,
        "history_days": 100,
    }
    base.update(overrides)
    return SectorConfig(**base)


def _engine(config: SectorConfig) -> tuple[HistoricalDataEngine, SectorStrengthEngine]:
    """Build a data engine and a sector engine sharing it."""
    repo = InMemoryCandleRepository()
    data_engine = HistoricalDataEngine(repo, ValidationEngine())
    indicator_engine = IndicatorEngine(
        data_engine, FakeClock(_END), get_indicator_registry()
    )
    sector_engine = SectorStrengthEngine(
        data_engine, indicator_engine, FakeClock(_END), config
    )
    return data_engine, sector_engine


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


def _rising(base: float) -> list[float]:
    return [base + i for i in range(12)]


def _falling(base: float) -> list[float]:
    return [base - i for i in range(12)]


# -- Ranking ---------------------------------------------------------------


async def test_ranking_orders_strong_over_weak() -> None:
    """A rising sector outranks a falling one, with matching trends."""
    config = _config({"AAA": "TECH", "BBB": "TECH", "CCC": "BANK", "DDD": "BANK"})
    data, sector = _engine(config)
    await _seed(data, "AAA", _rising(50))
    await _seed(data, "BBB", _rising(70))
    await _seed(data, "CCC", _falling(90))
    await _seed(data, "DDD", _falling(80))

    report = await sector.analyze(
        ["AAA", "BBB", "CCC", "DDD"], interval=Interval.ONE_DAY, end=_END
    )

    assert report.available
    names = [s.sector for s in report.ranked]
    assert names == ["TECH", "BANK"]
    assert report.ranked[0].score > report.ranked[1].score
    assert report.ranked[0].trend is SectorTrend.UP
    assert report.ranked[1].trend is SectorTrend.DOWN
    assert report.ranked[0].above_pct == 100.0
    assert report.ranked[1].above_pct == 0.0
    assert report.strongest[0].sector == "TECH"
    assert report.weakest[0].sector == "BANK"


async def test_trend_classification() -> None:
    """Rising, flat and falling sectors classify as UP, FLAT and DOWN."""
    config = _config(
        {
            "U1": "UP",
            "U2": "UP",
            "F1": "FLAT",
            "F2": "FLAT",
            "D1": "DOWN",
            "D2": "DOWN",
        }
    )
    data, sector = _engine(config)
    await _seed(data, "U1", _rising(50))
    await _seed(data, "U2", _rising(60))
    await _seed(data, "F1", [100.0] * 12)
    await _seed(data, "F2", [200.0] * 12)
    await _seed(data, "D1", _falling(90))
    await _seed(data, "D2", _falling(80))

    report = await sector.analyze(
        ["U1", "U2", "F1", "F2", "D1", "D2"], interval=Interval.ONE_DAY, end=_END
    )

    trends = {s.sector: s.trend for s in report.ranked}
    assert trends == {
        "UP": SectorTrend.UP,
        "FLAT": SectorTrend.FLAT,
        "DOWN": SectorTrend.DOWN,
    }


# -- Graceful degradation --------------------------------------------------


async def test_missing_metadata_is_unavailable() -> None:
    """With no sector metadata the report degrades to 'unavailable'."""
    data, sector = _engine(_config({}))
    await _seed(data, "AAA", _rising(50))

    report = await sector.analyze(["AAA"], interval=Interval.ONE_DAY, end=_END)

    assert report.available is False
    assert report.ranked == ()
    assert "unavailable" in report.detail.lower()


async def test_thin_sector_is_excluded_not_failed() -> None:
    """A sector with fewer than min_members valid stocks is dropped."""
    config = _config({"AAA": "TECH", "BBB": "TECH", "XXX": "SOLO"})
    data, sector = _engine(config)
    await _seed(data, "AAA", _rising(50))
    await _seed(data, "BBB", _rising(70))
    await _seed(data, "XXX", _rising(30))  # SOLO has only one member

    report = await sector.analyze(
        ["AAA", "BBB", "XXX"], interval=Interval.ONE_DAY, end=_END
    )

    assert report.available
    assert [s.sector for s in report.ranked] == ["TECH"]


async def test_short_member_data_drops_member() -> None:
    """A member with too little history is excluded from its sector."""
    config = _config({"AAA": "TECH", "BBB": "TECH"})
    data, sector = _engine(config)
    await _seed(data, "AAA", _rising(50))
    await _seed(data, "BBB", [1.0])  # too short: excluded, leaving TECH thin

    report = await sector.analyze(["AAA", "BBB"], interval=Interval.ONE_DAY, end=_END)

    assert report.available
    assert report.ranked == ()  # TECH now has one valid member, below min
