"""Tests for TASK-018 wiring: default watchlist, NIFTY benchmark, sector map."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

from app.config.watchlist import WatchlistConfig, get_watchlist_config, load_watchlist
from app.core.clock import FakeClock
from app.core.timezone import INDIA_TZ
from app.indicators.dependencies import get_indicator_registry
from app.indicators.engine import IndicatorEngine
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import Candle
from app.market.historical.validation import ValidationEngine
from app.market.regime.dependencies import get_market_regime_engine
from app.market.regime.engine import MarketRegimeEngine
from app.market.regime.models import RegimeConfig
from app.market.relative.dependencies import get_relative_strength_engine
from app.market.relative.engine import RelativeStrengthEngine
from app.market.relative.models import RSConfig
from app.market.sector.dependencies import get_sector_strength_engine
from app.market.sector.engine import SectorStrengthEngine
from app.market.sector.models import SectorConfig

_END = datetime(2025, 1, 6, 15, 30, tzinfo=INDIA_TZ)


# -- Default watchlist config ----------------------------------------------


def test_default_watchlist_has_symbols_and_sectors() -> None:
    """The built-in watchlist is non-empty and fully sectored."""
    config = WatchlistConfig()
    assert config.index_symbol == "NIFTY"
    assert len(config.symbols) >= 4
    # Every default symbol maps to a sector (so none is silently excluded).
    assert all(symbol in config.sectors for symbol in config.symbols)
    # Several sectors have >= 2 members, so the sector engine can rank them.
    counts: dict[str, int] = {}
    for sector in config.sectors.values():
        counts[sector] = counts.get(sector, 0) + 1
    assert sum(1 for n in counts.values() if n >= 2) >= 2


def test_get_watchlist_config_is_populated() -> None:
    """The DI accessor yields the seeded default watchlist."""
    config = get_watchlist_config()
    assert config.symbols
    assert config.sectors


def test_watchlist_file_override(tmp_path: Path) -> None:
    """A JSON file overrides the defaults and is normalised."""
    path = tmp_path / "watchlist.json"
    path.write_text(
        json.dumps(
            {
                "index_symbol": "banknifty",
                "symbols": ["hdfcbank", " icicibank ", "hdfcbank"],
                "sectors": {"hdfcbank": "BANKING", "icicibank": "banking"},
            }
        ),
        encoding="utf-8",
    )

    config = load_watchlist(path)

    assert config.index_symbol == "BANKNIFTY"
    assert config.symbols == ("HDFCBANK", "ICICIBANK")  # trimmed, upper, deduped
    assert config.sectors == {"HDFCBANK": "BANKING", "ICICIBANK": "banking"}


def test_watchlist_missing_file_falls_back(tmp_path: Path) -> None:
    """A missing/invalid file degrades to the defaults, never raising."""
    config = load_watchlist(tmp_path / "does-not-exist.json")
    assert config.symbols == WatchlistConfig().symbols


# -- DI wiring injects the benchmark + sector map --------------------------


def test_di_engines_are_wired_to_the_watchlist() -> None:
    """Regime/sector/RS engines built via DI carry the watchlist metadata."""
    watchlist = get_watchlist_config()

    regime = get_market_regime_engine()
    sector = get_sector_strength_engine()
    relative = get_relative_strength_engine()

    assert isinstance(regime, MarketRegimeEngine)
    assert regime.config.index_symbol == watchlist.index_symbol
    assert isinstance(sector, SectorStrengthEngine)
    assert sector.config.sector_map == watchlist.sectors
    assert isinstance(relative, RelativeStrengthEngine)
    assert relative.config.index_symbol == watchlist.index_symbol
    assert relative.config.sector_map == watchlist.sectors


# -- End-to-end: engines populated once fed a benchmark + sectors ----------


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


async def test_context_engines_populate_with_benchmark_and_sectors() -> None:
    """With NIFTY + a sectored watchlist, regime/sector/RS all populate."""
    repo = InMemoryCandleRepository()
    data = HistoricalDataEngine(repo, ValidationEngine())
    clock = FakeClock(_END)
    indicators = IndicatorEngine(data, clock, get_indicator_registry())

    sector_map = {"AAA": "IT", "BBB": "IT", "CCC": "BANK", "DDD": "BANK"}
    await _seed(data, "NIFTY", [100 + i for i in range(12)])
    await _seed(data, "AAA", [50 + 2 * i for i in range(12)])  # strong up
    await _seed(data, "BBB", [70 + i for i in range(12)])  # up
    await _seed(data, "CCC", [90 - i for i in range(12)])  # down
    await _seed(data, "DDD", [80 - 2 * i for i in range(12)])  # strong down
    symbols = ["AAA", "BBB", "CCC", "DDD"]

    regime = MarketRegimeEngine(
        data,
        indicators,
        clock,
        RegimeConfig(
            index_symbol="NIFTY",
            trend_fast_period=3,
            trend_slow_period=5,
            slope_lookback=2,
            breadth_period=3,
            atr_period=2,
            volatility_lookback=10,
            history_days=100,
        ),
    )
    sector = SectorStrengthEngine(
        data,
        indicators,
        clock,
        SectorConfig(
            sector_map=sector_map,
            ema_period=3,
            momentum_period=2,
            slope_lookback=2,
            min_members=2,
            history_days=100,
        ),
    )
    relative = RelativeStrengthEngine(
        data,
        clock,
        RSConfig(
            index_symbol="NIFTY",
            sector_map=sector_map,
            lookback_periods=3,
            history_days=100,
        ),
    )

    regime_report = await regime.analyze(symbols, interval=Interval.ONE_DAY, end=_END)
    sector_report = await sector.analyze(symbols, interval=Interval.ONE_DAY, end=_END)
    rs_report = await relative.analyze(symbols, interval=Interval.ONE_DAY, end=_END)

    # Regime classifies (not "unavailable").
    assert regime_report.available
    assert regime_report.regime is not None

    # Sectors rank (both IT and BANK bucketed).
    assert sector_report.available
    assert {s.sector for s in sector_report.ranked} == {"IT", "BANK"}

    # RS differentiates against the benchmark (not all neutral 50).
    assert rs_report.available
    assert rs_report.market_available
    composites = {entry.composite for entry in rs_report.entries}
    assert composites != {50.0}
