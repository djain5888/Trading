"""Unit tests for the market scanner engine."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from app.core.clock import SystemClock
from app.core.timezone import INDIA_TZ
from app.indicators.dependencies import get_indicator_registry
from app.indicators.engine import IndicatorEngine
from app.indicators.models import PriceSeries
from app.market.calendar.config import InMemoryCalendarConfigProvider
from app.market.calendar.holidays import StaticHolidayProvider
from app.market.calendar.service import MarketCalendarService
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import Candle, SeriesKey
from app.market.historical.validation import ValidationEngine
from app.scanner.engine import ScannerEngine
from app.scanner.exceptions import ScannerConfigError, UnknownScannerError
from app.scanner.models import ScannerContext, ScannerResult
from app.scanner.registry import default_registry
from app.scanner.scanners import (
    BreakoutScanner,
    EmaAlignmentScanner,
    GapUpScanner,
    RelativeVolumeScanner,
    RsiMomentumScanner,
    VwapStrengthScanner,
)

_BASE = datetime(2025, 1, 6, 9, 15, tzinfo=INDIA_TZ)


def _candle(
    i: int, *, o: float, h: float, low: float, c: float, v: int = 1000
) -> Candle:
    """Build a candle at minute offset ``i``."""
    return Candle(
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        interval=Interval.ONE_MINUTE,
        timestamp=_BASE + timedelta(minutes=i),
        open=Decimal(f"{o:.4f}"),
        high=Decimal(f"{h:.4f}"),
        low=Decimal(f"{low:.4f}"),
        close=Decimal(f"{c:.4f}"),
        volume=v,
    )


def _rising(n: int, *, start: float = 100.0, step: float = 1.0) -> list[Candle]:
    """Build a strictly rising candle series."""
    out: list[Candle] = []
    for i in range(n):
        close = start + i * step
        out.append(_candle(i, o=close, h=close + 0.5, low=close - 0.5, c=close))
    return out


def _indicator_engine() -> IndicatorEngine:
    return IndicatorEngine(
        HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine()),
        SystemClock(),
        get_indicator_registry(),
    )


def _context(candles: Sequence[Candle]) -> ScannerContext:
    return ScannerContext(
        key=SeriesKey(
            symbol="RELIANCE", exchange=Exchange.NSE, interval=Interval.ONE_MINUTE
        ),
        candles=tuple(candles),
        series=PriceSeries.from_candles(candles),
        indicator_engine=_indicator_engine(),
        clock=SystemClock(),
    )


# -- Registry --------------------------------------------------------------


def test_registry_lists_ten_scanners() -> None:
    """All ten scanners are registered."""
    names = default_registry().available()
    assert len(names) == 10
    assert "ema_alignment" in names and "gap_down" in names


def test_unknown_scanner_raises() -> None:
    """An unregistered scanner name is rejected."""
    with pytest.raises(UnknownScannerError):
        default_registry().create("nope")


def test_invalid_config_raises() -> None:
    """An invalid scanner configuration is rejected."""
    with pytest.raises(ScannerConfigError):
        BreakoutScanner(lookback=0)


# -- Individual scanners: match & filter -----------------------------------


def test_ema_alignment_matches_uptrend() -> None:
    """A strong uptrend yields a bullish EMA alignment result."""
    context = _context(_rising(260))
    result = EmaAlignmentScanner().scan(context)
    assert result is not None
    assert 0 <= result.score <= 100
    assert result.metadata["ema_short"] > result.metadata["ema_long"]


def test_ema_alignment_filters_flat() -> None:
    """A flat series is not EMA-aligned."""
    candles = [_candle(i, o=100, h=100.5, low=99.5, c=100) for i in range(260)]
    assert EmaAlignmentScanner().scan(_context(candles)) is None


def test_breakout_matches() -> None:
    """A close above the prior-high band is a breakout."""
    candles = [_candle(i, o=100, h=100.5, low=99.5, c=100) for i in range(25)]
    candles.append(_candle(25, o=100, h=110.5, low=99.5, c=110))
    result = BreakoutScanner(lookback=20).scan(_context(candles))
    assert result is not None
    assert result.metadata["close"] == 110.0


def test_relative_volume_matches() -> None:
    """A volume spike exceeds the average by the ratio."""
    candles = [_candle(i, o=100, h=100.5, low=99.5, c=100, v=100) for i in range(25)]
    candles.append(_candle(25, o=100, h=100.5, low=99.5, c=100, v=1000))
    result = RelativeVolumeScanner(period=20, min_ratio=1.5).scan(_context(candles))
    assert result is not None
    assert result.metadata["ratio"] > 1.5


def test_vwap_strength_matches_uptrend() -> None:
    """Price closes above VWAP in an uptrend."""
    result = VwapStrengthScanner().scan(_context(_rising(30)))
    assert result is not None
    assert result.metadata["close"] > result.metadata["vwap"]


def test_rsi_band_filters() -> None:
    """An out-of-band RSI configuration filters the symbol out."""
    context = _context(_rising(40))
    # An uptrend has RSI ~100; a low band excludes it.
    assert RsiMomentumScanner(period=14, lower=10, upper=20).scan(context) is None
    # A band that includes 100 matches.
    assert RsiMomentumScanner(period=14, lower=90, upper=100).scan(context) is not None


def test_gap_up_matches() -> None:
    """A large opening gap up is detected."""
    candles = [_candle(0, o=100, h=100.5, low=99.5, c=100)]
    candles.append(_candle(1, o=110, h=111, low=109, c=110))  # gap up 10%
    result = GapUpScanner(min_gap=0.02).scan(_context(candles))
    assert result is not None
    assert result.metadata["gap"] == pytest.approx(0.1)


def test_gap_up_filters_small_gap() -> None:
    """A tiny gap does not trigger the gap-up scanner."""
    candles = [
        _candle(0, o=100, h=100.5, low=99.5, c=100),
        _candle(1, o=100.5, h=101, low=100, c=100.5),
    ]
    assert GapUpScanner(min_gap=0.05).scan(_context(candles)) is None


def test_insufficient_history_returns_none() -> None:
    """Scanners return None (not error) on too little history."""
    context = _context(_rising(5))
    assert EmaAlignmentScanner().scan(context) is None


# -- Engine: ranking, dedupe, parallel -------------------------------------


async def _seed(
    repo: InMemoryCandleRepository, symbol: str, candles: list[Candle]
) -> None:
    """Persist candles under a given symbol."""
    retagged = [c.model_copy(update={"symbol": symbol}) for c in candles]
    await repo.save_many(retagged)


async def test_engine_ranks_and_dedupes() -> None:
    """Results are ranked by score and deduplicated per symbol."""
    repo = InMemoryCandleRepository()
    data_engine = HistoricalDataEngine(repo, ValidationEngine())
    await _seed(repo, "AAA", _rising(260, start=100, step=2))
    await _seed(repo, "BBB", _rising(260, start=100, step=1))

    engine = ScannerEngine(
        data_engine=data_engine,
        indicator_engine=IndicatorEngine(
            data_engine, SystemClock(), get_indicator_registry()
        ),
        calendar=MarketCalendarService(
            clock=SystemClock(),
            config_provider=InMemoryCalendarConfigProvider.with_defaults(),
            holiday_provider=StaticHolidayProvider({}),
        ),
        clock=SystemClock(),
    )
    keys = [
        SeriesKey(symbol=s, exchange=Exchange.NSE, interval=Interval.ONE_MINUTE)
        for s in ("AAA", "BBB")
    ]

    results = await engine.run_all(keys, _BASE, _BASE + timedelta(minutes=300))
    symbols = [r.symbol for r in results]
    assert len(symbols) == len(set(symbols))  # deduped
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)  # ranked


async def test_engine_no_dedupe_keeps_all() -> None:
    """Without dedupe, multiple scanners can each report a symbol."""
    repo = InMemoryCandleRepository()
    data_engine = HistoricalDataEngine(repo, ValidationEngine())
    await _seed(repo, "AAA", _rising(260, start=100, step=2))
    engine = ScannerEngine(
        data_engine=data_engine,
        indicator_engine=IndicatorEngine(
            data_engine, SystemClock(), get_indicator_registry()
        ),
        calendar=MarketCalendarService(
            clock=SystemClock(),
            config_provider=InMemoryCalendarConfigProvider.with_defaults(),
            holiday_provider=StaticHolidayProvider({}),
        ),
        clock=SystemClock(),
    )
    keys = [
        SeriesKey(symbol="AAA", exchange=Exchange.NSE, interval=Interval.ONE_MINUTE)
    ]

    deduped = await engine.run_all(keys, _BASE, _BASE + timedelta(minutes=300))
    raw = await engine.run_all(
        keys, _BASE, _BASE + timedelta(minutes=300), dedupe=False
    )
    assert len(raw) >= len(deduped)


async def test_engine_parallel_large_universe() -> None:
    """The engine scans many symbols concurrently."""
    repo = InMemoryCandleRepository()
    data_engine = HistoricalDataEngine(repo, ValidationEngine())
    keys = []
    for n in range(50):
        symbol = f"SYM{n:03d}"
        await _seed(repo, symbol, _rising(60, start=100, step=1))
        keys.append(
            SeriesKey(
                symbol=symbol, exchange=Exchange.NSE, interval=Interval.ONE_MINUTE
            )
        )
    engine = ScannerEngine(
        data_engine=data_engine,
        indicator_engine=IndicatorEngine(
            data_engine, SystemClock(), get_indicator_registry()
        ),
        calendar=MarketCalendarService(
            clock=SystemClock(),
            config_provider=InMemoryCalendarConfigProvider.with_defaults(),
            holiday_provider=StaticHolidayProvider({}),
        ),
        clock=SystemClock(),
    )

    results = await engine.run(
        "vwap_strength", keys, _BASE, _BASE + timedelta(minutes=120)
    )
    assert len(results) == 50  # every rising symbol trades above VWAP


async def test_engine_missing_symbol_skipped() -> None:
    """A symbol with no stored candles is skipped silently."""
    repo = InMemoryCandleRepository()
    data_engine = HistoricalDataEngine(repo, ValidationEngine())
    engine = ScannerEngine(
        data_engine=data_engine,
        indicator_engine=IndicatorEngine(
            data_engine, SystemClock(), get_indicator_registry()
        ),
        calendar=MarketCalendarService(
            clock=SystemClock(),
            config_provider=InMemoryCalendarConfigProvider.with_defaults(),
            holiday_provider=StaticHolidayProvider({}),
        ),
        clock=SystemClock(),
    )
    keys = [
        SeriesKey(symbol="GHOST", exchange=Exchange.NSE, interval=Interval.ONE_MINUTE)
    ]
    assert await engine.run_all(keys, _BASE, _BASE + timedelta(minutes=10)) == []


def test_rank_static() -> None:
    """rank() sorts by score and dedupes by symbol."""
    a = ScannerResult(
        symbol="A", scanner_name="x", score=10, confidence=0.1, timestamp=_BASE
    )
    a2 = ScannerResult(
        symbol="A", scanner_name="y", score=90, confidence=0.9, timestamp=_BASE
    )
    b = ScannerResult(
        symbol="B", scanner_name="x", score=50, confidence=0.5, timestamp=_BASE
    )
    ranked = ScannerEngine.rank([a, a2, b], dedupe=True)
    assert [r.symbol for r in ranked] == ["A", "B"]
    assert ranked[0].score == 90


def test_result_rejects_out_of_range_score() -> None:
    """Scores must be within 0-100."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ScannerResult(
            symbol="A", scanner_name="x", score=150, confidence=0.5, timestamp=_BASE
        )


def test_di_singleton() -> None:
    """The DI accessor returns a cached singleton engine."""
    from app.scanner.dependencies import _engine_singleton, get_scanner_engine

    _engine_singleton.cache_clear()
    try:
        assert get_scanner_engine() is get_scanner_engine()
    finally:
        _engine_singleton.cache_clear()
