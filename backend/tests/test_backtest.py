"""Tests for the backtesting engine, guard and expectancy maths (no network)."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from app.backtest import metrics
from app.backtest.engine import BacktestEngine
from app.backtest.guard import LookaheadError, LookaheadGuard
from app.backtest.models import (
    BacktestConfig,
    BacktestCosts,
    BacktestTrade,
    ExitKind,
)
from app.core.clock import FakeClock
from app.core.timezone import INDIA_TZ
from app.indicators.dependencies import get_indicator_registry
from app.indicators.engine import IndicatorEngine
from app.market.calendar.config import InMemoryCalendarConfigProvider
from app.market.calendar.holidays import StaticHolidayProvider
from app.market.calendar.service import MarketCalendarService
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import Candle, SeriesKey
from app.market.historical.validation import ValidationEngine
from app.market.regime.engine import MarketRegimeEngine
from app.market.regime.models import Regime, RegimeReport
from app.paper.models import ExitReason
from app.strategy.engine import StrategyEngine
from app.strategy.models import StrategyReport, StrategySetup

_TZ = INDIA_TZ


# -- Fakes & fixtures ------------------------------------------------------


def _calendar() -> MarketCalendarService:
    """Return an NSE calendar with default sessions and no holidays."""
    return MarketCalendarService(
        clock=FakeClock(datetime(2025, 1, 6, 9, 15, tzinfo=_TZ)),
        config_provider=InMemoryCalendarConfigProvider.with_defaults(),
        holiday_provider=StaticHolidayProvider({}),
    )


def _data() -> HistoricalDataEngine:
    """Return an in-memory data engine."""
    return HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())


def _candle(
    symbol: str, on: date, o: float, h: float, low: float, c: float, volume: int = 1000
) -> Candle:
    """Build one daily domain candle for ``symbol`` on ``on``."""
    return Candle(
        symbol=symbol,
        exchange=Exchange.NSE,
        interval=Interval.ONE_DAY,
        timestamp=datetime(on.year, on.month, on.day, 9, 15, tzinfo=_TZ),
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(low)),
        close=Decimal(str(c)),
        volume=volume,
    )


async def _seed(engine: HistoricalDataEngine, candles: Sequence[Candle]) -> None:
    """Import a single symbol's candles."""
    await engine.import_candles(list(candles))


class _FixedStrategyEngine(StrategyEngine):
    """A strategy engine that emits pre-scripted setups on given dates."""

    def __init__(self, setups: dict[date, tuple[StrategySetup, ...]]) -> None:
        self._setups = setups

    async def analyze(
        self,
        symbols: Sequence[str],
        *,
        exchange: Exchange = Exchange.NSE,
        interval: Interval = Interval.ONE_DAY,
        end: datetime | None = None,
        scan_results: Sequence[object] = (),
        regime: object = None,
        sectors: object = None,
        relative: object = None,
    ) -> StrategyReport:
        now = end or datetime(2025, 1, 6, tzinfo=_TZ)
        setups = self._setups.get(now.date(), ())
        return StrategyReport(
            available=bool(setups), setups=setups, detail="stub", generated_at=now
        )


class _FixedRegimeEngine(MarketRegimeEngine):
    """A regime engine that returns a fixed classification (or unavailable)."""

    def __init__(self, regime: Regime | None = None) -> None:
        self._regime = regime

    async def analyze(
        self,
        symbols: Sequence[str],
        *,
        exchange: Exchange = Exchange.NSE,
        interval: Interval = Interval.ONE_DAY,
        end: datetime | None = None,
    ) -> RegimeReport:
        now = end or datetime(2025, 1, 6, tzinfo=_TZ)
        if self._regime is None:
            return RegimeReport.unavailable(
                index_symbol="NIFTY", generated_at=now, detail="stub"
            )
        return RegimeReport(
            available=True,
            regime=self._regime,
            confidence=60.0,
            index_symbol="NIFTY",
            detail="stub",
            generated_at=now,
        )


def _setup(
    entry: float = 100.0, stop: float = 95.0, target: float = 110.0
) -> StrategySetup:
    """Build a breakout setup with the given levels."""
    return StrategySetup(
        symbol="AAA",
        strategy="breakout",
        confidence=100.0,
        signal_strength=80.0,
        entry=entry,
        stop=stop,
        target=target,
        reward_risk=2.0,
    )


def _engine(
    data: HistoricalDataEngine,
    setups: dict[date, tuple[StrategySetup, ...]],
    *,
    config: BacktestConfig | None = None,
    regime: Regime | None = None,
) -> BacktestEngine:
    """Build a backtest engine over a look-ahead-guarded view of ``data``."""
    guard = LookaheadGuard(data)
    return BacktestEngine(
        data=guard,
        strategy_engine=_FixedStrategyEngine(setups),
        regime_engine=_FixedRegimeEngine(regime),
        calendar=_calendar(),
        config=config or BacktestConfig(),
    )


_MON = date(2025, 1, 6)  # Monday, a trading day
_TUE = date(2025, 1, 7)
_WED = date(2025, 1, 8)
_THU = date(2025, 1, 9)
_FRI = date(2025, 1, 10)


# -- Look-ahead guard ------------------------------------------------------


async def test_guard_blocks_future_candles() -> None:
    """Freezing the guard makes reading past the sim date a structural error."""
    data = _data()
    key = SeriesKey(symbol="AAA", exchange=Exchange.NSE, interval=Interval.ONE_DAY)
    await _seed(
        data,
        [_candle("AAA", _MON + timedelta(days=i), 100, 101, 99, 100) for i in range(5)],
    )
    guard = LookaheadGuard(data)

    guard.freeze_at(_MON)
    # A read bounded at the sim date is fine.
    ok = await guard.get_candles(
        key, datetime(2025, 1, 1, tzinfo=_TZ), BacktestEngine._eod(_MON)
    )
    assert len(ok) == 1
    # A read that reaches into the future raises rather than leaking data.
    with pytest.raises(LookaheadError):
        await guard.get_candles(
            key, datetime(2025, 1, 1, tzinfo=_TZ), datetime(2025, 1, 31, tzinfo=_TZ)
        )


async def test_guard_is_inert_until_frozen() -> None:
    """Before freezing, the guard passes reads straight through."""
    data = _data()
    key = SeriesKey(symbol="AAA", exchange=Exchange.NSE, interval=Interval.ONE_DAY)
    await _seed(data, [_candle("AAA", _MON, 100, 101, 99, 100)])
    guard = LookaheadGuard(data)

    candles = await guard.get_candles(
        key, datetime(2025, 1, 1, tzinfo=_TZ), datetime(2025, 12, 31, tzinfo=_TZ)
    )
    assert len(candles) == 1


# -- Fill realism: next-day open, costs, gaps ------------------------------


async def test_entry_fills_next_day_open_and_hits_target() -> None:
    """A signal at Monday's close fills at Tuesday's open, then hits target."""
    data = _data()
    await _seed(
        data,
        [
            _candle("AAA", _MON, 99, 100, 98, 100),
            _candle("AAA", _TUE, 100, 103, 99, 102),  # entry at open=100
            _candle("AAA", _WED, 105, 112, 101, 108),  # high 112 >= target 110
        ],
    )
    config = BacktestConfig(
        costs=BacktestCosts(slippage_pct=0, brokerage_pct=0, stt_pct=0)
    )
    engine = _engine(data, {_MON: (_setup(),)}, config=config)

    report = await engine.run(["AAA"], _MON, _FRI)

    assert report.trades == 1
    trade = report.closed_trades[0]
    assert trade.entry_date == _TUE  # next-day-open fill, not the signal bar
    assert trade.entry_price == 100.0
    assert trade.exit_date == _WED
    assert trade.exit_reason is ExitReason.TARGET
    assert trade.exit_kind is ExitKind.TARGET_HIT
    assert trade.signal_price == 100.0  # the setup's reference entry
    assert trade.exit_price == 110.0
    assert trade.size == 200.0  # 1% of 100k / 5.00 risk-per-share
    assert trade.net_pnl == 2000.0
    assert trade.r_multiple == 2.0
    assert report.expectancy_r == 2.0
    assert report.win_rate == 100.0


async def test_costs_and_slippage_are_applied() -> None:
    """Slippage moves both fills and charges are deducted from gross P&L."""
    data = _data()
    await _seed(
        data,
        [
            _candle("AAA", _MON, 99, 100, 98, 100),
            _candle("AAA", _TUE, 100, 103, 99, 102),
            _candle("AAA", _WED, 105, 112, 101, 108),
        ],
    )
    costs = BacktestCosts(slippage_pct=0.15, brokerage_pct=0.03, stt_pct=0.10)
    engine = _engine(data, {_MON: (_setup(),)}, config=BacktestConfig(costs=costs))

    report = await engine.run(["AAA"], _MON, _FRI)
    trade = report.closed_trades[0]

    assert trade.entry_price == 100.15  # 100 * (1 + 0.15%)
    assert trade.exit_price == 109.835  # 110 * (1 - 0.15%)
    gross = (109.835 - 100.15) * trade.size
    charges = (100.15 + 109.835) * trade.size * 0.13 / 100.0
    assert trade.gross_pnl == pytest.approx(round(gross, 2))
    assert trade.costs == pytest.approx(round(charges, 2))
    assert trade.net_pnl == pytest.approx(round(gross - charges, 2))
    assert trade.costs > 0 and trade.net_pnl < trade.gross_pnl


async def test_gap_past_stop_fills_at_open_not_stop() -> None:
    """A gap-down through the stop fills at the (worse) open, not the stop."""
    data = _data()
    await _seed(
        data,
        [
            _candle("AAA", _MON, 99, 100, 98, 100),
            _candle("AAA", _TUE, 100, 103, 99, 102),  # entry at open=100
            _candle("AAA", _WED, 90, 95, 88, 92),  # opens 90, below stop 95
        ],
    )
    config = BacktestConfig(
        costs=BacktestCosts(slippage_pct=0, brokerage_pct=0, stt_pct=0)
    )
    engine = _engine(data, {_MON: (_setup(),)}, config=config)

    report = await engine.run(["AAA"], _MON, _FRI)
    trade = report.closed_trades[0]

    assert trade.exit_reason is ExitReason.STOP
    assert trade.exit_kind is ExitKind.GAP_EXIT  # gap, not a clean stop
    assert trade.exit_price == 90.0  # filled at the gapped open, not the 95 stop
    assert trade.net_pnl == -2000.0  # worse than the -1000 a clean stop would give


async def test_no_position_on_entry_bar_and_timeout_close() -> None:
    """A setup with no stop/target hit is force-closed at the window end."""
    data = _data()
    candles = [_candle("AAA", _MON, 99, 100, 98, 100)]
    # Flat prices: never hits stop 95 or target 110 -> times out at window end.
    for day in (_TUE, _WED, _THU, _FRI):
        candles.append(_candle("AAA", day, 100, 101, 99, 100))
    await _seed(data, candles)
    config = BacktestConfig(
        costs=BacktestCosts(slippage_pct=0, brokerage_pct=0, stt_pct=0)
    )
    engine = _engine(data, {_MON: (_setup(),)}, config=config)

    report = await engine.run(["AAA"], _MON, _FRI)

    assert report.trades == 1
    trade = report.closed_trades[0]
    assert trade.entry_date == _TUE
    assert trade.exit_reason is ExitReason.TIMEOUT
    # Held < max_holding_days but the window ended: an end-of-backtest close,
    # not a holding-period timeout — the finer exit kind distinguishes them.
    assert trade.exit_kind is ExitKind.END_OF_BACKTEST
    assert trade.exit_date == _FRI  # closed at the last replayed day


async def test_regime_is_tagged_on_trades() -> None:
    """The regime classified at entry is recorded and segmented in the report."""
    data = _data()
    await _seed(
        data,
        [
            _candle("AAA", _MON, 99, 100, 98, 100),
            _candle("AAA", _TUE, 100, 103, 99, 102),
            _candle("AAA", _WED, 105, 112, 101, 108),
        ],
    )
    config = BacktestConfig(
        costs=BacktestCosts(slippage_pct=0, brokerage_pct=0, stt_pct=0)
    )
    engine = _engine(data, {_MON: (_setup(),)}, config=config, regime=Regime.BULL)

    report = await engine.run(["AAA"], _MON, _FRI)

    assert report.closed_trades[0].regime == "BULL"
    assert [r.regime for r in report.per_regime] == ["BULL"]


async def test_backtest_is_deterministic() -> None:
    """The same data and dates produce an identical report."""
    setups: dict[date, tuple[StrategySetup, ...]] = {_MON: (_setup(),)}
    seeds = [
        _candle("AAA", _MON, 99, 100, 98, 100),
        _candle("AAA", _TUE, 100, 103, 99, 102),
        _candle("AAA", _WED, 105, 112, 101, 108),
    ]

    async def _run() -> BacktestTrade:
        data = _data()
        await _seed(data, seeds)
        report = await _engine(data, setups).run(["AAA"], _MON, _FRI)
        return report.closed_trades[0]

    first = await _run()
    second = await _run()
    assert first.model_dump() == second.model_dump()


async def test_rejects_inverted_range() -> None:
    """An end date before the start is rejected."""
    engine = _engine(_data(), {})
    with pytest.raises(ValueError):
        await engine.run(["AAA"], _FRI, _MON)


# -- Expectancy maths (hand-computed inputs) -------------------------------


def _trade(
    net: float,
    r: float,
    *,
    strategy: str = "breakout",
    regime: str = "BULL",
    day: date,
    exit_kind: ExitKind = ExitKind.TARGET_HIT,
    hold: int = 1,
    signal_price: float = 100.0,
) -> BacktestTrade:
    """Build a closed trade carrying a known net P&L, R multiple and exit kind."""
    return BacktestTrade(
        symbol="AAA",
        strategy=strategy,
        regime=regime,
        entry_date=day - timedelta(days=hold),
        exit_date=day,
        signal_price=signal_price,
        entry_price=100.0,
        exit_price=max(1.0, 100.0 + net),
        stop_price=95.0,
        target_price=110.0,
        size=1.0,
        exit_reason=ExitReason.TARGET if net >= 0 else ExitReason.STOP,
        exit_kind=exit_kind,
        gross_pnl=net,
        costs=0.0,
        net_pnl=net,
        r_multiple=r,
    )


def test_metrics_match_hand_computed_values() -> None:
    """Win rate, profit factor, expectancy, drawdown and streak are exact."""
    trades = [
        _trade(100.0, 1.0, day=date(2025, 1, 6)),
        _trade(-50.0, -0.5, day=date(2025, 1, 7)),
        _trade(-80.0, -0.8, day=date(2025, 1, 8)),
        _trade(30.0, 0.3, day=date(2025, 1, 9)),
    ]

    assert metrics.win_rate(trades) == 50.0
    assert metrics.profit_factor(trades) == 1.0  # (100+30) / (50+80)
    assert metrics.expectancy_r(trades) == 0.0  # mean(1, -0.5, -0.8, 0.3)
    assert metrics.avg_win(trades) == 65.0  # mean(100, 30)
    assert metrics.avg_loss(trades) == -65.0  # mean(-50, -80)
    # Equity 1000 -> 1100 -> 1050 -> 970 -> 1000; deepest drop 130 from peak 1100.
    assert metrics.max_drawdown_pct(trades, 1000.0) == 11.82
    assert metrics.longest_losing_streak(trades) == 2


def test_metrics_segment_by_strategy_and_regime() -> None:
    """Per-strategy and per-regime breakdowns bucket and total correctly."""
    trades = [
        _trade(100.0, 1.0, strategy="breakout", regime="BULL", day=date(2025, 1, 6)),
        _trade(-40.0, -0.4, strategy="pullback", regime="BEAR", day=date(2025, 1, 7)),
        _trade(60.0, 0.6, strategy="breakout", regime="BULL", day=date(2025, 1, 8)),
    ]

    by_strategy = {s.strategy: s for s in metrics.per_strategy(trades)}
    assert by_strategy["breakout"].trades == 2
    assert by_strategy["breakout"].total_pnl == 160.0
    assert by_strategy["pullback"].total_pnl == -40.0

    by_regime = {r.regime: r for r in metrics.per_regime(trades)}
    assert by_regime["BULL"].trades == 2
    assert by_regime["BEAR"].win_rate == 0.0

    months = metrics.monthly_returns(trades, 1000.0)
    assert len(months) == 1
    assert months[0].month == "2025-01"
    assert months[0].net_pnl == 120.0


def test_metrics_handle_empty_input() -> None:
    """Metrics degrade to neutral zeros on no trades."""
    assert metrics.win_rate([]) == 0.0
    assert metrics.profit_factor([]) == 0.0
    assert metrics.expectancy_r([]) == 0.0
    assert metrics.max_drawdown_pct([], 1000.0) == 0.0
    assert metrics.longest_losing_streak([]) == 0
    assert metrics.exit_analysis([]).by_reason == ()


# -- Exit diagnostics (hand-computed inputs) -------------------------------


def test_exit_breakdown_accounts_by_kind() -> None:
    """Exit breakdown counts, wins, avg R and holding days per exit kind."""
    trades = [
        _trade(200.0, 2.0, day=date(2025, 1, 6), exit_kind=ExitKind.TARGET_HIT, hold=3),
        _trade(150.0, 1.5, day=date(2025, 1, 7), exit_kind=ExitKind.TARGET_HIT, hold=5),
        _trade(-100.0, -1.0, day=date(2025, 1, 8), exit_kind=ExitKind.STOP_HIT, hold=2),
        _trade(-90.0, -0.9, day=date(2025, 1, 9), exit_kind=ExitKind.GAP_EXIT, hold=1),
        _trade(50.0, 0.5, day=date(2025, 1, 10), exit_kind=ExitKind.TIMEOUT, hold=10),
        _trade(-30.0, -0.3, day=date(2025, 1, 13), exit_kind=ExitKind.TIMEOUT, hold=10),
    ]

    by_kind = {row.kind: row for row in metrics.exit_breakdown(trades)}
    assert by_kind["target_hit"].trades == 2
    assert by_kind["target_hit"].wins == 2
    assert by_kind["target_hit"].avg_r == 1.75  # mean(2.0, 1.5)
    assert by_kind["target_hit"].avg_holding_days == 4.0  # mean(3, 5)
    assert by_kind["timeout"].trades == 2
    assert by_kind["timeout"].total_pnl == 20.0  # +50 - 30


def test_exit_analysis_winner_distribution_and_timeouts() -> None:
    """Winner R buckets, timeout profitability and end-to-end wiring are correct."""
    trades = [
        _trade(200.0, 2.0, day=date(2025, 1, 6), exit_kind=ExitKind.TARGET_HIT),
        _trade(150.0, 1.5, day=date(2025, 1, 7), exit_kind=ExitKind.TARGET_HIT),
        _trade(50.0, 0.5, day=date(2025, 1, 10), exit_kind=ExitKind.TIMEOUT, hold=10),
        _trade(-30.0, -0.3, day=date(2025, 1, 13), exit_kind=ExitKind.TIMEOUT, hold=10),
        _trade(-20.0, -0.2, day=date(2025, 1, 14), exit_kind=ExitKind.END_OF_BACKTEST),
    ]

    analysis = metrics.exit_analysis(trades)
    dist = analysis.winner_r
    assert dist.winners == 3  # +200, +150, +50
    assert dist.reached_2r == 1  # only the R=2.0 winner realised the 2R target
    assert dist.between_1_and_2r == 1  # R=1.5
    assert dist.between_0_and_1r == 1  # R=0.5
    assert analysis.timeout_trades == 2
    assert analysis.timeout_profitable == 1  # only the +50 timeout was green


def test_exit_analysis_measures_slippage_and_bull_segment() -> None:
    """Entry slippage vs signal price and the BULL-only breakdown are reported."""
    trades = [
        _trade(
            10.0,
            0.1,
            day=date(2025, 1, 6),
            regime="BULL",
            exit_kind=ExitKind.STOP_HIT,
            signal_price=99.75,
        ),
        _trade(
            200.0,
            2.0,
            day=date(2025, 1, 7),
            regime="BEAR",
            exit_kind=ExitKind.TARGET_HIT,
            signal_price=99.75,
        ),
    ]

    analysis = metrics.exit_analysis(trades)
    # Fills at 100.0 vs a 99.75 signal: +0.25/share, ~+0.2506%.
    assert analysis.avg_entry_slippage == 0.25
    assert analysis.avg_entry_slippage_pct == round(0.25 / 99.75 * 100.0, 4)
    bull = {row.kind: row for row in analysis.bull_by_reason}
    assert set(bull) == {"stop_hit"}  # only the BULL trade is segmented in
    assert bull["stop_hit"].trades == 1


# -- Integration: the real strategy path under the guard -------------------


async def test_real_strategy_path_runs_without_lookahead() -> None:
    """The real StrategyEngine replays under the guard with future data present.

    Data extends well past the backtest window; a look-ahead leak would raise
    ``LookaheadError``. A clean run whose trades never exit after the window end
    proves the guard holds end-to-end through the exact live strategy code path.
    """
    data = _data()
    base = date(2024, 8, 1)
    # A long, mildly noisy uptrend so indicators/strategies have real history.
    aaa: list[Candle] = []
    nifty: list[Candle] = []
    for i in range(180):
        day = base + timedelta(days=i)
        price = 100.0 + i * 0.5 + (2.0 if i % 3 == 0 else 0.0)
        aaa.append(_candle("AAA", day, price, price + 2, price - 2, price + 1))
        nifty.append(_candle("NIFTY", day, price, price + 1, price - 1, price))
    await _seed(data, aaa)
    await _seed(data, nifty)

    guard = LookaheadGuard(data)
    indicators = IndicatorEngine(
        data, FakeClock(datetime(2025, 1, 6, tzinfo=_TZ)), get_indicator_registry()
    )
    engine = BacktestEngine(
        data=guard,
        strategy_engine=StrategyEngine(
            guard, indicators, FakeClock(datetime(2025, 1, 6, tzinfo=_TZ))
        ),
        regime_engine=MarketRegimeEngine(
            guard, indicators, FakeClock(datetime(2025, 1, 6, tzinfo=_TZ))
        ),
        calendar=_calendar(),
    )

    window_end = date(2025, 1, 17)
    report = await engine.run(["AAA"], date(2025, 1, 2), window_end)

    assert report.symbols == ("AAA",)
    assert all(trade.exit_date <= window_end for trade in report.closed_trades)
    # Determinism across a second, independent run.
    second = await engine.run(["AAA"], date(2025, 1, 2), window_end)
    assert report.model_dump() == second.model_dump()


# -- Live-vs-backtest parity and empty-store diagnostics -------------------


def _trading_days_seq(
    cal: MarketCalendarService, start: date, count: int
) -> list[date]:
    """Return the first ``count`` NSE trading days on/after ``start``."""
    days: list[date] = []
    day = start
    while len(days) < count:
        if cal.is_trading_day(day, Exchange.NSE):
            days.append(day)
        day += timedelta(days=1)
    return days


def _uptrend(symbol: str, days: Sequence[date]) -> list[Candle]:
    """A rising series with periodic pullbacks — fires real pullback setups."""
    base = 1000.0
    out: list[Candle] = []
    for i, day in enumerate(days):
        drift = 1.0 + 0.0015 * i
        close = base * drift * (1.0 + 0.05 * math.sin(i / 9.0))
        open_ = base * drift * (1.0 + 0.05 * math.sin((i - 1) / 9.0))
        high = max(open_, close) * 1.01
        low = min(open_, close) * 0.985
        volume = 100_000 + int(30_000 * (1.0 + math.sin(i / 5.0)))
        out.append(_candle(symbol, day, open_, high, low, close, volume))
    return out


def _clock() -> FakeClock:
    return FakeClock(datetime(2025, 1, 6, tzinfo=_TZ))


async def test_backtest_setups_match_live_strategy_on_same_date() -> None:
    """The replay's guarded strategy read yields the SAME setups as the live call.

    This is the parity guarantee: on any date, the look-ahead-guarded data view
    the backtest feeds the StrategyEngine produces exactly what a direct live
    StrategyEngine call produces on the same stored candles.
    """
    data = _data()
    cal = _calendar()
    days = _trading_days_seq(cal, date(2025, 1, 1), 200)
    await _seed(data, _uptrend("AAA", days))
    indicators = IndicatorEngine(data, _clock(), get_indicator_registry())
    live = StrategyEngine(data, indicators, _clock())

    def eod(day: date) -> datetime:
        return datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=_TZ)

    fire = None
    for day in days[120:190]:
        if (await live.analyze(["AAA"], end=eod(day))).setups:
            fire = day
            break
    assert fire is not None, "the seeded series should fire at least one setup"

    live_setups = (await live.analyze(["AAA"], end=eod(fire))).setups
    guard = LookaheadGuard(data)
    guard.freeze_at(fire)
    guarded = StrategyEngine(guard, indicators, _clock())
    backtest_setups = (await guarded.analyze(["AAA"], end=eod(fire))).setups

    assert backtest_setups  # non-empty
    assert backtest_setups == live_setups  # live vs backtest parity


async def test_backtest_opens_trades_on_seeded_history() -> None:
    """With candles present, the full replay opens a non-zero number of trades."""
    data = _data()
    cal = _calendar()
    days = _trading_days_seq(cal, date(2025, 1, 1), 200)
    await _seed(data, _uptrend("AAA", days))
    indicators = IndicatorEngine(data, _clock(), get_indicator_registry())
    guard = LookaheadGuard(data)
    engine = BacktestEngine(
        data=guard,
        strategy_engine=StrategyEngine(guard, indicators, _clock()),
        regime_engine=MarketRegimeEngine(guard, indicators, _clock()),
        calendar=cal,
        config=BacktestConfig(),
    )

    report = await engine.run(["AAA"], days[60], days[-1])

    assert report.trades > 0


async def test_backtest_warns_loudly_when_store_is_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An empty store yields 0 trades AND a loud, named warning (never silent)."""
    data = _data()  # nothing seeded
    cal = _calendar()
    indicators = IndicatorEngine(data, _clock(), get_indicator_registry())
    guard = LookaheadGuard(data)
    engine = BacktestEngine(
        data=guard,
        strategy_engine=StrategyEngine(guard, indicators, _clock()),
        regime_engine=MarketRegimeEngine(guard, indicators, _clock()),
        calendar=cal,
        config=BacktestConfig(),
    )

    with caplog.at_level(logging.WARNING):
        report = await engine.run(["AAA"], date(2025, 1, 6), date(2025, 1, 17))

    assert report.trades == 0
    assert "NO stored candles" in caplog.text
