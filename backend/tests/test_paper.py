"""Unit tests for the paper-trading engine."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from app.core.clock import FakeClock
from app.core.timezone import INDIA_TZ
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.memory import InMemoryCandleRepository
from app.market.historical.models import Candle
from app.market.historical.validation import ValidationEngine
from app.paper.engine import PaperTradingEngine
from app.paper.memory import InMemoryPaperTradeRepository
from app.paper.models import ExitReason, PaperConfig, PaperTrade, TradeStatus
from app.paper.repository import PaperTradeRepository
from app.strategy.models import StrategySetup

_END = datetime(2025, 1, 20, 15, 30, tzinfo=INDIA_TZ)
_DAY = timedelta(days=1)


def _engine(
    config: PaperConfig | None = None,
) -> tuple[PaperTradingEngine, HistoricalDataEngine, PaperTradeRepository]:
    repo = InMemoryPaperTradeRepository()
    data = HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())
    engine = PaperTradingEngine(repo, data, FakeClock(_END), config or PaperConfig())
    return engine, data, repo


async def _bar(
    data: HistoricalDataEngine,
    symbol: str,
    at: datetime,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> None:
    await data.import_candles(
        [
            Candle(
                symbol=symbol,
                exchange=Exchange.NSE,
                interval=Interval.ONE_DAY,
                timestamp=at,
                open=Decimal(f"{open_:.2f}"),
                high=Decimal(f"{high:.2f}"),
                low=Decimal(f"{low:.2f}"),
                close=Decimal(f"{close:.2f}"),
                volume=1000,
            )
        ]
    )


def _setup(
    symbol: str,
    *,
    entry: float,
    stop: float,
    target: float,
    strategy: str = "breakout",
    confidence: float = 80.0,
) -> StrategySetup:
    return StrategySetup(
        symbol=symbol,
        strategy=strategy,
        confidence=confidence,
        signal_strength=70.0,
        entry=entry,
        stop=stop,
        target=target,
        reward_risk=2.0,
    )


# -- Position sizing + opening ---------------------------------------------


async def test_open_and_position_sizing() -> None:
    """size = capital * risk_pct / (entry - stop); a trade opens from a setup."""
    engine, data, _ = _engine()  # 100k capital, 1% risk
    await _bar(data, "AAA", _END - _DAY, open_=100, high=101, low=99, close=100)

    report = await engine.run([_setup("AAA", entry=100, stop=90, target=120)], end=_END)

    assert len(report.open_positions) == 1
    trade = report.open_positions[0]
    assert trade.size == 100.0  # 100000 * 0.01 / (100 - 90) = 100
    assert trade.status is TradeStatus.OPEN
    assert trade.entry_price == 100.0


async def test_max_positions_limit() -> None:
    """No more than max_positions concurrent trades are opened."""
    engine, data, _ = _engine(PaperConfig(max_positions=2))
    for symbol in ("AAA", "BBB", "CCC"):
        await _bar(data, symbol, _END - _DAY, open_=100, high=101, low=99, close=100)

    report = await engine.run(
        [
            _setup("AAA", entry=100, stop=90, target=120),
            _setup("BBB", entry=100, stop=90, target=120),
            _setup("CCC", entry=100, stop=90, target=120),
        ],
        end=_END,
    )

    assert len(report.open_positions) == 2  # third setup exceeds the limit


# -- Exit logic ------------------------------------------------------------


async def _open_one(engine: PaperTradingEngine, data: HistoricalDataEngine) -> None:
    await _bar(data, "AAA", _END - 3 * _DAY, open_=100, high=101, low=99, close=100)
    await engine.run(
        [_setup("AAA", entry=100, stop=90, target=120)], end=_END - 3 * _DAY
    )


async def test_stop_hit_closes_at_stop() -> None:
    """A bar whose low breaches the stop closes the trade at the stop price."""
    engine, data, _ = _engine()
    await _open_one(engine, data)
    await _bar(data, "AAA", _END - 2 * _DAY, open_=95, high=96, low=89, close=92)

    report = await engine.run([], end=_END - 2 * _DAY)

    assert not report.open_positions
    trade = report.closed_trades[0]
    assert trade.exit_reason is ExitReason.STOP
    assert trade.exit_price == 90.0
    assert trade.pnl == (90.0 - 100.0) * 100.0  # -1000


async def test_target_hit_closes_at_target() -> None:
    """A bar whose high reaches the target closes the trade at the target."""
    engine, data, _ = _engine()
    await _open_one(engine, data)
    await _bar(data, "AAA", _END - 2 * _DAY, open_=110, high=121, low=108, close=119)

    report = await engine.run([], end=_END - 2 * _DAY)

    trade = report.closed_trades[0]
    assert trade.exit_reason is ExitReason.TARGET
    assert trade.exit_price == 120.0
    assert trade.pnl == (120.0 - 100.0) * 100.0  # +2000


async def test_timeout_closes_at_close() -> None:
    """Holding past max_holding_days force-closes at the latest close."""
    engine, data, _ = _engine(PaperConfig(max_holding_days=1))
    await _open_one(engine, data)  # opened at _END - 3d
    # A later bar that does not hit stop/target, held > 1 day.
    await _bar(data, "AAA", _END - _DAY, open_=104, high=108, low=101, close=105)

    report = await engine.run([], end=_END - _DAY)

    trade = report.closed_trades[0]
    assert trade.exit_reason is ExitReason.TIMEOUT
    assert trade.exit_price == 105.0


async def test_missing_price_holds_position() -> None:
    """With no new candle since entry, the position is held, not closed."""
    engine, data, _ = _engine()
    await _open_one(engine, data)

    report = await engine.run([], end=_END - 3 * _DAY)  # no new bar imported

    assert len(report.open_positions) == 1
    assert not report.closed_trades


# -- Persistence -----------------------------------------------------------


async def test_persistence_across_runs() -> None:
    """A second engine sharing the repo sees the first run's open trade."""
    repo = InMemoryPaperTradeRepository()
    data = HistoricalDataEngine(InMemoryCandleRepository(), ValidationEngine())
    await _bar(data, "AAA", _END - _DAY, open_=100, high=101, low=99, close=100)

    first = PaperTradingEngine(repo, data, FakeClock(_END))
    await first.run([_setup("AAA", entry=100, stop=90, target=120)], end=_END)

    second = PaperTradingEngine(repo, data, FakeClock(_END))
    report = await second.report(_END)

    assert len(report.open_positions) == 1
    assert report.open_positions[0].symbol == "AAA"


async def test_idempotent_no_duplicate_open() -> None:
    """Re-running with the same setup and no new bar does not re-open."""
    engine, data, repo = _engine()
    await _bar(data, "AAA", _END - _DAY, open_=100, high=101, low=99, close=100)
    setup = _setup("AAA", entry=100, stop=90, target=120)

    await engine.run([setup], end=_END)
    await engine.run([setup], end=_END)

    assert len(await repo.all()) == 1  # deterministic, one trade only


# -- Performance metrics ---------------------------------------------------


def _closed(strategy: str, pnl: float, ident: str) -> PaperTrade:
    exit_price = 120.0 if pnl > 0 else 90.0
    return PaperTrade(
        id=ident,
        symbol="AAA",
        strategy=strategy,
        exchange=Exchange.NSE,
        interval=Interval.ONE_DAY,
        entry_price=100.0,
        stop_price=90.0,
        target_price=130.0,
        size=1.0,
        opened_at=_END - 5 * _DAY,
        status=TradeStatus.CLOSED,
        last_price=exit_price,
        pnl=pnl,
        exit_price=exit_price,
        exit_reason=ExitReason.TARGET if pnl > 0 else ExitReason.STOP,
        closed_at=_END,
    )


async def test_win_rate_and_profit_factor() -> None:
    """Win rate, averages and profit factor come from the closed trades."""
    engine, _, repo = _engine()
    await repo.upsert(_closed("breakout", 200.0, "t1"))
    await repo.upsert(_closed("breakout", 100.0, "t2"))
    await repo.upsert(_closed("momentum", -50.0, "t3"))

    report = await engine.report(_END)

    assert report.wins == 2
    assert report.losses == 1
    assert report.win_rate == round(200.0 / 3, 1)  # 66.7
    assert report.avg_win == 150.0
    assert report.avg_loss == -50.0
    assert report.profit_factor == 6.0  # 300 / 50
    assert report.realized_pnl == 250.0
    by_strategy = {s.strategy: s for s in report.per_strategy}
    assert by_strategy["breakout"].wins == 2
    assert by_strategy["breakout"].total_pnl == 300.0
    assert by_strategy["momentum"].total_pnl == -50.0
