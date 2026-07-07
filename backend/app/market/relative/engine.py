"""The relative-strength engine.

Scores each stock's strength versus the market (NIFTY) and versus its sector so
ranking reflects leaders, not just raw signals. Returns are read through the
:class:`HistoricalDataEngine`; no new indicator maths is introduced.

A missing index or sector neutralises that component (never fails), and a symbol
with too little history is excluded from leadership rather than failing the run.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from statistics import fmean

from app.core.clock import Clock
from app.core.logging import get_logger
from app.market.enums import Exchange, Interval
from app.market.historical.engine import HistoricalDataEngine
from app.market.historical.models import SeriesKey
from app.market.relative.models import (
    NEUTRAL_RS,
    RelativeStrength,
    RSConfig,
    RSReport,
)

logger = get_logger(__name__)


def _percentile_ranks(raw: dict[str, float]) -> dict[str, float]:
    """Percentile-rank each value in ``[0, 100]`` (highest value ~100)."""
    total = len(raw)
    return {
        symbol: 100.0 * sum(1 for other in raw.values() if other <= value) / total
        for symbol, value in raw.items()
    }


class RelativeStrengthEngine:
    """Scores symbols by their return relative to the market and their sector."""

    def __init__(
        self,
        data_engine: HistoricalDataEngine,
        clock: Clock,
        config: RSConfig | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            data_engine: Source of stored candles (read-only).
            clock: Time source for the report timestamp and default range.
            config: Tuning and sector metadata; defaults to NIFTY/daily.
        """
        self._data = data_engine
        self._clock = clock
        self._config = config or RSConfig()

    @property
    def config(self) -> RSConfig:
        """Return the engine's configuration."""
        return self._config

    async def analyze(
        self,
        symbols: Sequence[str],
        *,
        exchange: Exchange = Exchange.NSE,
        interval: Interval = Interval.ONE_DAY,
        end: datetime | None = None,
    ) -> RSReport:
        """Score relative strength for ``symbols``.

        Args:
            symbols: Symbols to score.
            exchange: Exchange the symbols trade on.
            interval: Candle interval to read.
            end: Inclusive range end; defaults to the clock's now.

        Returns:
            An :class:`RSReport`; ``available`` is ``False`` only when no symbol
            has enough history to score.
        """
        now = end or self._clock.now()
        start = now - timedelta(days=self._config.history_days)
        market_return = await self._return(
            self._config.index_symbol, self._config.index_exchange, interval, start, now
        )
        returns: dict[str, float] = {}
        for symbol in symbols:
            value = await self._return(symbol, exchange, interval, start, now)
            if value is not None:
                returns[symbol.strip().upper()] = value
        if not returns:
            return RSReport.unavailable(
                generated_at=now,
                detail="No symbol had enough history — relative strength unavailable.",
            )

        market_pct = self._market_percentiles(returns, market_return)
        sector_pct = self._sector_percentiles(returns)
        entries = self._build_entries(returns, market_pct, sector_pct)
        leaders = tuple(entry for entry in entries if entry.leader)
        return RSReport(
            available=True,
            market_available=market_return is not None,
            entries=entries,
            leaders=leaders,
            detail=(
                f"{len(entries)} symbol(s) scored, {len(leaders)} leader(s)"
                f"{'' if market_return is not None else ' (market neutral)'}."
            ),
            generated_at=now,
        )

    # -- Percentile components ---------------------------------------------

    def _market_percentiles(
        self, returns: dict[str, float], market_return: float | None
    ) -> dict[str, float] | None:
        """Percentile-rank each symbol's excess return over the market."""
        if market_return is None:
            return None
        return _percentile_ranks(
            {symbol: value - market_return for symbol, value in returns.items()}
        )

    def _sector_percentiles(self, returns: dict[str, float]) -> dict[str, float] | None:
        """Percentile-rank each symbol's excess return over its sector."""
        buckets: dict[str, list[float]] = defaultdict(list)
        for symbol, value in returns.items():
            sector = self._config.sector_map.get(symbol)
            if sector is not None:
                buckets[sector].append(value)
        if not buckets:
            return None
        sector_return = {sector: fmean(values) for sector, values in buckets.items()}
        raw = {
            symbol: value - sector_return[self._config.sector_map[symbol]]
            for symbol, value in returns.items()
            if symbol in self._config.sector_map
        }
        return _percentile_ranks(raw) if raw else None

    def _build_entries(
        self,
        returns: dict[str, float],
        market_pct: dict[str, float] | None,
        sector_pct: dict[str, float] | None,
    ) -> tuple[RelativeStrength, ...]:
        """Assemble ranked per-symbol relative-strength entries."""
        entries: list[RelativeStrength] = []
        for symbol in returns:
            rs_market = market_pct[symbol] if market_pct else NEUTRAL_RS
            rs_sector = sector_pct.get(symbol, NEUTRAL_RS) if sector_pct else NEUTRAL_RS
            leader = (
                rs_market >= self._config.leader_percentile
                and rs_sector >= self._config.leader_percentile
            )
            entries.append(
                RelativeStrength(
                    symbol=symbol,
                    rs_vs_market=round(rs_market, 1),
                    rs_vs_sector=round(rs_sector, 1),
                    composite=round((rs_market + rs_sector) / 2, 1),
                    leader=leader,
                )
            )
        entries.sort(key=lambda entry: entry.composite, reverse=True)
        return tuple(entries)

    # -- Returns -----------------------------------------------------------

    async def _return(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> float | None:
        """Return the percent change over the lookback, or ``None`` if too short."""
        key = SeriesKey(symbol=symbol, exchange=exchange, interval=interval)
        candles = await self._data.get_candles(key, start, end)
        if len(candles) < self._config.required_candles:
            return None
        latest = float(candles[-1].close)
        base = float(candles[-1 - self._config.lookback_periods].close)
        if base == 0:
            return None
        return (latest / base - 1.0) * 100.0
