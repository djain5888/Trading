"""An alternative Groww provider backed by the official ``growwapi`` SDK.

This is a second implementation of :class:`MarketDataProvider`, selectable with
``MARKET_DATA_PROVIDER=groww_sdk``. It authenticates with ``GrowwAPI(access_token=...)``
using the configured API key and implements the two operations the SDK path
needs: historical daily candles and the latest traded price for a watchlist.

The SDK is synchronous, so calls run in a worker thread to avoid blocking the
event loop. The SDK client is a small :class:`GrowwSDKClient` protocol so it can
be injected (and mocked) — the real ``growwapi`` package is imported lazily and
only when no client is supplied, keeping it an optional dependency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, TypeVar, cast

from app.config.broker import GrowwSettings
from app.core.clock import Clock, SystemClock
from app.core.logging import get_logger
from app.core.timezone import INDIA_TZ
from app.market.enums import Exchange, Interval
from app.market.models import (
    Candle,
    HistoricalData,
    MarketDepth,
    MarketStatusInfo,
    Quote,
    SymbolSearchResponse,
)
from app.providers.base import MarketDataProvider
from app.providers.exceptions import ProviderError

logger = get_logger(__name__)

_T = TypeVar("_T")

#: Equity cash segment token used by the Groww SDK.
_SEGMENT_CASH = "CASH"

#: Titan interval -> Groww ``interval_in_minutes``. Daily is the primary path.
_INTERVAL_MINUTES: dict[Interval, int] = {
    Interval.ONE_MINUTE: 1,
    Interval.FIVE_MINUTE: 5,
    Interval.FIFTEEN_MINUTE: 15,
    Interval.THIRTY_MINUTE: 30,
    Interval.ONE_HOUR: 60,
    Interval.ONE_DAY: 1440,
    Interval.ONE_WEEK: 10080,
}


class GrowwSDKClient(Protocol):
    """The subset of the ``growwapi.GrowwAPI`` surface this provider uses."""

    def get_historical_candle_data(
        self,
        *,
        trading_symbol: str,
        exchange: str,
        segment: str,
        start_time: str,
        end_time: str,
        interval_in_minutes: int,
    ) -> Mapping[str, Any]:
        """Return historical candles for one instrument."""
        ...

    def get_ltp(
        self, *, segment: str, exchange_trading_symbols: tuple[str, ...]
    ) -> Mapping[str, Any]:
        """Return the last traded price for one or more instruments."""
        ...


class GrowwSDKProvider(MarketDataProvider):
    """Groww market-data provider implemented over the ``growwapi`` SDK."""

    def __init__(
        self,
        settings: GrowwSettings,
        *,
        client: GrowwSDKClient | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Initialise the provider.

        Args:
            settings: Groww credentials and configuration.
            client: A pre-built SDK client (injected in tests); when omitted the
                real ``growwapi`` client is built lazily on first use.
            clock: Time source for quote timestamps.
        """
        self._settings = settings
        self._client = client
        self._clock = clock or SystemClock()

    def _sdk(self) -> GrowwSDKClient:
        """Return the SDK client, building the real one on first use."""
        if self._client is None:
            self._client = _build_growwapi(self._settings)
        return self._client

    async def _call(self, func: Callable[[], _T], *, what: str) -> _T:
        """Run a synchronous SDK call off the event loop, mapping failures."""
        try:
            return await asyncio.to_thread(func)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate any SDK failure
            raise ProviderError(f"Groww SDK {what} failed.", details=str(exc)) from exc

    async def get_historical_data(
        self,
        symbol: str,
        interval: Interval,
        start_date: datetime,
        end_date: datetime,
        exchange: Exchange = Exchange.NSE,
    ) -> HistoricalData:
        """Return historical OHLCV candles via the SDK."""
        trading_symbol = symbol.strip().upper()
        minutes = _INTERVAL_MINUTES.get(interval)
        if minutes is None:
            raise ProviderError(
                f"Groww SDK provider does not support interval '{interval.value}'."
            )
        payload = await self._call(
            lambda: self._sdk().get_historical_candle_data(
                trading_symbol=trading_symbol,
                exchange=exchange.value,
                segment=_SEGMENT_CASH,
                start_time=_format_time(start_date),
                end_time=_format_time(end_date),
                interval_in_minutes=minutes,
            ),
            what="historical candles",
        )
        return HistoricalData(
            symbol=trading_symbol,
            exchange=exchange,
            interval=interval,
            candles=_parse_candles(payload),
        )

    async def get_quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Quote:
        """Return a quote carrying the latest traded price via the SDK."""
        trading_symbol = symbol.strip().upper()
        key = f"{exchange.value}_{trading_symbol}"
        payload = await self._call(
            lambda: self._sdk().get_ltp(
                segment=_SEGMENT_CASH, exchange_trading_symbols=(key,)
            ),
            what="LTP",
        )
        return Quote(
            symbol=trading_symbol,
            exchange=exchange,
            last_price=_ltp_value(payload, key),
            timestamp=self._clock.now(),
        )

    async def get_quotes(
        self, symbols: Sequence[str], exchange: Exchange = Exchange.NSE
    ) -> list[Quote]:
        """Return LTP quotes for a watchlist, isolating per-symbol failures."""
        quotes: list[Quote] = []
        for symbol in symbols:
            try:
                quotes.append(await self.get_quote(symbol, exchange))
            except ProviderError as exc:
                logger.warning("Groww SDK LTP failed for %s: %s", symbol, exc)
        return quotes

    async def get_market_status(
        self, exchange: Exchange = Exchange.NSE
    ) -> MarketStatusInfo:
        """Not supported by the SDK provider."""
        raise ProviderError("Market status is not supported by the Groww SDK provider.")

    async def search_symbol(
        self, query: str, exchange: Exchange | None = None
    ) -> SymbolSearchResponse:
        """Not supported by the SDK provider."""
        raise ProviderError("Symbol search is not supported by the Groww SDK provider.")

    async def get_market_depth(
        self, symbol: str, exchange: Exchange = Exchange.NSE
    ) -> MarketDepth:
        """Not supported by the SDK provider."""
        raise ProviderError("Market depth is not supported by the Groww SDK provider.")


# -- Response mapping ------------------------------------------------------


def _parse_candles(payload: Mapping[str, Any]) -> tuple[Candle, ...]:
    """Map a SDK historical payload into canonical candles."""
    rows = payload.get("candles")
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise ProviderError("Groww SDK historical payload had no 'candles' list.")
    try:
        return tuple(_candle(row) for row in rows)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ProviderError(
            "Malformed Groww SDK candle row.", details=str(exc)
        ) from exc


def _candle(row: Any) -> Candle:
    """Map one SDK candle row (``[epoch, o, h, l, c, v]`` or a mapping)."""
    if isinstance(row, Mapping):
        timestamp = row.get("timestamp", row.get("time"))
        values = (row["open"], row["high"], row["low"], row["close"], row["volume"])
    else:
        timestamp = row[0]
        values = (row[1], row[2], row[3], row[4], row[5])
    open_, high, low, close, volume = values
    return Candle(
        timestamp=_to_datetime(timestamp),
        open=_dec(open_),
        high=_dec(high),
        low=_dec(low),
        close=_dec(close),
        volume=int(volume),
    )


def _ltp_value(payload: Mapping[str, Any], key: str) -> Decimal:
    """Extract the last traded price for ``key`` from an LTP payload."""
    if key in payload:
        return _dec(payload[key])
    values = list(payload.values())
    if len(values) == 1:
        return _dec(values[0])
    raise ProviderError(f"Groww SDK LTP response missing '{key}'.")


def _to_datetime(value: Any) -> datetime:
    """Convert an epoch, ISO string or datetime into a tz-aware datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, bool):  # guard: bool is an int subclass
        raise ValueError("Timestamp must not be a boolean.")
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=INDIA_TZ)
    return datetime.fromisoformat(str(value))


def _dec(value: Any) -> Decimal:
    """Parse a Decimal from a string or number."""
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal '{value}'.") from exc


def _format_time(value: datetime) -> str:
    """Format a datetime as the SDK's ``YYYY-MM-DD HH:MM:SS`` in IST."""
    aware = value if value.tzinfo else value.replace(tzinfo=INDIA_TZ)
    return aware.astimezone(INDIA_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _build_growwapi(settings: GrowwSettings) -> GrowwSDKClient:
    """Build the real ``growwapi`` client, or fail clearly if it is absent."""
    try:
        from growwapi import GrowwAPI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ProviderError(
            "The 'growwapi' SDK is not installed; run 'pip install growwapi' or "
            "use MARKET_DATA_PROVIDER=groww.",
            details=str(exc),
        ) from exc
    if not settings.api_key:
        raise ProviderError("Groww API key is not configured.")
    return cast("GrowwSDKClient", GrowwAPI(access_token=settings.api_key))
