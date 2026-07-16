"""The Groww provider backed by the official ``growwapi`` SDK.

The default :class:`MarketDataProvider` (``MARKET_DATA_PROVIDER=groww_sdk``).
It builds ``GrowwAPI(access_token=...)`` and implements the two operations the
platform needs: historical daily candles and the latest traded price for a
watchlist.

Auth resolves an access token in one of two ways (TOTP preferred, since Groww
keys reset daily at 6 AM):

* TOTP: when ``GROWW_TOTP_TOKEN`` and ``GROWW_TOTP_SECRET`` are set, a fresh
  token is minted per run via ``pyotp`` + ``GrowwAPI.get_access_token``.
* Direct: otherwise ``GROWW_ACCESS_TOKEN`` is used as-is.

The SDK is synchronous, so calls (and the lazy client build) run in a worker
thread. ``growwapi``/``pyotp`` are imported lazily so they stay optional, and
the client is a :class:`GrowwSDKClient` protocol so tests inject a fake. An
"Access forbidden" error maps to :class:`AuthenticationError` so ``titan health``
flags an expired or unapproved key.
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
from app.providers.exceptions import AuthenticationError, ProviderError

logger = get_logger(__name__)

_T = TypeVar("_T")

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

    SEGMENT_CASH: str
    EXCHANGE_NSE: str
    EXCHANGE_BSE: str

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
        self, *, segment: str, exchange_trading_symbols: str
    ) -> Mapping[str, Any]:
        """Return ``{"<EXCH>_<SYM>": price}`` for one instrument."""
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
        """Run a synchronous SDK call off the event loop, mapping failures.

        Raises:
            AuthenticationError: If Groww reports the key is forbidden.
            ProviderError: For any other SDK failure.
        """
        try:
            return await asyncio.to_thread(func)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate any SDK failure
            if _is_forbidden(exc):
                raise AuthenticationError(
                    "Groww access forbidden — API key expired or not approved "
                    "(Groww keys reset daily at 6 AM).",
                    details=str(exc),
                ) from exc
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

        def _run() -> Mapping[str, Any]:
            client = self._sdk()
            return client.get_historical_candle_data(
                trading_symbol=trading_symbol,
                exchange=_exchange_token(client, exchange),
                segment=client.SEGMENT_CASH,
                start_time=_format_time(start_date),
                end_time=_format_time(end_date),
                interval_in_minutes=minutes,
            )

        payload = await self._call(_run, what="historical candles")
        return HistoricalData(
            symbol=trading_symbol,
            exchange=exchange,
            interval=interval,
            candles=_parse_candles(payload),
        )

    async def get_quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Quote:
        """Return a quote carrying the latest traded price (LTP) via the SDK."""
        trading_symbol = symbol.strip().upper()
        key = f"{exchange.value}_{trading_symbol}"

        def _run() -> Mapping[str, Any]:
            client = self._sdk()
            return client.get_ltp(
                segment=client.SEGMENT_CASH, exchange_trading_symbols=key
            )

        payload = await self._call(_run, what="LTP")
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


# -- Error classification --------------------------------------------------


def _is_forbidden(exc: Exception) -> bool:
    """Return whether an SDK exception signals a forbidden (expired) key."""
    return "forbidden" in str(exc).lower()


# -- Exchange / response mapping -------------------------------------------


def _exchange_token(client: GrowwSDKClient, exchange: Exchange) -> str:
    """Map a Titan exchange to the SDK's exchange constant."""
    if exchange is Exchange.NSE:
        return client.EXCHANGE_NSE
    if exchange is Exchange.BSE:
        return client.EXCHANGE_BSE
    raise ProviderError(
        f"Groww SDK provider does not support exchange '{exchange.value}'."
    )


def _parse_candles(payload: Mapping[str, Any]) -> tuple[Candle, ...]:
    """Map a SDK historical payload into canonical candles.

    Index series (e.g. NIFTY) differ from equities — they may carry no/zero
    volume and a shorter row. Rows are mapped defensively: a genuinely malformed
    row is skipped (never fails the whole series), volume defaults to zero.
    """
    rows = payload.get("candles")
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise ProviderError("Groww SDK historical payload had no 'candles' list.")
    candles: list[Candle] = []
    skipped = 0
    for row in rows:
        candle = _candle(row)
        if candle is None:
            skipped += 1
        else:
            candles.append(candle)
    if skipped:
        logger.debug("Skipped %d malformed Groww SDK candle row(s).", skipped)
    return tuple(candles)


def _candle(row: Any) -> Candle | None:
    """Map one SDK candle row into a Candle, or ``None`` if it is malformed.

    Accepts ``[epoch, o, h, l, c]`` or ``[epoch, o, h, l, c, v]`` (or a mapping).
    Volume is optional and defaults to zero for index rows.
    """
    try:
        if isinstance(row, Mapping):
            timestamp = row.get("timestamp", row.get("time"))
            open_, high, low, close = (
                row["open"],
                row["high"],
                row["low"],
                row["close"],
            )
            volume = row.get("volume", 0)
        else:
            if len(row) < 5:
                return None
            timestamp = row[0]
            open_, high, low, close = row[1], row[2], row[3], row[4]
            volume = row[5] if len(row) > 5 else 0
        if timestamp is None:
            return None
        return Candle(
            timestamp=_to_datetime(timestamp),
            open=_dec(open_),
            high=_dec(high),
            low=_dec(low),
            close=_dec(close),
            volume=_volume(volume),
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _volume(value: Any) -> int:
    """Parse a tolerant, non-negative integer volume (defaults to 0)."""
    if value is None:
        return 0
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def _ltp_value(payload: Mapping[str, Any], key: str) -> Decimal:
    """Extract the last traded price for ``key`` from an LTP payload."""
    if key in payload:
        return _dec(payload[key])
    values = list(payload.values())
    if len(values) == 1:
        return _dec(values[0])
    raise ProviderError(f"Groww SDK LTP response missing '{key}'.")


def _to_datetime(value: Any) -> datetime:
    """Convert an epoch (seconds), ISO string or datetime into a tz-aware IST time."""
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


# -- Client construction ---------------------------------------------------


def _resolve_access_token(settings: GrowwSettings, groww_api: Any) -> str:
    """Resolve an access token, preferring TOTP regeneration.

    Raises:
        ProviderError: If no SDK credentials are configured.
    """
    token = settings.totp_token
    secret = settings.totp_secret
    if token is not None and secret is not None:
        import pyotp  # type: ignore[import-not-found]

        code = pyotp.TOTP(secret).now()
        return str(groww_api.get_access_token(api_key=token, totp=code))
    if settings.access_token is not None:
        return settings.access_token
    raise ProviderError(
        "Groww SDK auth is not configured: set GROWW_ACCESS_TOKEN, or "
        "GROWW_TOTP_TOKEN + GROWW_TOTP_SECRET."
    )


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
    access_token = _resolve_access_token(settings, GrowwAPI)
    # The SDK constructor takes the token positionally: GrowwAPI(access_token).
    return cast("GrowwSDKClient", GrowwAPI(access_token))
