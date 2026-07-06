"""Maps Groww JSON payloads into Titan domain models.

This is the only place raw Groww JSON is interpreted; every method returns a
domain model and raises :class:`ProviderError` on a malformed payload. No raw
JSON structure escapes the provider boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal, InvalidOperation
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from pydantic import ValidationError

from app.market.enums import Exchange, Interval
from app.market.models import (
    Candle,
    DepthLevel,
    HistoricalData,
    MarketDepth,
    MarketStatusInfo,
    Quote,
    SymbolSearchResponse,
    SymbolSearchResult,
)
from app.providers.exceptions import ProviderError
from app.providers.groww.endpoints import STATUS_TOKENS

_P = ParamSpec("_P")
_R = TypeVar("_R")

#: Exceptions that indicate a malformed payload rather than a domain error.
_MALFORMED = (KeyError, ValueError, TypeError, IndexError, ValidationError)


def _mapping(what: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Wrap a parse method so structural errors become :class:`ProviderError`."""

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return func(*args, **kwargs)
            except _MALFORMED as exc:
                raise ProviderError(
                    f"Malformed Groww {what} payload.", details=str(exc)
                ) from exc

        return wrapper

    return decorator


class GrowwResponseMapper:
    """Translates Groww payloads into domain models."""

    @_mapping("market status")
    def parse_market_status(
        self, payload: Mapping[str, Any], exchange: Exchange
    ) -> MarketStatusInfo:
        """Parse a market-status payload."""
        token = str(payload["status"]).strip().lower()
        status = STATUS_TOKENS.get(token)
        if status is None:
            raise ProviderError(f"Unknown Groww market status '{token}'.")
        return MarketStatusInfo(
            exchange=exchange,
            status=status,
            timestamp=_parse_dt(payload["timestamp"]),
            message=_opt_str(payload.get("message")),
        )

    @_mapping("quote")
    def parse_quote(self, payload: Mapping[str, Any], exchange: Exchange) -> Quote:
        """Parse a single-quote payload."""
        data = payload.get("data", payload)
        if not isinstance(data, Mapping):
            raise ProviderError("Groww quote payload was not an object.")
        return _quote(data, exchange)

    @_mapping("quotes")
    def parse_quotes(
        self, payload: Mapping[str, Any], exchange: Exchange
    ) -> list[Quote]:
        """Parse a multi-quote payload."""
        rows = _require_list(payload.get("data", payload.get("quotes")), "quotes")
        return [_quote(row, exchange) for row in rows]

    @_mapping("historical data")
    def parse_historical(
        self,
        payload: Mapping[str, Any],
        symbol: str,
        exchange: Exchange,
        interval: Interval,
    ) -> HistoricalData:
        """Parse a historical-candles payload."""
        rows = _require_list(payload.get("candles", payload.get("data")), "candles")
        return HistoricalData(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            candles=tuple(_candle(row) for row in rows),
        )

    @_mapping("search")
    def parse_search(
        self, payload: Mapping[str, Any], query: str
    ) -> SymbolSearchResponse:
        """Parse a symbol-search payload."""
        rows = _require_list(payload.get("results", payload.get("data")), "results")
        return SymbolSearchResponse(
            query=query, results=tuple(_search_result(row) for row in rows)
        )

    @_mapping("market depth")
    def parse_market_depth(
        self, payload: Mapping[str, Any], exchange: Exchange
    ) -> MarketDepth:
        """Parse a market-depth payload."""
        return MarketDepth(
            symbol=str(payload["symbol"]),
            exchange=exchange,
            bids=tuple(_depth_level(row) for row in payload.get("bids", [])),
            asks=tuple(_depth_level(row) for row in payload.get("asks", [])),
            timestamp=_parse_dt(payload["timestamp"]),
        )


# -- Row helpers -----------------------------------------------------------


def _quote(row: Mapping[str, Any], exchange: Exchange) -> Quote:
    """Parse one quote row."""
    return Quote(
        symbol=str(row["symbol"]),
        exchange=exchange,
        last_price=_dec(row["last_price"]),
        open=_opt_dec(row.get("open")),
        high=_opt_dec(row.get("high")),
        low=_opt_dec(row.get("low")),
        previous_close=_opt_dec(row.get("previous_close")),
        volume=_opt_int(row.get("volume")),
        timestamp=_parse_dt(row["timestamp"]),
    )


def _candle(row: Mapping[str, Any]) -> Candle:
    """Parse one OHLCV candle row."""
    return Candle(
        timestamp=_parse_dt(row["timestamp"]),
        open=_dec(row["open"]),
        high=_dec(row["high"]),
        low=_dec(row["low"]),
        close=_dec(row["close"]),
        volume=int(row["volume"]),
    )


def _depth_level(row: Mapping[str, Any]) -> DepthLevel:
    """Parse one order-book level."""
    return DepthLevel(
        price=_dec(row["price"]),
        quantity=int(row["quantity"]),
        orders=_opt_int(row.get("orders")),
    )


def _search_result(row: Mapping[str, Any]) -> SymbolSearchResult:
    """Parse one search result."""
    return SymbolSearchResult(
        symbol=str(row["symbol"]),
        name=str(row["name"]),
        exchange=Exchange(str(row["exchange"])),
        instrument_type=str(row.get("instrument_type", "EQ")),
    )


def _require_list(value: Any, field: str) -> Sequence[Mapping[str, Any]]:
    """Return ``value`` as a list of objects, or raise for a bad shape."""
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ProviderError(f"Groww payload had no '{field}' list.")
    return value


def _parse_dt(value: Any) -> datetime:
    """Parse an ISO-8601 timestamp string."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _dec(value: Any) -> Decimal:
    """Parse a Decimal from a string or number."""
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal '{value}'.") from exc


def _opt_dec(value: Any) -> Decimal | None:
    """Parse an optional Decimal."""
    return None if value is None else _dec(value)


def _opt_int(value: Any) -> int | None:
    """Parse an optional integer."""
    return None if value is None else int(value)


def _opt_str(value: Any) -> str | None:
    """Parse an optional string."""
    return None if value is None else str(value)
