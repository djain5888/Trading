"""Unit tests for the MarketDataService gateway."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from app.market.enums import Exchange, Interval, MarketStatus
from app.market.models import (
    Candle,
    HistoricalData,
    MarketDepth,
    MarketStatusInfo,
    Quote,
    SymbolSearchResponse,
)
from app.providers.base import MarketDataProvider
from app.providers.exceptions import InvalidSymbolError, NetworkError, ProviderError
from app.services.market_data import MarketDataService


def _quote(symbol: str, exchange: Exchange = Exchange.NSE, price: str = "100") -> Quote:
    """Build a minimal quote for tests."""
    return Quote(
        symbol=symbol,
        exchange=exchange,
        last_price=Decimal(price),
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )


class FakeProvider(MarketDataProvider):
    """Controllable in-memory provider for exercising the service."""

    def __init__(self) -> None:
        self.status_calls = 0
        self.quote_calls = 0
        self.market_status = MarketStatus.OPEN
        self.next_quote: Quote | None = None
        self.next_quotes: list[Quote] | None = None
        self.next_history: HistoricalData | None = None
        self.raise_error: ProviderError | None = None

    async def get_market_status(
        self, exchange: Exchange = Exchange.NSE
    ) -> MarketStatusInfo:
        self.status_calls += 1
        if self.raise_error is not None:
            raise self.raise_error
        return MarketStatusInfo(
            exchange=exchange,
            status=self.market_status,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        )

    async def get_quote(self, symbol: str, exchange: Exchange = Exchange.NSE) -> Quote:
        self.quote_calls += 1
        if self.raise_error is not None:
            raise self.raise_error
        return self.next_quote or _quote(symbol, exchange)

    async def get_quotes(
        self, symbols: Sequence[str], exchange: Exchange = Exchange.NSE
    ) -> list[Quote]:
        if self.raise_error is not None:
            raise self.raise_error
        if self.next_quotes is not None:
            return self.next_quotes
        return [_quote(s, exchange) for s in symbols]

    async def get_historical_data(
        self,
        symbol: str,
        interval: Interval,
        start_date: datetime,
        end_date: datetime,
        exchange: Exchange = Exchange.NSE,
    ) -> HistoricalData:
        if self.raise_error is not None:
            raise self.raise_error
        if self.next_history is not None:
            return self.next_history
        return HistoricalData(symbol=symbol, exchange=exchange, interval=interval)

    async def search_symbol(
        self, query: str, exchange: Exchange | None = None
    ) -> SymbolSearchResponse:
        return SymbolSearchResponse(query=query)

    async def get_market_depth(
        self, symbol: str, exchange: Exchange = Exchange.NSE
    ) -> MarketDepth:
        return MarketDepth(
            symbol=symbol,
            exchange=exchange,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        )


class FakeClock:
    """A manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def provider() -> FakeProvider:
    """Return a fresh fake provider."""
    return FakeProvider()


@pytest.fixture
def clock() -> FakeClock:
    """Return a controllable clock."""
    return FakeClock()


@pytest.fixture
def service(provider: FakeProvider, clock: FakeClock) -> MarketDataService:
    """Return a service wired to the fake provider and clock."""
    return MarketDataService(provider, market_status_ttl_seconds=30.0, clock=clock)


# -- Quotes ----------------------------------------------------------------


async def test_get_quote_normalizes_symbol(
    service: MarketDataService, provider: FakeProvider
) -> None:
    """Symbols are trimmed and upper-cased before hitting the provider."""
    provider.next_quote = _quote("RELIANCE")
    quote = await service.get_quote("  reliance  ")
    assert quote.symbol == "RELIANCE"
    assert provider.quote_calls == 1


async def test_get_quote_blank_symbol_raises(service: MarketDataService) -> None:
    """A blank symbol is rejected before any provider call."""
    with pytest.raises(InvalidSymbolError):
        await service.get_quote("   ")


async def test_get_quote_validates_symbol_mismatch(
    service: MarketDataService, provider: FakeProvider
) -> None:
    """A mismatched symbol in the response is a provider error."""
    provider.next_quote = _quote("TCS")
    with pytest.raises(ProviderError):
        await service.get_quote("RELIANCE")


async def test_get_quotes_dedupes_and_orders(
    service: MarketDataService, provider: FakeProvider
) -> None:
    """Duplicates/blanks are removed and results follow request order."""
    provider.next_quotes = [_quote("TCS"), _quote("RELIANCE")]
    quotes = await service.get_quotes(["reliance", "TCS", "reliance", "  "])
    assert [q.symbol for q in quotes] == ["RELIANCE", "TCS"]


async def test_get_quotes_missing_symbol_raises(
    service: MarketDataService, provider: FakeProvider
) -> None:
    """A requested symbol missing from the response is an error."""
    provider.next_quotes = [_quote("RELIANCE")]
    with pytest.raises(ProviderError):
        await service.get_quotes(["RELIANCE", "TCS"])


async def test_get_quotes_empty_raises(service: MarketDataService) -> None:
    """An all-blank symbol list is rejected."""
    with pytest.raises(InvalidSymbolError):
        await service.get_quotes(["", "  "])


# -- Historical ------------------------------------------------------------


async def test_get_historical_validates_interval(
    service: MarketDataService, provider: FakeProvider
) -> None:
    """A mismatched interval in the response is an error."""
    provider.next_history = HistoricalData(
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        interval=Interval.ONE_MINUTE,
        candles=(
            Candle(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                open=Decimal("1"),
                high=Decimal("2"),
                low=Decimal("1"),
                close=Decimal("2"),
                volume=10,
            ),
        ),
    )
    with pytest.raises(ProviderError):
        await service.get_historical_data(
            "RELIANCE",
            Interval.ONE_DAY,
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 2, 1, tzinfo=UTC),
        )


async def test_get_historical_rejects_inverted_range(
    service: MarketDataService,
) -> None:
    """end_date must be strictly after start_date."""
    with pytest.raises(ValueError):
        await service.get_historical_data(
            "RELIANCE",
            Interval.ONE_DAY,
            datetime(2024, 2, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
        )


# -- Market status caching -------------------------------------------------


async def test_market_status_is_cached_within_ttl(
    service: MarketDataService, provider: FakeProvider
) -> None:
    """The provider is hit once while the cache is fresh."""
    await service.get_market_status()
    await service.get_market_status()
    assert provider.status_calls == 1


async def test_market_status_refreshes_after_ttl(
    service: MarketDataService, provider: FakeProvider, clock: FakeClock
) -> None:
    """The cache expires after the TTL and the provider is hit again."""
    await service.get_market_status()
    clock.now += 31.0
    await service.get_market_status()
    assert provider.status_calls == 2


async def test_market_status_cached_per_exchange(
    service: MarketDataService, provider: FakeProvider
) -> None:
    """Different exchanges are cached independently."""
    await service.get_market_status(Exchange.NSE)
    await service.get_market_status(Exchange.BSE)
    assert provider.status_calls == 2


# -- Market open -----------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [(MarketStatus.OPEN, True), (MarketStatus.CLOSED, False)],
)
async def test_is_market_open(
    service: MarketDataService,
    provider: FakeProvider,
    status: MarketStatus,
    expected: bool,
) -> None:
    """is_market_open reflects the reported status."""
    provider.market_status = status
    assert await service.is_market_open() is expected


async def test_is_market_open_uses_status_cache(
    service: MarketDataService, provider: FakeProvider
) -> None:
    """is_market_open shares the market-status cache."""
    await service.get_market_status()
    await service.is_market_open()
    assert provider.status_calls == 1


# -- Error handling --------------------------------------------------------


async def test_provider_error_propagates(
    service: MarketDataService, provider: FakeProvider
) -> None:
    """Provider errors are propagated to the caller."""
    provider.raise_error = NetworkError("boom")
    with pytest.raises(NetworkError):
        await service.get_quote("RELIANCE")


# -- Dependency injection --------------------------------------------------


def test_service_depends_only_on_interface(provider: FakeProvider) -> None:
    """The service accepts any MarketDataProvider implementation."""
    service = MarketDataService(provider)
    assert isinstance(service, MarketDataService)


def test_di_returns_singleton() -> None:
    """The DI accessor returns a cached singleton service."""
    from app.services.dependencies import (
        _service_singleton,
        get_market_data_service,
    )

    _service_singleton.cache_clear()
    try:
        assert get_market_data_service() is get_market_data_service()
    finally:
        _service_singleton.cache_clear()
