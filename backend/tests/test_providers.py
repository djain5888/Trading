"""Unit tests for the market-data provider abstraction and Groww skeleton."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from app.config.broker import GrowwSettings
from app.market import (
    Candle,
    Exchange,
    HistoricalDataRequest,
    Interval,
    MarketStatus,
    Quote,
)
from app.providers import (
    AuthenticationError,
    InvalidSymbolError,
    MarketDataProvider,
    NetworkError,
    ProviderError,
    RateLimitError,
    RequestSpec,
    build_market_data_provider,
    get_market_data_provider,
)
from app.providers.base import ResponseParser
from app.providers.groww import (
    GrowwAuthenticator,
    GrowwProvider,
    GrowwRequestBuilder,
    GrowwResponseParser,
)
from app.providers.http import HttpMethod
from pydantic import ValidationError


@pytest.fixture
def settings() -> GrowwSettings:
    """Return Groww settings with a placeholder API key configured."""
    return GrowwSettings(api_key="test-key")


@pytest.fixture
def provider(settings: GrowwSettings) -> GrowwProvider:
    """Return a Groww provider built from the test settings."""
    return GrowwProvider(settings=settings)


# ---------------------------------------------------------------------------
# Interface & structure
# ---------------------------------------------------------------------------


def test_provider_interface_is_abstract() -> None:
    """The abstract interface cannot be instantiated directly."""
    with pytest.raises(TypeError):
        MarketDataProvider()  # type: ignore[abstract]


def test_groww_provider_implements_interface(provider: GrowwProvider) -> None:
    """GrowwProvider is a MarketDataProvider."""
    assert isinstance(provider, MarketDataProvider)


def test_groww_parser_satisfies_response_parser() -> None:
    """The Groww parser structurally satisfies the ResponseParser protocol."""
    assert isinstance(GrowwResponseParser(), ResponseParser)


# ---------------------------------------------------------------------------
# Endpoints raise NotImplementedError until API integration
# ---------------------------------------------------------------------------


async def test_get_market_status_not_implemented(provider: GrowwProvider) -> None:
    """Market status is not implemented yet."""
    with pytest.raises(NotImplementedError):
        await provider.get_market_status()


async def test_get_quote_not_implemented(provider: GrowwProvider) -> None:
    """Quote retrieval is not implemented yet."""
    with pytest.raises(NotImplementedError):
        await provider.get_quote("RELIANCE")


async def test_get_quotes_not_implemented(provider: GrowwProvider) -> None:
    """Batch quote retrieval is not implemented yet."""
    with pytest.raises(NotImplementedError):
        await provider.get_quotes(["RELIANCE", "TCS"])


async def test_get_historical_not_implemented(provider: GrowwProvider) -> None:
    """Historical data retrieval is not implemented yet."""
    with pytest.raises(NotImplementedError):
        await provider.get_historical_data(
            "RELIANCE",
            Interval.ONE_DAY,
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 2, 1, tzinfo=UTC),
        )


async def test_search_symbol_not_implemented(provider: GrowwProvider) -> None:
    """Symbol search is not implemented yet."""
    with pytest.raises(NotImplementedError):
        await provider.search_symbol("RELI")


async def test_get_market_depth_not_implemented(provider: GrowwProvider) -> None:
    """Market depth retrieval is not implemented yet."""
    with pytest.raises(NotImplementedError):
        await provider.get_market_depth("RELIANCE")


async def test_is_market_open_delegates_to_status(provider: GrowwProvider) -> None:
    """is_market_open derives from get_market_status and thus is not ready."""
    with pytest.raises(NotImplementedError):
        await provider.is_market_open()


# ---------------------------------------------------------------------------
# Authentication / session management
# ---------------------------------------------------------------------------


async def test_authenticate_requires_credentials() -> None:
    """Authentication fails fast when the API key is missing."""
    authenticator = GrowwAuthenticator(GrowwSettings(api_key=""))
    with pytest.raises(AuthenticationError):
        await authenticator.authenticate()


async def test_authenticate_with_credentials_not_implemented(
    settings: GrowwSettings,
) -> None:
    """With credentials present, the live token exchange is not implemented."""
    authenticator = GrowwAuthenticator(settings)
    with pytest.raises(NotImplementedError):
        await authenticator.authenticate()


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------


def test_request_builder_builds_quote_spec(settings: GrowwSettings) -> None:
    """The quote request spec carries method, versioned path and params."""
    builder = GrowwRequestBuilder(settings)
    from app.market.models import QuoteRequest

    spec = builder.build_quote(QuoteRequest(symbol="RELIANCE", exchange=Exchange.NSE))

    assert spec.method is HttpMethod.GET
    assert spec.path == "/v1/quote"
    assert spec.params == {"symbol": "RELIANCE", "exchange": "NSE"}
    assert spec.headers["Authorization"] == "Bearer test-key"


def test_request_spec_full_url_uses_base_url(settings: GrowwSettings) -> None:
    """The absolute URL is composed from the configured base URL."""
    spec = RequestSpec(method=HttpMethod.GET, path="/v1/quote")
    assert spec.full_url("https://gw.example/") == "https://gw.example/v1/quote"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [AuthenticationError, RateLimitError, InvalidSymbolError, NetworkError],
)
def test_exceptions_derive_from_provider_error(error: type[ProviderError]) -> None:
    """Every provider exception is a ProviderError."""
    assert issubclass(error, ProviderError)


def test_rate_limit_error_carries_retry_after() -> None:
    """RateLimitError preserves the retry-after hint."""
    err = RateLimitError("slow down", retry_after_seconds=1.5)
    assert err.retry_after_seconds == 1.5


def test_invalid_symbol_error_carries_symbol() -> None:
    """InvalidSymbolError preserves the offending symbol."""
    err = InvalidSymbolError("unknown", symbol="XYZ")
    assert err.symbol == "XYZ"


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


def test_quote_model_is_frozen() -> None:
    """Quote value objects are immutable."""
    quote = Quote(
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        last_price=Decimal("2500.50"),
        timestamp=datetime.now(UTC),
    )
    with pytest.raises(ValidationError):
        quote.last_price = Decimal("1")  # type: ignore[misc]


def test_historical_request_rejects_inverted_range() -> None:
    """A historical request with end <= start is rejected."""
    with pytest.raises(ValidationError):
        HistoricalDataRequest(
            symbol="RELIANCE",
            interval=Interval.ONE_DAY,
            start=datetime(2024, 2, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, tzinfo=UTC),
        )


def test_candle_rejects_negative_volume() -> None:
    """A candle cannot have negative volume."""
    with pytest.raises(ValidationError):
        Candle(
            timestamp=datetime.now(UTC),
            open=Decimal("1"),
            high=Decimal("2"),
            low=Decimal("1"),
            close=Decimal("2"),
            volume=-5,
        )


def test_enums_have_expected_values() -> None:
    """The market enums expose the expected string values."""
    assert Exchange.NSE.value == "NSE"
    assert Interval.ONE_DAY.value == "1d"
    assert Interval.ONE_MINUTE.is_intraday
    assert not Interval.ONE_DAY.is_intraday
    assert MarketStatus.OPEN.value == "open"


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


def test_build_market_data_provider_returns_interface() -> None:
    """The factory returns something typed as the interface."""
    assert isinstance(build_market_data_provider(), MarketDataProvider)


def test_get_market_data_provider_is_singleton() -> None:
    """The DI accessor returns a cached singleton."""
    from app.providers.dependencies import _provider_singleton

    _provider_singleton.cache_clear()
    try:
        assert get_market_data_provider() is get_market_data_provider()
    finally:
        _provider_singleton.cache_clear()
