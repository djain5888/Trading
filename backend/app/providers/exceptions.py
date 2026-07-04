"""Provider exception hierarchy.

All provider failures derive from :class:`ProviderError` so callers can catch
the whole family with a single ``except`` while still discriminating on
specific subclasses when they need to.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for all market-data provider errors."""

    def __init__(self, message: str, *, details: str | None = None) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description of the failure.
            details: Optional additional context (e.g. an upstream payload).
        """
        super().__init__(message)
        self.message = message
        self.details = details


class AuthenticationError(ProviderError):
    """Raised when authentication or session establishment fails."""


class RateLimitError(ProviderError):
    """Raised when the provider reports that a rate limit was exceeded."""

    def __init__(
        self,
        message: str,
        *,
        details: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description of the failure.
            details: Optional additional context.
            retry_after_seconds: Suggested wait before retrying, if known.
        """
        super().__init__(message, details=details)
        self.retry_after_seconds = retry_after_seconds


class MarketClosedError(ProviderError):
    """Raised when an operation requires an open market that is closed."""


class InvalidSymbolError(ProviderError):
    """Raised when a requested symbol is unknown to the provider."""

    def __init__(
        self, message: str, *, symbol: str | None = None, details: str | None = None
    ) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description of the failure.
            symbol: The offending symbol, if applicable.
            details: Optional additional context.
        """
        super().__init__(message, details=details)
        self.symbol = symbol


class NetworkError(ProviderError):
    """Raised when the provider cannot be reached due to a transport error."""
