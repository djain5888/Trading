"""Async HTTP client for Groww.

Wraps a single reused :class:`httpx.AsyncClient` with connection pooling,
keep-alive and timeouts, and layers a retry policy that honours ``Retry-After``
and uses exponential backoff. Every HTTP failure is mapped to a domain
:class:`ProviderError`; no ``httpx`` exception or raw status ever escapes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.config.broker import GrowwSettings
from app.core.logging import get_logger
from app.providers.exceptions import (
    AuthenticationError,
    InvalidSymbolError,
    NetworkError,
    ProviderError,
    RateLimitError,
)
from app.providers.http import RequestSpec

logger = get_logger(__name__)

#: Async sleep signature, injectable so backoff is instant in tests.
Sleeper = Callable[[float], Awaitable[None]]

_UNAUTHORIZED = 401
_FORBIDDEN = 403
_NOT_FOUND = 404
_TOO_MANY_REQUESTS = 429
_SERVER_ERROR_FLOOR = 500


class GrowwHTTPClient:
    """A retrying, error-mapping async HTTP client for the Groww API."""

    def __init__(
        self,
        settings: GrowwSettings,
        *,
        client: httpx.AsyncClient | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        """Initialise the client.

        Args:
            settings: Endpoint, timeout, pool and retry configuration.
            client: Pre-built client (used in tests); a pooled client is
                created from settings when omitted.
            sleeper: Async sleep used for backoff (injectable for tests).
        """
        self._settings = settings
        self._sleeper = sleeper or asyncio.sleep
        self._owns_client = client is None
        self._client = client or self._build_client(settings)

    @staticmethod
    def _build_client(settings: GrowwSettings) -> httpx.AsyncClient:
        """Build the pooled, keep-alive async client from settings."""
        limits = httpx.Limits(
            max_connections=settings.pool_max_connections,
            max_keepalive_connections=settings.pool_max_keepalive_connections,
            keepalive_expiry=settings.keepalive_expiry_seconds,
        )
        timeout = httpx.Timeout(
            settings.timeout_seconds, connect=settings.connect_timeout_seconds
        )
        return httpx.AsyncClient(
            base_url=settings.base_url, limits=limits, timeout=timeout
        )

    async def request(
        self,
        spec: RequestSpec,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute a request with retries and return the decoded JSON body.

        Args:
            spec: The request to execute.
            extra_headers: Headers merged over the spec's headers (e.g. auth).

        Returns:
            The parsed JSON object.

        Raises:
            ProviderError: Or a subclass, for any failure.
        """
        headers = {**spec.headers, **(extra_headers or {})}
        attempt = 0
        while True:
            response, error, retryable, retry_after = await self._attempt(spec, headers)
            if response is not None:
                return self._decode(response)
            assert error is not None  # noqa: S101 - invariant: no response => error
            if not retryable or attempt >= self._settings.max_retries:
                raise error
            await self._sleeper(self._backoff(attempt, retry_after))
            attempt += 1

    async def _attempt(
        self, spec: RequestSpec, headers: dict[str, str]
    ) -> tuple[httpx.Response | None, ProviderError | None, bool, float | None]:
        """Perform one HTTP attempt.

        Returns:
            ``(response, error, retryable, retry_after)``. On success the
            response is set and error is ``None``; otherwise error is set.
        """
        try:
            response = await self._client.request(
                spec.method.value,
                spec.path,
                params=spec.params,
                headers=headers,
                json=spec.json_body,
            )
        except httpx.TimeoutException as exc:
            return (
                None,
                NetworkError("Groww request timed out.", details=str(exc)),
                True,
                None,
            )
        except httpx.TransportError as exc:
            return (
                None,
                NetworkError("Groww transport error.", details=str(exc)),
                True,
                None,
            )

        if response.is_success:
            return response, None, False, None

        error, retryable, retry_after = self._map_status(response)
        return None, error, retryable, retry_after

    @staticmethod
    def _map_status(
        response: httpx.Response,
    ) -> tuple[ProviderError, bool, float | None]:
        """Map a non-2xx response to a domain error and retry decision."""
        status = response.status_code
        detail = response.text[:512]
        if status in (_UNAUTHORIZED, _FORBIDDEN):
            return (
                AuthenticationError("Groww rejected credentials.", details=detail),
                False,
                None,
            )
        if status == _NOT_FOUND:
            return (
                InvalidSymbolError("Groww resource not found.", details=detail),
                False,
                None,
            )
        if status == _TOO_MANY_REQUESTS:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            return (
                RateLimitError(
                    "Groww rate limit exceeded.",
                    details=detail,
                    retry_after_seconds=retry_after,
                ),
                True,
                retry_after,
            )
        if status >= _SERVER_ERROR_FLOOR:
            return (
                NetworkError(f"Groww server error {status}.", details=detail),
                True,
                None,
            )
        return (
            ProviderError(f"Groww request failed ({status}).", details=detail),
            False,
            None,
        )

    def _decode(self, response: httpx.Response) -> dict[str, Any]:
        """Decode a JSON object body, mapping malformed payloads."""
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                "Groww returned a malformed JSON response.", details=str(exc)
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderError("Groww response was not a JSON object.")
        return payload

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        """Return the delay before the next attempt."""
        if retry_after is not None:
            return min(retry_after, self._settings.backoff_max_seconds)
        delay = self._settings.backoff_base_seconds * float(2**attempt)
        return min(delay, self._settings.backoff_max_seconds)

    async def aclose(self) -> None:
        """Close the client if it was created internally."""
        if self._owns_client:
            await self._client.aclose()


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header expressed in seconds."""
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None
