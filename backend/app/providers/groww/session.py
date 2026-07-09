"""Groww authentication and session management.

Obtains an access token and refreshes it before expiry. The authentication mode
is explicit (``GROWW_AUTH_MODE``): ``token`` uses the API key directly as the
bearer token (no exchange); ``key_secret`` and ``totp`` exchange credentials for
an access token. Token exchange goes through the shared HTTP client, so it
inherits retries, timeouts and error mapping. Access is serialised with a lock
to avoid a refresh stampede. Time comes from an injected :class:`Clock`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from app.config.broker import GrowwSettings
from app.core.clock import Clock, SystemClock
from app.core.logging import get_logger
from app.providers.exceptions import AuthenticationError
from app.providers.groww import endpoints
from app.providers.groww.http_client import GrowwHTTPClient
from app.providers.groww.totp import generate_totp
from app.providers.http import HttpMethod, RequestSpec

logger = get_logger(__name__)

_TOKEN_KEYS = ("access_token", "token", "accessToken")
_EXPIRY_KEYS = ("expires_in", "expiresIn", "expiry_seconds")


class GrowwSessionManager:
    """Manages the Groww access token lifecycle."""

    def __init__(
        self,
        settings: GrowwSettings,
        http_client: GrowwHTTPClient,
        *,
        clock: Clock | None = None,
    ) -> None:
        """Initialise the session manager.

        Args:
            settings: Groww credentials and token configuration.
            http_client: Shared HTTP client for the token exchange.
            clock: Time source for expiry checks.
        """
        self._settings = settings
        self._http = http_client
        self._clock = clock or SystemClock()
        self._lock = asyncio.Lock()
        self._token: str | None = None
        self._expires_at: datetime | None = None

    async def get_access_token(self) -> str:
        """Return a valid access token, refreshing it if necessary.

        Raises:
            AuthenticationError: If credentials are missing or the exchange
                returns no usable token.
        """
        if not self._settings.api_key:
            raise AuthenticationError("Groww API key is not configured.")
        if not self._settings.uses_token_exchange:
            return self._settings.api_key

        async with self._lock:
            if self._is_valid():
                assert self._token is not None  # noqa: S101 - guaranteed by _is_valid
                return self._token
            return await self._refresh()

    def invalidate(self, stale_token: str | None = None) -> None:
        """Drop the cached token so the next call re-authenticates.

        Used to recover from a mid-run 401 by forcing a single re-auth. When
        ``stale_token`` is given, the cache is cleared only if it still holds
        that exact token: a concurrent caller may already have refreshed it, in
        which case dropping the fresh token would trigger a redundant refresh.
        This compare-and-swap keeps re-auth single-flight under concurrency.

        Args:
            stale_token: The token that just failed; when set, invalidation is
                a no-op unless it is still the cached token.
        """
        if stale_token is not None and self._token != stale_token:
            return
        self._token = None
        self._expires_at = None

    def _is_valid(self) -> bool:
        """Return whether the cached token is present and unexpired."""
        if self._token is None or self._expires_at is None:
            return False
        skew = timedelta(seconds=self._settings.token_refresh_skew_seconds)
        return self._clock.now() < self._expires_at - skew

    async def _refresh(self) -> str:
        """Exchange credentials for an access token, honouring the auth mode.

        The mode is honoured exactly: ``key_secret`` sends the API secret and
        ``totp`` sends a fresh TOTP — never both, and never a fallback.
        """
        body: dict[str, str] = {"key": self._settings.api_key}
        if self._settings.auth_mode == "key_secret":
            if self._settings.api_secret is None:
                raise AuthenticationError(
                    "GROWW_AUTH_MODE=key_secret requires an API secret."
                )
            body["secret"] = self._settings.api_secret
        elif self._settings.auth_mode == "totp":
            if self._settings.totp_seed is None:
                raise AuthenticationError("GROWW_AUTH_MODE=totp requires a TOTP seed.")
            body["totp"] = generate_totp(
                self._settings.totp_seed, now=self._clock.now()
            )
        spec = RequestSpec(
            method=HttpMethod.POST,
            path=endpoints.auth(self._settings.api_version),
            json_body=dict(body),
        )
        payload = await self._http.request(spec)
        token = _first_str(payload, _TOKEN_KEYS)
        if token is None:
            raise AuthenticationError("Groww auth response contained no token.")
        ttl = _first_float(payload, _EXPIRY_KEYS)
        if ttl is None:
            ttl = self._settings.default_token_ttl_seconds
        self._token = token
        self._expires_at = self._clock.now() + timedelta(seconds=ttl)
        logger.info("Refreshed Groww access token (ttl=%.0fs).", ttl)
        return token


def _first_str(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Return the first present string value among ``keys``."""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _first_float(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Return the first present numeric value among ``keys``."""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None
