"""Groww authentication and session management.

This module models the *structure* of Groww authentication — credentials, a
session with an access token and expiry, and the authenticator that manages
them — without performing any live token exchange. The network round-trip is
deliberately left as ``NotImplementedError`` until API integration.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.config.broker import GrowwSettings
from app.providers.exceptions import AuthenticationError


class GrowwSession(BaseModel):
    """A Groww access session.

    Holds the bearer token and its validity window so the provider can decide
    when to (re)authenticate.
    """

    model_config = ConfigDict(frozen=True)

    access_token: str = Field(description="Bearer access token.")
    issued_at: datetime = Field(description="Token issue time (UTC).")
    expires_at: datetime = Field(description="Token expiry time (UTC).")

    def is_valid(self, *, now: datetime | None = None) -> bool:
        """Return whether the session is currently usable.

        Args:
            now: Reference time; defaults to the current UTC time.

        Returns:
            ``True`` if the token is present and unexpired.
        """
        reference = now or datetime.now(UTC)
        return bool(self.access_token) and reference < self.expires_at


class GrowwAuthenticator:
    """Manages the Groww session lifecycle.

    The authenticator validates that credentials are present and caches the
    active session. Establishing a session performs a network call, which is
    not yet implemented.
    """

    def __init__(self, settings: GrowwSettings) -> None:
        """Initialise the authenticator.

        Args:
            settings: Groww credentials and endpoint configuration.
        """
        self._settings = settings
        self._session: GrowwSession | None = None

    @property
    def session(self) -> GrowwSession | None:
        """Return the cached session, if any."""
        return self._session

    def _require_credentials(self) -> None:
        """Ensure credentials are configured.

        Raises:
            AuthenticationError: If the API key is missing.
        """
        if not self._settings.api_key:
            raise AuthenticationError("Groww API key is not configured.")

    async def authenticate(self) -> GrowwSession:
        """Establish and cache a new Groww session.

        Validates credentials, then exchanges them for an access token.

        Raises:
            AuthenticationError: If credentials are not configured.
            NotImplementedError: The live token exchange is not yet integrated.
        """
        self._require_credentials()
        raise NotImplementedError("Groww authentication requires live API integration.")

    async def ensure_session(self) -> GrowwSession:
        """Return a valid session, authenticating if necessary.

        Returns:
            A valid :class:`GrowwSession`.

        Raises:
            AuthenticationError: If credentials are not configured.
            NotImplementedError: The live token exchange is not yet integrated.
        """
        if self._session is not None and self._session.is_valid():
            return self._session
        self._session = await self.authenticate()
        return self._session
