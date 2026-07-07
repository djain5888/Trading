"""Broker / market-data provider credentials.

Only credential configuration lives here — no API clients or market logic.
Groww is the initial broker integration for the Indian market.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from app.config.base import BaseConfig, build_config


class GrowwSettings(BaseConfig):
    """Groww API credentials (``GROWW_*`` environment variables)."""

    model_config = build_config(env_prefix="GROWW_")

    api_key: str = Field(default="", description="Groww API key.")
    api_secret: str | None = Field(
        default=None,
        description="Groww API secret. Optional until secret-based auth is used.",
    )
    totp_seed: str | None = Field(
        default=None,
        description="Base32 TOTP seed for API-key + TOTP auth. Never hardcode.",
    )

    @field_validator("api_secret", "totp_seed", mode="before")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        """Treat a blank optional secret as unset.

        Environment variables set to an empty or whitespace string (e.g. an
        unfilled ``GROWW_TOTP_SEED=`` line) would otherwise parse as ``""`` and
        wrongly enable that auth mode — forcing TOTP when only a secret is set.
        """
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    # Endpoint configuration. Kept here (not in code) so no URL is hardcoded in
    # the provider and every deployment can point at its own gateway/mock.
    base_url: str = Field(
        default="https://api.groww.in",
        description="Groww API base URL.",
    )
    api_version: str = Field(default="v1", description="Groww API version segment.")
    timeout_seconds: float = Field(
        default=10.0, gt=0, description="Per-request (read) timeout in seconds."
    )
    connect_timeout_seconds: float = Field(
        default=5.0, gt=0, description="Connection-establishment timeout in seconds."
    )

    # Retry / backoff policy.
    max_retries: int = Field(
        default=3, ge=0, description="Retries after the first attempt."
    )
    backoff_base_seconds: float = Field(
        default=0.5, gt=0, description="Initial exponential-backoff delay."
    )
    backoff_max_seconds: float = Field(
        default=30.0, gt=0, description="Maximum backoff delay."
    )

    # Connection pool / keep-alive.
    pool_max_connections: int = Field(
        default=20, ge=1, description="Maximum total pooled connections."
    )
    pool_max_keepalive_connections: int = Field(
        default=10, ge=1, description="Maximum idle keep-alive connections."
    )
    keepalive_expiry_seconds: float = Field(
        default=30.0, gt=0, description="Idle keep-alive expiry in seconds."
    )

    # Session / token management.
    default_token_ttl_seconds: float = Field(
        default=3600.0, gt=0, description="Fallback access-token lifetime."
    )
    token_refresh_skew_seconds: float = Field(
        default=60.0, ge=0, description="Refresh the token this early before expiry."
    )

    # Rate limiting. 0 disables throttling; otherwise cap outbound requests.
    throttle_rate_per_second: float = Field(
        default=0.0,
        ge=0,
        description="Max outbound requests per second (0 = unlimited).",
    )

    @property
    def is_configured(self) -> bool:
        """Return ``True`` when an API key has been provided."""
        return bool(self.api_key)

    @property
    def uses_token_exchange(self) -> bool:
        """Return whether auth requires a token exchange (secret or TOTP set)."""
        return self.api_secret is not None or self.totp_seed is not None
