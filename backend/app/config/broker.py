"""Broker / market-data provider credentials.

Only credential configuration lives here — no API clients or market logic.
Groww is the initial broker integration for the Indian market.
"""

from __future__ import annotations

from pydantic import Field

from app.config.base import BaseConfig, build_config


class GrowwSettings(BaseConfig):
    """Groww API credentials (``GROWW_*`` environment variables)."""

    model_config = build_config(env_prefix="GROWW_")

    api_key: str = Field(default="", description="Groww API key.")
    api_secret: str | None = Field(
        default=None,
        description="Groww API secret. Optional until secret-based auth is used.",
    )

    # Endpoint configuration. Kept here (not in code) so no URL is hardcoded in
    # the provider and every deployment can point at its own gateway/mock.
    base_url: str = Field(
        default="https://api.groww.in",
        description="Groww API base URL.",
    )
    api_version: str = Field(default="v1", description="Groww API version segment.")
    timeout_seconds: float = Field(
        default=10.0, gt=0, description="Per-request timeout in seconds."
    )

    @property
    def is_configured(self) -> bool:
        """Return ``True`` when an API key has been provided."""
        return bool(self.api_key)
