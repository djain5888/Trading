"""Aggregate application configuration.

The top-level :class:`Settings` object composes the per-domain settings groups
(database, cache, broker) into a single strongly typed, immutable, cached
object. Configuration is read from the environment and an optional ``.env``
file, validated on construction, and exposed through the cached
:func:`get_settings` accessor so no global mutable state is required.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from app.config.base import build_config
from app.config.broker import GrowwSettings
from app.config.database import DuckDBSettings, PostgresSettings, RedisSettings
from app.config.environment import Environment

#: Backwards-compatible alias. Prefer :class:`app.config.environment.Environment`.
__all__ = ["Environment", "Settings", "get_settings"]

#: Selectable market-data provider backends.
MarketDataBackend = Literal["groww", "groww_sdk", "fake"]


class Settings(BaseSettings):
    """Root application settings.

    Reads unprefixed top-level variables (``APP_NAME``, ``ENVIRONMENT`` ...)
    and delegates each infrastructure domain to a nested settings group that
    owns its own environment-variable prefix.
    """

    model_config = build_config()

    # --- Application metadata ---
    app_name: str = Field(default="Titan", description="Human-readable app name.")
    app_version: str = Field(default="0.1.0", description="Semantic version string.")
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment the app is running in.",
    )
    debug: bool = Field(default=False, description="Enable verbose debug behaviour.")

    # --- Observability ---
    log_level: str = Field(default="INFO", description="Root logging level.")

    # --- HTTP server ---
    host: str = Field(default="0.0.0.0", description="Bind host for the API server.")
    port: int = Field(default=8000, ge=1, le=65535, description="API server port.")
    api_prefix: str = Field(default="/api/v1", description="Base path for the API.")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Origins allowed to make cross-origin requests.",
    )

    # --- Providers ---
    market_data_provider: MarketDataBackend = Field(
        default="groww_sdk",
        description=(
            "Market-data backend: 'groww_sdk' (growwapi SDK, default), 'groww' "
            "(real httpx) or 'fake' (offline skeleton)."
        ),
    )
    watchlist_file: str | None = Field(
        default=None,
        description="Optional JSON file overriding the default watchlist/sectors.",
    )

    # --- Infrastructure groups ---
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    duckdb: DuckDBSettings = Field(default_factory=DuckDBSettings)
    groww: GrowwSettings = Field(default_factory=GrowwSettings)

    @property
    def is_production(self) -> bool:
        """Return ``True`` when running in the production environment."""
        return self.environment is Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Return ``True`` when running in the development environment."""
        return self.environment is Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        """Return ``True`` when running in the testing environment."""
        return self.environment is Environment.TESTING

    @model_validator(mode="after")
    def _validate_production_requirements(self) -> Settings:
        """Fail fast when production is missing required secrets.

        Development and testing run with safe local defaults, but production
        must supply real credentials. Missing values raise a clear error at
        construction time rather than surfacing as an obscure runtime failure.

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If a required production value is unset.
        """
        if self.environment is not Environment.PRODUCTION:
            return self

        missing: list[str] = []
        if not self.postgres.password:
            missing.append("POSTGRES_PASSWORD")
        if not self.groww.api_key:
            missing.append("GROWW_API_KEY")

        if missing:
            raise ValueError(
                "Missing required production configuration: "
                + ", ".join(missing)
                + ". Set these environment variables before starting Titan in "
                "production."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, validated application settings singleton.

    The :func:`functools.lru_cache` decorator guarantees the environment is
    parsed and validated exactly once per process while keeping the object
    free of global mutable state. Tests may reset it with
    ``get_settings.cache_clear()``.

    Returns:
        The validated :class:`Settings` instance.

    Raises:
        pydantic.ValidationError: If configuration fails validation.
    """
    return Settings()
