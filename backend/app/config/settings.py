"""Application configuration.

Settings are loaded from environment variables (and an optional ``.env``
file) using ``pydantic-settings``. A cached accessor is exposed via
:func:`get_settings` so the configuration is parsed only once per process.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production", "test"]


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Values are read from environment variables. Names are case-insensitive
    and may optionally be prefixed with ``TITAN_``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TITAN_",
        case_sensitive=False,
        extra="ignore",
    )

    # Application metadata.
    app_name: str = Field(default="Titan", description="Human-readable app name.")
    app_version: str = Field(default="0.1.0", description="Semantic version string.")
    environment: Environment = Field(
        default="development",
        description="Deployment environment the app is running in.",
    )
    debug: bool = Field(default=False, description="Enable verbose debug behaviour.")

    # HTTP server.
    host: str = Field(default="0.0.0.0", description="Bind host for the API server.")
    port: int = Field(default=8000, description="Bind port for the API server.")
    api_prefix: str = Field(default="/api/v1", description="Base path for the API.")

    # CORS.
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Origins allowed to make cross-origin requests.",
    )

    # Persistence.
    database_url: str = Field(
        default="sqlite+aiosqlite:///./titan.db",
        description="SQLAlchemy async database URL.",
    )

    # Observability.
    log_level: str = Field(default="INFO", description="Root logging level.")

    @property
    def is_production(self) -> bool:
        """Return ``True`` when running in the production environment."""
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance.

    Using an LRU cache guarantees a single parse of the environment per
    process while remaining trivially overridable in tests via
    ``get_settings.cache_clear()``.
    """
    return Settings()
