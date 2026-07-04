"""Data-store connection settings.

Covers the three persistence backends Titan relies on: PostgreSQL (relational
state), Redis (cache / messaging) and DuckDB (analytical storage). These
classes describe *connection configuration only* — no engines, sessions or
clients are created here.
"""

from __future__ import annotations

from pydantic import Field

from app.config.base import BaseConfig, build_config


class PostgresSettings(BaseConfig):
    """PostgreSQL connection settings (``POSTGRES_*`` environment variables)."""

    model_config = build_config(env_prefix="POSTGRES_")

    host: str = Field(default="localhost", description="Database server host.")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port.")
    db: str = Field(default="titan", description="Database name.")
    user: str = Field(default="titan", description="Database user.")
    password: str = Field(default="", description="Database password.")

    @property
    def dsn(self) -> str:
        """Return an async SQLAlchemy DSN for this PostgreSQL instance."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )


class RedisSettings(BaseConfig):
    """Redis connection settings (``REDIS_*`` environment variables)."""

    model_config = build_config(env_prefix="REDIS_")

    host: str = Field(default="localhost", description="Redis server host.")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port.")

    @property
    def url(self) -> str:
        """Return a ``redis://`` connection URL for this instance."""
        return f"redis://{self.host}:{self.port}"


class DuckDBSettings(BaseConfig):
    """DuckDB storage settings (``DUCKDB_*`` environment variables)."""

    model_config = build_config(env_prefix="DUCKDB_")

    path: str = Field(
        default="./titan.duckdb",
        description="Filesystem path to the DuckDB database file.",
    )
