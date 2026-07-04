"""Unit tests for the configuration system."""

from __future__ import annotations

import pytest
from app.config import (
    Environment,
    GrowwSettings,
    PostgresSettings,
    RedisSettings,
    Settings,
    get_settings,
)
from pydantic import ValidationError


def test_defaults_use_development_environment() -> None:
    """A settings object with no env overrides defaults to development."""
    settings = Settings()

    assert settings.environment is Environment.DEVELOPMENT
    assert settings.is_development
    assert not settings.is_production
    assert settings.app_name == "Titan"


def test_nested_groups_are_populated() -> None:
    """Each infrastructure group is instantiated with sane defaults."""
    settings = Settings()

    assert isinstance(settings.postgres, PostgresSettings)
    assert isinstance(settings.redis, RedisSettings)
    assert settings.postgres.port == 5432
    assert settings.redis.port == 6379
    assert settings.duckdb.path.endswith(".duckdb")


def test_env_vars_are_read_with_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefixed environment variables populate the nested groups."""
    monkeypatch.setenv("APP_NAME", "Titan-Test")
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("REDIS_HOST", "cache.internal")
    monkeypatch.setenv("GROWW_API_KEY", "abc123")

    settings = Settings()

    assert settings.app_name == "Titan-Test"
    assert settings.postgres.host == "db.internal"
    assert settings.postgres.port == 6543
    assert settings.redis.host == "cache.internal"
    assert settings.groww.api_key == "abc123"
    assert settings.groww.is_configured


def test_postgres_dsn_is_built_from_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The PostgreSQL DSN is assembled from the individual components."""
    monkeypatch.setenv("POSTGRES_USER", "quant")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_DB", "titan_prod")

    dsn = PostgresSettings().dsn

    assert dsn == "postgresql+asyncpg://quant:secret@localhost:5432/titan_prod"


def test_invalid_port_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """An out-of-range port raises a validation error at construction time."""
    monkeypatch.setenv("POSTGRES_PORT", "70000")

    with pytest.raises(ValidationError):
        PostgresSettings()


def test_invalid_environment_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown environment value is rejected."""
    monkeypatch.setenv("ENVIRONMENT", "staging")

    with pytest.raises(ValidationError):
        Settings()


def test_production_requires_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production without required secrets fails fast with a clear message."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("GROWW_API_KEY", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    message = str(exc_info.value)
    assert "POSTGRES_PASSWORD" in message
    assert "GROWW_API_KEY" in message


def test_production_succeeds_with_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production validates once the required secrets are present."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("POSTGRES_PASSWORD", "super-secret")
    monkeypatch.setenv("GROWW_API_KEY", "live-key")

    settings = Settings()

    assert settings.is_production
    assert settings.postgres.password == "super-secret"


def test_get_settings_is_cached() -> None:
    """The accessor returns a cached singleton instance."""
    get_settings.cache_clear()
    try:
        first = get_settings()
        second = get_settings()
        assert first is second
    finally:
        get_settings.cache_clear()


def test_environment_enum_compares_to_string() -> None:
    """The environment enum is a string subclass for transparent comparison."""
    assert Environment.PRODUCTION.value == "production"
    assert str(Environment.PRODUCTION) == "production"
    assert GrowwSettings().is_configured is False
