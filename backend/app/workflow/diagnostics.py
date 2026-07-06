"""Health and version diagnostics.

These support the ``titan health`` and ``titan version`` commands. They only
inspect configuration and lightweight liveness; they never mutate state.
"""

from __future__ import annotations

import importlib
import platform
import subprocess
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from app.config.settings import Settings, get_settings
from app.market.calendar.dependencies import get_market_calendar_service
from app.market.enums import Exchange

#: Runtime dependencies verified by the health check.
_REQUIRED_PACKAGES = ("fastapi", "pydantic", "duckdb", "httpx", "typer")


@dataclass(frozen=True)
class HealthCheck:
    """The result of a single health check."""

    name: str
    healthy: bool
    critical: bool
    detail: str


class VersionInfo(BaseModel):
    """Version and environment information."""

    model_config = ConfigDict(frozen=True)

    app_version: str = Field(description="Application version.")
    git_commit: str = Field(description="Current git commit (or 'unknown').")
    python_version: str = Field(description="Running Python version.")
    environment: str = Field(description="Active deployment environment.")


def version_info(settings: Settings | None = None) -> VersionInfo:
    """Return application version and environment details."""
    resolved = settings or get_settings()
    return VersionInfo(
        app_version=resolved.app_version,
        git_commit=_git_commit(),
        python_version=platform.python_version(),
        environment=resolved.environment,
    )


def run_health_checks(settings: Settings | None = None) -> list[HealthCheck]:
    """Run all health checks and return their results."""
    resolved = settings or get_settings()
    return [
        _check_configuration(resolved),
        _check_database(),
        _check_calendar(),
        _check_provider(resolved),
        _check_dependencies(),
    ]


def is_healthy(checks: list[HealthCheck]) -> bool:
    """Return whether every critical check passed."""
    return all(check.healthy for check in checks if check.critical)


# -- Individual checks -----------------------------------------------------


def _check_configuration(settings: Settings) -> HealthCheck:
    """Verify configuration loaded."""
    return HealthCheck(
        name="Configuration",
        healthy=True,
        critical=True,
        detail=f"{settings.app_name} v{settings.app_version} ({settings.environment})",
    )


def _check_database() -> HealthCheck:
    """Verify the DuckDB store can be opened."""
    try:
        from app.market.historical.storage.duckdb_repository import (
            DuckDBCandleRepository,
        )

        repository = DuckDBCandleRepository(":memory:")
        repository.close()
    except Exception as exc:  # noqa: BLE001 - report as unhealthy
        return HealthCheck("Database", False, True, f"DuckDB error: {exc}")
    return HealthCheck("Database", True, True, "DuckDB reachable (in-memory)")


def _check_calendar() -> HealthCheck:
    """Verify the market calendar responds."""
    try:
        calendar = get_market_calendar_service()
        state = calendar.get_market_state(Exchange.NSE)
    except Exception as exc:  # noqa: BLE001 - report as unhealthy
        return HealthCheck("Market Calendar", False, True, f"Error: {exc}")
    return HealthCheck("Market Calendar", True, True, f"NSE state: {state.value}")


def _check_provider(settings: Settings) -> HealthCheck:
    """Verify the market-data provider is configured (non-critical)."""
    configured = settings.groww.is_configured
    detail = "Groww API key configured" if configured else "Groww API key missing"
    return HealthCheck("Provider", configured, False, detail)


def _check_dependencies() -> HealthCheck:
    """Verify required runtime packages import."""
    missing: list[str] = []
    for package in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    if missing:
        return HealthCheck(
            "Dependencies", False, True, f"Missing: {', '.join(missing)}"
        )
    return HealthCheck("Dependencies", True, True, "All packages importable")


def _git_commit() -> str:
    """Return the short git commit, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"
