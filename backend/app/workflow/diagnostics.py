"""Health and version diagnostics.

These support the ``titan health`` and ``titan version`` commands. They only
inspect configuration and lightweight liveness; they never mutate state.
"""

from __future__ import annotations

import asyncio
import importlib
import platform
import subprocess
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config.broker import GrowwSettings
from app.config.settings import Settings, get_settings
from app.market.calendar.dependencies import get_market_calendar_service
from app.market.enums import Exchange
from app.providers.exceptions import AuthenticationError, ProviderError

#: Async probe that authenticates against the broker, raising on failure.
AuthProbe = Callable[[GrowwSettings], Coroutine[Any, Any, None]]

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


def run_health_checks(
    settings: Settings | None = None, *, auth_probe: AuthProbe | None = None
) -> list[HealthCheck]:
    """Run all health checks and return their results.

    Args:
        settings: Application settings; defaults to the cached settings.
        auth_probe: Overrides the live broker auth probe (test injection).
    """
    resolved = settings or get_settings()
    return [
        _check_configuration(resolved),
        _check_database(),
        _check_calendar(),
        _check_provider(resolved, auth_probe),
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


def _check_provider(
    settings: Settings, auth_probe: AuthProbe | None = None
) -> HealthCheck:
    """Verify the market-data provider.

    A missing key is a non-critical, expected state (offline/CI). When token
    exchange is configured, a live auth probe runs: a hard authentication
    failure is surfaced as a critical, unhealthy check so ``titan health``
    exits non-zero and the operator sees the broker rejected the credentials.
    """
    groww = settings.groww
    if not groww.is_configured:
        return HealthCheck("Provider", False, False, "Groww API key missing")
    if not groww.uses_token_exchange:
        return HealthCheck("Provider", True, False, "Groww API key configured")
    probe = auth_probe or _live_auth_probe
    try:
        asyncio.run(probe(groww))
    except AuthenticationError as exc:
        return HealthCheck(
            "Provider", False, True, f"Groww authentication failed: {_describe(exc)}"
        )
    except ProviderError as exc:
        return HealthCheck(
            "Provider", False, False, f"Groww auth unavailable: {_describe(exc)}"
        )
    return HealthCheck(
        "Provider", True, False, f"Groww authenticated ({groww.auth_mode})"
    )


def _describe(exc: ProviderError) -> str:
    """Return the provider error's message plus any upstream detail (code+body)."""
    return f"{exc.message} — {exc.details}" if exc.details else exc.message


async def _live_auth_probe(groww: GrowwSettings) -> None:
    """Obtain an access token from Groww, releasing the client afterwards."""
    from app.providers.groww.http_client import GrowwHTTPClient
    from app.providers.groww.session import GrowwSessionManager

    http = GrowwHTTPClient(groww)
    try:
        await GrowwSessionManager(groww, http).get_access_token()
    finally:
        await http.aclose()


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
