"""System endpoints: health and version.

These endpoints provide lightweight liveness and metadata information and
are intentionally free of business logic or external dependencies.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config.settings import Settings, get_settings
from app.schemas.health import HealthResponse, VersionResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """Return a static health indicator.

    Returns:
        A :class:`HealthResponse` with ``status='ok'``.
    """
    return HealthResponse()


@router.get("/version", response_model=VersionResponse, summary="Service metadata")
async def version(
    settings: Settings = Depends(get_settings),
) -> VersionResponse:
    """Return application name, version and environment.

    Args:
        settings: Injected application settings.

    Returns:
        A :class:`VersionResponse` describing the running service.
    """
    return VersionResponse(
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )
