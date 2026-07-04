"""Tests for the system endpoints."""

from __future__ import annotations

from app.config.settings import get_settings
from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    """The health endpoint reports an ``ok`` status."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_returns_metadata(client: TestClient) -> None:
    """The version endpoint returns the configured app metadata."""
    settings = get_settings()

    response = client.get("/version")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == settings.app_name
    assert body["version"] == settings.app_version
    assert body["environment"] == settings.environment
