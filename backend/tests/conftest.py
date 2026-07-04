"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from app.core.application import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide a FastAPI test client bound to a fresh application instance."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
