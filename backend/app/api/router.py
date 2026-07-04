"""Aggregate API router.

Individual feature routers are included here so that ``main`` only needs to
mount a single router. As new modules gain HTTP surfaces, add their routers
below.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api import health

api_router = APIRouter()
api_router.include_router(health.router)
