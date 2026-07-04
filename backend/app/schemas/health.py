"""Pydantic schemas for system/meta endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response body for the ``/health`` endpoint."""

    status: Literal["ok"] = Field(
        default="ok", description="Overall service health indicator."
    )


class VersionResponse(BaseModel):
    """Response body for the ``/version`` endpoint."""

    name: str = Field(description="Application name.")
    version: str = Field(description="Application semantic version.")
    environment: str = Field(description="Active deployment environment.")
