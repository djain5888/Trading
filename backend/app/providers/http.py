"""Transport-agnostic HTTP request description.

A :class:`RequestSpec` captures *what* request a provider wants to make
without prescribing *how* it is sent. This keeps request construction (pure,
easily tested) separate from the transport (I/O), and ensures no endpoint host
is hardcoded — the base URL is supplied from configuration at send time.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HttpMethod(StrEnum):
    """HTTP verbs used by providers."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class RequestSpec(BaseModel):
    """A declarative description of a single HTTP request."""

    model_config = ConfigDict(frozen=True)

    method: HttpMethod = Field(description="HTTP method.")
    path: str = Field(description="Path relative to the provider base URL.")
    params: dict[str, str] = Field(
        default_factory=dict, description="Query-string parameters."
    )
    headers: dict[str, str] = Field(
        default_factory=dict, description="Request headers."
    )
    json_body: dict[str, Any] | None = Field(
        default=None, description="Optional JSON request body."
    )

    def full_url(self, base_url: str) -> str:
        """Join a configured base URL with this request's path.

        Args:
            base_url: The provider base URL, supplied from configuration.

        Returns:
            The absolute request URL.
        """
        return f"{base_url.rstrip('/')}/{self.path.lstrip('/')}"
