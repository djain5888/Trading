"""Deployment environment definitions.

Titan runs in exactly one of three environments. Keeping the enum in its own
module avoids import cycles between the individual settings modules.
"""

from __future__ import annotations

from enum import StrEnum


class Environment(StrEnum):
    """Supported deployment environments.

    Inherits from :class:`enum.StrEnum` so values serialise transparently to
    JSON and compare equal to their plain-string form (e.g.
    ``Environment.PRODUCTION == "production"``).
    """

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

    @property
    def is_development(self) -> bool:
        """Return ``True`` when this is the development environment."""
        return self is Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        """Return ``True`` when this is the testing environment."""
        return self is Environment.TESTING

    @property
    def is_production(self) -> bool:
        """Return ``True`` when this is the production environment."""
        return self is Environment.PRODUCTION
