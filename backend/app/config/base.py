"""Shared building blocks for the configuration modules.

Each logical group of settings (database, cache, broker, ...) lives in its own
module and subclasses :class:`BaseConfig`. A small helper builds a consistent
``SettingsConfigDict`` so every group reads the same ``.env`` file with the
same casing and extra-field rules while supplying its own environment prefix.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

#: Default environment file consulted by every settings group.
ENV_FILE = ".env"


def build_config(env_prefix: str = "") -> SettingsConfigDict:
    """Return a :class:`SettingsConfigDict` shared across settings groups.

    Args:
        env_prefix: Prefix applied to environment-variable names for the
            owning settings group (for example ``"POSTGRES_"``).

    Returns:
        A configuration dict suitable for assignment to ``model_config``.
    """
    return SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix=env_prefix,
        case_sensitive=False,
        extra="ignore",
    )


class BaseConfig(BaseSettings):
    """Base class for all Titan settings groups.

    Subclasses assign their own ``model_config`` via :func:`build_config` to
    declare an environment-variable prefix.
    """

    model_config = build_config()
