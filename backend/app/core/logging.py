"""Logging configuration for the Titan backend."""

from __future__ import annotations

import logging
from logging.config import dictConfig

from app.config.settings import Settings


def configure_logging(settings: Settings) -> None:
    """Configure the root logger from application settings.

    Args:
        settings: The active application settings.
    """
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": (
                        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
                    ),
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                },
            },
            "root": {
                "level": settings.log_level.upper(),
                "handlers": ["console"],
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
