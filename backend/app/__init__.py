"""Titan backend application package.

Titan is an AI quant trading platform for the Indian stock market. This
package hosts the FastAPI application and its supporting modules.
"""

from app.config.settings import get_settings

__all__ = ["__version__", "get_settings"]

__version__: str = get_settings().app_version
