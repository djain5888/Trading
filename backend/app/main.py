"""Application entry point.

Exposes the ASGI ``app`` object for servers such as Uvicorn:

    uvicorn app.main:app --reload

Running this module directly starts a development server.
"""

from __future__ import annotations

from app.config.settings import get_settings
from app.core.application import create_app

app = create_app()


def run() -> None:
    """Run a development server using Uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()
