"""The paper-trade repository port.

Storage-independent interface. Concrete adapters (DuckDB, in-memory) implement
it so the engine persists trades without knowing the backing store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.paper.models import PaperTrade


class PaperTradeRepository(ABC):
    """Persistence port for paper trades."""

    @abstractmethod
    async def upsert(self, trade: PaperTrade) -> None:
        """Insert or replace a trade by its id."""

    @abstractmethod
    async def all(self) -> list[PaperTrade]:
        """Return every stored trade."""
