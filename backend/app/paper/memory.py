"""In-memory paper-trade repository (tests and ephemeral runs)."""

from __future__ import annotations

from app.paper.models import PaperTrade
from app.paper.repository import PaperTradeRepository


class InMemoryPaperTradeRepository(PaperTradeRepository):
    """A dict-backed :class:`PaperTradeRepository`."""

    def __init__(self) -> None:
        """Initialise an empty store."""
        self._trades: dict[str, PaperTrade] = {}

    async def upsert(self, trade: PaperTrade) -> None:
        """Insert or replace a trade by its id."""
        self._trades[trade.id] = trade

    async def all(self) -> list[PaperTrade]:
        """Return every stored trade in insertion order."""
        return list(self._trades.values())
