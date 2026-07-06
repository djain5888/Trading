"""Enumerations for the live market collector."""

from __future__ import annotations

from enum import StrEnum


class CollectionMode(StrEnum):
    """The mode a collector runs in."""

    LIVE = "live"
    """Collect real live market data and persist snapshots."""
    PAPER = "paper"
    """Collect live data for paper trading (same collection, no trading)."""
    BACKTEST_REPLAY = "backtest_replay"
    """Replay historical data as a live feed. Stub only."""
