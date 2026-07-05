"""Canonical timezone definitions.

Titan's market domain operates exclusively in India Standard Time. Centralising
the zone here prevents stray timezone strings and accidental UTC assumptions.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

#: Name of the market timezone. The single source of truth for the IST zone.
MARKET_TIMEZONE_NAME = "Asia/Kolkata"

#: India Standard Time. All market timestamps are expressed in this zone.
INDIA_TZ = ZoneInfo(MARKET_TIMEZONE_NAME)
