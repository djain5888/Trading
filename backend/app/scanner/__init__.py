"""Titan market scanner package.

Discovers and ranks candidate trading opportunities. It never emits buy/sell
decisions, targets or stops — it only filters and scores symbols.
"""

from app.scanner.base import Scanner
from app.scanner.dependencies import (
    get_scanner_engine,
    get_scanner_registry,
)
from app.scanner.engine import ScannerEngine
from app.scanner.exceptions import (
    ScannerConfigError,
    ScannerError,
    UnknownScannerError,
)
from app.scanner.models import ScannerContext, ScannerResult
from app.scanner.registry import ScannerRegistry, default_registry

__all__ = [
    "Scanner",
    "ScannerConfigError",
    "ScannerContext",
    "ScannerEngine",
    "ScannerError",
    "ScannerRegistry",
    "ScannerResult",
    "UnknownScannerError",
    "default_registry",
    "get_scanner_engine",
    "get_scanner_registry",
]
