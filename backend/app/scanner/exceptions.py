"""Scanner engine exceptions."""

from __future__ import annotations


class ScannerError(Exception):
    """Base class for all scanner-engine errors."""


class UnknownScannerError(ScannerError):
    """Raised when a scanner name is not registered."""


class ScannerConfigError(ScannerError):
    """Raised when a scanner is configured with invalid parameters."""
