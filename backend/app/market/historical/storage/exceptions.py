"""Storage-layer exceptions.

These wrap backend-specific failures so the domain never sees a raw DuckDB
error. All derive from :class:`StorageError`.
"""

from __future__ import annotations


class StorageError(Exception):
    """Base class for all candle-storage failures."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        """Initialise the error.

        Args:
            message: Human-readable description.
            cause: The underlying backend exception, if any.
        """
        super().__init__(message)
        self.message = message
        self.cause = cause


class StorageConnectionError(StorageError):
    """Raised when the storage backend cannot be opened."""


class SchemaInitializationError(StorageError):
    """Raised when the schema cannot be created or verified."""


class DuplicateCandleError(StorageError):
    """Raised when an insert violates the candle primary-key constraint."""


class QueryError(StorageError):
    """Raised when a read or write query fails."""
