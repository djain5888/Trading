"""Data-quality reporting value objects."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.market.historical.enums import IssueType


class ValidationIssue(BaseModel):
    """A single data-quality issue with optional location."""

    model_config = ConfigDict(frozen=True)

    issue_type: IssueType = Field(description="The kind of issue detected.")
    message: str = Field(description="Human-readable description.")
    timestamp: datetime | None = Field(
        default=None, description="Candle timestamp the issue relates to, if any."
    )


class DataQualityReport(BaseModel):
    """Aggregate statistics describing the outcome of an import."""

    model_config = ConfigDict(frozen=True)

    imported: int = Field(default=0, ge=0, description="New candles stored.")
    skipped: int = Field(default=0, ge=0, description="Candles not stored.")
    duplicates: int = Field(default=0, ge=0, description="Candles already present.")
    repaired: int = Field(default=0, ge=0, description="Existing candles updated.")
    invalid: int = Field(default=0, ge=0, description="Candles failing validation.")
    gaps_found: int = Field(default=0, ge=0, description="Missing candle slots.")
    issues: tuple[ValidationIssue, ...] = Field(
        default_factory=tuple, description="Detailed issues detected."
    )

    @classmethod
    def empty(cls) -> DataQualityReport:
        """Return a report describing that nothing was processed."""
        return cls()
