from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ResultStatus(str, Enum):
    """Status values for an individual evaluation result."""

    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Result:
    """Normalized output produced by an individual evaluation."""

    individual_id: str
    status: ResultStatus
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        individual_id: str,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "Result":
        """Create a successful result for an individual."""
        return cls(
            individual_id=individual_id,
            status=ResultStatus.SUCCESS,
            metrics=dict(metrics or {}),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def failed(
        cls,
        individual_id: str,
        error: str,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "Result":
        """Create a failed result for an individual."""
        return cls(
            individual_id=individual_id,
            status=ResultStatus.FAILED,
            metrics=dict(metrics or {}),
            error=error,
            metadata=dict(metadata or {}),
        )
