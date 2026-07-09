from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ResultStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Result:
    """Normalized output produced by an individual evaluation."""

    individual_id: str
    status: ResultStatus
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def success(
        cls,
        individual_id: str,
        metrics: dict[str, float] | None = None,
    ) -> "Result":
        return cls(
            individual_id=individual_id,
            status=ResultStatus.SUCCESS,
            metrics=metrics or {},
        )

    @classmethod
    def failed(
        cls,
        individual_id: str,
        error: str,
        metrics: dict[str, float] | None = None,
    ) -> "Result":
        return cls(
            individual_id=individual_id,
            status=ResultStatus.FAILED,
            metrics=metrics or {},
            error=error,
        )
