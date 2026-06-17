from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResultStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Result:
    """Normalized output produced by a candidate evaluation."""

    candidate_id: str
    status: ResultStatus
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def success(
        cls,
        candidate_id: str,
        metrics: dict[str, float] | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> "Result":
        return cls(
            candidate_id=candidate_id,
            status=ResultStatus.SUCCESS,
            metrics=metrics or {},
            artifacts=artifacts or {},
        )

    @classmethod
    def failed(
        cls,
        candidate_id: str,
        error: str,
        metrics: dict[str, float] | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> "Result":
        return cls(
            candidate_id=candidate_id,
            status=ResultStatus.FAILED,
            metrics=metrics or {},
            artifacts=artifacts or {},
            error=error,
        )
