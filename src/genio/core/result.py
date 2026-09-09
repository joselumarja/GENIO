"""Normalized success and failure values returned by evaluation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ResultStatus(str, Enum):
    """Status values for an individual evaluation result."""

    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Result:
    """Contain the normalized output of an individual evaluation.

    Attributes:
        individual_id: Identifier of the evaluated individual.
        status: Whether the workflow completed successfully.
        metrics: Numeric measurements keyed as ``step_id.metric_name``.
        error: Human-readable failure description, or ``None`` on success.
        metadata: Executor metadata such as per-step cache information.
    """

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
        """Create a successful result for an individual.

        Args:
            individual_id: Identifier of the evaluated candidate.
            metrics: Numeric measurements produced by metric artifacts.
            metadata: Additional executor information to attach to the result.

        Returns:
            A result whose status is :attr:`ResultStatus.SUCCESS`.
        """
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
        """Create a failed result while preserving any partial metrics.

        Args:
            individual_id: Identifier of the evaluated candidate.
            error: Human-readable description of the execution failure.
            metrics: Metrics produced by steps completed before the failure.
            metadata: Additional executor information to attach to the result.

        Returns:
            A result whose status is :attr:`ResultStatus.FAILED`.
        """
        return cls(
            individual_id=individual_id,
            status=ResultStatus.FAILED,
            metrics=dict(metrics or {}),
            error=error,
            metadata=dict(metadata or {}),
        )
