"""Evaluation tasks and their minimal local runtime context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from genio.evaluation.artifact import Artifact
from genio.core.individual import Individual


@dataclass(frozen=True, slots=True)
class EvaluationOutput:
    """Local metrics and named artifacts produced by one task."""

    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, Artifact] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactInput:
    """Qualified upstream artifact needed by a task."""

    source: str
    artifact_type: type[Artifact]


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Expose the task's prepared workspace and shared local resources."""

    workspace: Path
    resources: Mapping[str, Any] = field(default_factory=dict)


class _TaskTimeoutInterrupted(BaseException):
    """Interrupt a worker so it can perform cooperative timeout cleanup."""


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationTask(ABC):
    """Executable work for one individual in one workflow step.

    Attributes:
        individual: Candidate whose configuration the task evaluates.
        id: Optional explicit task identifier.
        step_id: Producing workflow-step identifier used for workspace names.
        artifacts: Resolved inputs keyed by qualified producer and artifact name.
        metadata: Fixed task configuration.
        runtime_state: Mutable process-local state shared by ``run`` and
            ``on_timeout``. Changes are not returned to the session.
        requires: Class-level declarations of qualified input artifacts.
        uses_domains: Optional direct design domains that affect this task's
            output. ``None`` conservatively selects every domain.
    """

    individual: Individual
    id: str | None = None
    step_id: str | None = None
    artifacts: Mapping[str, Artifact] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    runtime_state: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    requires: ClassVar[tuple[ArtifactInput, ...]] = ()
    uses_domains: ClassVar[tuple[str, ...] | None] = None

    @property
    def task_id(self) -> str:
        """Return the stable identifier for this evaluation task."""

        if self.id is not None:
            return self.id
        if self.step_id is not None:
            return f"{self.individual.id}:{self.step_id}"
        return self.individual.id

    @abstractmethod
    def run(self, context: TaskContext) -> EvaluationOutput:
        """Run in the prepared workspace and return metrics and artifacts.

        Persistent artifact files should be placed under ``context.workspace``.
        Exceptions are converted to failed results by the executor.
        """

    def on_timeout(self, context: TaskContext) -> None:
        """Optionally release resources outside the worker's process group."""

    def _execute(self, context: TaskContext) -> EvaluationOutput:
        """Wrap ``run`` with cooperative timeout cleanup inside the worker."""
        try:
            return self.run(context)
        except _TaskTimeoutInterrupted:
            self.on_timeout(context)
            raise


def _workspace_segment(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).is_absolute()
        or len(Path(value).parts) != 1
    ):
        raise ValueError(f"{name} must be a safe workspace path segment.")
    return value
