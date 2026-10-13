"""Generic declaration and factory for executable evaluation tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from inspect import signature
from typing import Any

from genio.evaluation.artifact import Artifact
from genio.core.individual import Individual
from genio.evaluation.task import ArtifactInput, EvaluationTask


@dataclass(frozen=True, slots=True)
class EvaluationStep:
    """Configure a task type and its artifact contracts for one workflow step.

    The task class declares ``requires``. The configuration is checked against
    its constructor once; only the resulting task travels to the worker process
    for each individual. Outputs are validated when produced and consumed.
    """

    id: str
    task_class: type[EvaluationTask]
    config: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_class, type) or not issubclass(
            self.task_class, EvaluationTask
        ):
            raise TypeError("task_class must be an EvaluationTask subclass.")
        if not isinstance(self.config, Mapping) or any(
            not isinstance(key, str) for key in self.config
        ):
            raise TypeError("Step config must map field names to values.")
        if {"individual", "artifacts", "step_id"} & self.config.keys():
            raise ValueError("Step config cannot override injected task fields.")
        declarations = self.task_class.requires
        if not isinstance(declarations, (tuple, list)) or any(
            not isinstance(item, ArtifactInput) for item in declarations
        ):
            raise TypeError("Task ArtifactInput declarations must be a list or tuple.")
        sources = [item.source for item in declarations]
        if any(not isinstance(source, str) for source in sources):
            raise TypeError("Task artifact sources must be strings.")
        if len(sources) != len(set(sources)):
            raise ValueError("Task declares duplicate artifact sources.")
        try:
            signature(self.task_class).bind(
                individual=None,
                artifacts={},
                step_id=self.id,
                **dict(self.config),
            )
        except TypeError as exc:
            raise ValueError(
                f"Invalid configuration for task {self.task_class.__name__}: {exc}"
            ) from exc

    @property
    def required_artifacts(self) -> Mapping[str, type[Artifact]]:
        """Return the task's qualified input declarations."""
        return {
            item.source: item.artifact_type for item in self.task_class.requires
        }

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Describe the task implementation, configuration and artifact contracts."""
        return {
            "id": self.id,
            "task_class": self.task_class,
            "timeout_seconds": self.timeout_seconds,
            "config": dict(self.config),
            "requires": self.required_artifacts,
        }

    def create_task(
        self,
        individual: Individual,
        artifacts: Mapping[str, Artifact],
    ) -> EvaluationTask:
        """Inject the individual and its resolved inputs into a new task."""
        return self.task_class(
            individual=individual,
            artifacts=artifacts,
            step_id=self.id,
            **dict(self.config),
        )
