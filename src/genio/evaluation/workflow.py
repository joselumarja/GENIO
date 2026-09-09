"""Validation and topological ordering of evaluation-step graphs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from genio.artifacts import Artifact
from genio.evaluation.step import EvaluationStep
from genio.evaluation.task import EvaluationTask


class EvaluationWorkflowError(Exception):
    """Base error for invalid evaluation workflows."""


@dataclass(frozen=True, slots=True)
class EvaluationWorkflow:
    """Declare and validate a directed acyclic graph of evaluation steps.

    Attributes:
        steps: Steps in stable declaration order. The order breaks ties between
            multiple nodes whose dependencies are simultaneously satisfied.

    Construction rejects duplicate or dotted step IDs, unknown dependencies,
    cyclic graphs, and malformed artifact requirements. An empty workflow is
    valid and evaluates individuals successfully without metrics.
    """

    steps: tuple[EvaluationStep, ...]

    def __post_init__(self) -> None:
        self._validate()

    def execution_order(self) -> tuple[EvaluationStep, ...]:
        """Return a stable topological ordering of all evaluation steps.

        Returns:
            Steps ordered after their dependencies. Ready-step ties preserve the
            order in ``steps``.

        Raises:
            EvaluationWorkflowError: If dependencies contain a cycle.
        """

        ordered: list[EvaluationStep] = []
        completed: set[str] = set()

        while len(ordered) < len(self.steps):
            ready = [
                step
                for step in self.steps
                if step.id not in completed and set(step.depends_on).issubset(completed)
            ]
            if not ready:
                msg = "Evaluation workflow contains cyclic dependencies"
                raise EvaluationWorkflowError(msg)

            for step in ready:
                ordered.append(step)
                completed.add(step.id)

        return tuple(ordered)

    def ready_steps(self, completed: set[str]) -> tuple[EvaluationStep, ...]:
        """Return uncompleted steps whose dependencies are all completed.

        This helper supports custom dynamic schedulers. The standard executor
        uses :meth:`execution_order` and processes each step as a batch barrier.
        """

        return tuple(
            step
            for step in self.steps
            if step.id not in completed and set(step.depends_on).issubset(completed)
        )

    def _validate(self) -> None:
        """Validate identifiers, dependencies, artifact contracts, and acyclicity."""
        ids = [step.id for step in self.steps]
        invalid_ids = [
            step_id
            for step_id in ids
            if not isinstance(step_id, str) or not step_id or "." in step_id
        ]
        if invalid_ids:
            raise EvaluationWorkflowError(
                "Evaluation step ids must be non-empty strings without '.': "
                f"{invalid_ids!r}"
            )
        duplicate_ids = {step_id for step_id in ids if ids.count(step_id) > 1}
        if duplicate_ids:
            msg = f"Duplicate evaluation step ids: {sorted(duplicate_ids)}"
            raise EvaluationWorkflowError(msg)

        known_ids = set(ids)
        missing_dependencies = {
            dependency
            for step in self.steps
            for dependency in step.depends_on
            if dependency not in known_ids
        }
        if missing_dependencies:
            msg = f"Unknown evaluation step dependencies: {sorted(missing_dependencies)}"
            raise EvaluationWorkflowError(msg)

        artifact_catalog: dict[str, type[Artifact]] = {}
        for step in self.steps:
            self._validate_produced_artifacts(step, artifact_catalog)
        for step in self.steps:
            self._validate_required_artifacts(step, artifact_catalog)

        self.execution_order()

    @staticmethod
    def _validate_produced_artifacts(
        step: EvaluationStep,
        artifact_catalog: dict[str, type[Artifact]],
    ) -> None:
        """Validate and register one step's local artifact outputs."""

        declarations = step.produced_artifacts
        if not isinstance(declarations, Mapping):
            raise EvaluationWorkflowError(
                f"Evaluation step {step.id!r} produced_artifacts must be a mapping."
            )
        if not isinstance(step.task_type, type) or not issubclass(
            step.task_type, EvaluationTask
        ):
            raise EvaluationWorkflowError(
                f"Evaluation step {step.id!r} task_type must be an "
                "EvaluationTask subclass."
            )
        for artifact_name, artifact_type in declarations.items():
            if (
                not isinstance(artifact_name, str)
                or not artifact_name
                or "." in artifact_name
            ):
                raise EvaluationWorkflowError(
                    f"Evaluation step {step.id!r} produced artifact names must be "
                    "non-empty local strings without '.'."
                )
            if not isinstance(artifact_type, type) or not issubclass(
                artifact_type, Artifact
            ):
                raise EvaluationWorkflowError(
                    f"Evaluation step {step.id!r} produced artifact {artifact_name!r} "
                    "must declare an Artifact subclass."
                )
            artifact_catalog[f"{step.id}.{artifact_name}"] = artifact_type

    @staticmethod
    def _validate_required_artifacts(
        step: EvaluationStep,
        artifact_catalog: Mapping[str, type[Artifact]],
    ) -> None:
        """Validate one step's qualified direct-dependency artifact mapping."""
        requirements = step.required_artifacts
        if not isinstance(requirements, Mapping):
            raise EvaluationWorkflowError(
                f"Evaluation step {step.id!r} required_artifacts must be a mapping."
            )

        for artifact_key, artifact_type in requirements.items():
            if not isinstance(artifact_key, str):
                raise EvaluationWorkflowError(
                    f"Evaluation step {step.id!r} artifact requirement keys must be strings."
                )
            producer_id, separator, artifact_name = artifact_key.partition(".")
            if not separator or not producer_id or not artifact_name:
                raise EvaluationWorkflowError(
                    f"Evaluation step {step.id!r} artifact requirement {artifact_key!r} "
                    "must use the qualified form 'step_id.artifact_name'."
                )
            if producer_id not in step.depends_on:
                raise EvaluationWorkflowError(
                    f"Evaluation step {step.id!r} artifact requirement {artifact_key!r} "
                    f"comes from {producer_id!r}, which is not a declared dependency."
                )
            if not isinstance(artifact_type, type) or not issubclass(
                artifact_type, Artifact
            ):
                raise EvaluationWorkflowError(
                    f"Evaluation step {step.id!r} artifact requirement {artifact_key!r} "
                    "must declare an Artifact subclass."
                )
            produced_type = artifact_catalog.get(artifact_key)
            if produced_type is None:
                raise EvaluationWorkflowError(
                    f"Evaluation step {step.id!r} requires artifact {artifact_key!r}, "
                    "but its producer does not declare it."
                )
            if not issubclass(produced_type, artifact_type):
                raise EvaluationWorkflowError(
                    f"Evaluation step {step.id!r} requires artifact {artifact_key!r} "
                    f"compatible with {artifact_type.__name__}, but its producer "
                    f"declares {produced_type.__name__}."
                )
