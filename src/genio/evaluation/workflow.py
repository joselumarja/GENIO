from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from genio.artifacts import Artifact
from genio.evaluation.step import EvaluationStep


class EvaluationWorkflowError(Exception):
    """Base error for invalid evaluation workflows."""


@dataclass(frozen=True, slots=True)
class EvaluationWorkflow:
    """Declarative dependency graph of evaluation steps."""

    steps: tuple[EvaluationStep, ...]

    def __post_init__(self) -> None:
        self._validate()

    def execution_order(self) -> tuple[EvaluationStep, ...]:
        """Return evaluation steps in dependency order."""

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
        """Return steps whose dependencies have been completed."""

        return tuple(
            step
            for step in self.steps
            if step.id not in completed and set(step.depends_on).issubset(completed)
        )

    def _validate(self) -> None:
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

        for step in self.steps:
            self._validate_required_artifacts(step)

        self.execution_order()

    @staticmethod
    def _validate_required_artifacts(step: EvaluationStep) -> None:
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
