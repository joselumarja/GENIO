"""Validation and stable ordering of artifact-dependent evaluation steps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from genio.evaluation.artifact import Artifact
from genio.evaluation.step import EvaluationStep


class EvaluationWorkflowError(Exception):
    """Base error for invalid evaluation workflows."""


@dataclass(frozen=True, slots=True)
class EvaluationWorkflow:
    """Declare and validate an artifact-dependent evaluation workflow.

    Attributes:
        steps: Steps in preferred order; artifact requirements may reorder them.

    Construction rejects duplicate or dotted step IDs, invalid requirements, and
    cyclic artifact dependencies. An empty workflow is
    valid and evaluates individuals successfully without metrics.
    """

    steps: tuple[EvaluationStep, ...]

    def __post_init__(self) -> None:
        self._validate()

    def execution_order(self) -> tuple[EvaluationStep, ...]:
        """Order producers before consumers, preserving declaration-order ties."""
        dependencies = {
            step.id: {key.partition(".")[0] for key in step.required_artifacts}
            for step in self.steps
        }
        ordered: list[EvaluationStep] = []
        completed: set[str] = set()
        while len(ordered) < len(self.steps):
            ready = next(
                (
                    step
                    for step in self.steps
                    if step.id not in completed
                    and dependencies[step.id].issubset(completed)
                ),
                None,
            )
            if ready is None:
                raise EvaluationWorkflowError(
                    "Evaluation workflow contains cyclic dependencies."
                )
            ordered.append(ready)
            completed.add(ready.id)
        return tuple(ordered)

    def checkpoint_signature(self) -> list[Mapping[str, object]]:
        """Return each step's signature in execution order."""
        return [step.checkpoint_signature() for step in self.execution_order()]

    def _validate(self) -> None:
        """Validate identifiers, artifact contracts, and inferred dependencies."""
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
        for step in self.steps:
            self._validate_required_artifacts(step, known_ids)
        self.execution_order()

    @staticmethod
    def _validate_required_artifacts(
        step: EvaluationStep,
        known_ids: set[str],
    ) -> None:
        """Validate required types and references to other workflow steps."""
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
            if (
                not separator
                or not producer_id
                or not artifact_name
                or "." in artifact_name
            ):
                raise EvaluationWorkflowError(
                    f"Evaluation step {step.id!r} artifact requirement {artifact_key!r} "
                    "must use the qualified form 'step_id.artifact_name'."
                )
            if producer_id == step.id:
                raise EvaluationWorkflowError(
                    f"Evaluation step {step.id!r} cannot require its own artifact {artifact_key!r}."
                )
            if not isinstance(artifact_type, type) or not issubclass(
                artifact_type, Artifact
            ):
                raise EvaluationWorkflowError(
                    f"Evaluation step {step.id!r} artifact requirement {artifact_key!r} "
                    "must declare an Artifact subclass."
                )
            if producer_id not in known_ids:
                raise EvaluationWorkflowError(
                    f"Evaluation step {step.id!r} requires artifact {artifact_key!r}, "
                    f"but producer step {producer_id!r} does not exist."
                )
