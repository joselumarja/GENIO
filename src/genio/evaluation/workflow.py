from __future__ import annotations

from dataclasses import dataclass

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

        self.execution_order()
