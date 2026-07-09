from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

from genio.backend.base import Backend
from genio.core.artifact import Artifact, MetricArtifact
from genio.core.individual import Individual
from genio.core.result import Result
from genio.evaluation.step import EvaluationStep
from genio.evaluation.workflow import EvaluationWorkflow


class EvaluationExecutionError(Exception):
    """Raised when an evaluation step creates an invalid task."""


class EvaluationExecutor:
    """Executes an evaluation workflow for individuals using a backend."""

    def __init__(self, workflow: EvaluationWorkflow, backend: Backend) -> None:
        self.workflow = workflow
        self.backend = backend

    def evaluate(self, individual: Individual) -> Result:
        accumulated_artifacts: dict[str, Artifact] = {}
        accumulated_metrics: dict[str, float] = {}

        for step in self.workflow.execution_order():
            task = step.create_task(individual, accumulated_artifacts)
            if not isinstance(task, step.task_type):
                msg = (
                    f"Evaluation step {step.id!r} declared task type "
                    f"{step.task_type.__name__}, but created "
                    f"{type(task).__name__}."
                )
                raise EvaluationExecutionError(msg)
            handle = self.backend.submit(task)
            artifacts = self.backend.collect(handle)
            for artifact in artifacts:
                self._accumulate_artifact(step, artifact, accumulated_artifacts)
                if isinstance(artifact, MetricArtifact):
                    self._accumulate_metrics(step, artifact, accumulated_metrics)

        return Result.success(individual.id, metrics=accumulated_metrics)

    def evaluate_many(self, individuals: Sequence[Individual]) -> list[Result]:
        return [self.evaluate(individual) for individual in individuals]

    def _accumulate_artifact(
        self,
        step: EvaluationStep,
        artifact: Artifact,
        accumulated_artifacts: dict[str, Artifact],
    ) -> None:
        artifact_key = f"{step.id}.{artifact.name}"
        if artifact_key in accumulated_artifacts:
            msg = f"Duplicate artifact key {artifact_key!r}."
            raise EvaluationExecutionError(msg)
        accumulated_artifacts[artifact_key] = artifact

    def _accumulate_metrics(
        self,
        step: EvaluationStep,
        artifact: MetricArtifact,
        accumulated_metrics: dict[str, float],
    ) -> None:
        for metric_name, value in artifact.metrics().items():
            metric_key = f"{step.id}.{metric_name}"
            if metric_key in accumulated_metrics:
                msg = f"Duplicate metric key {metric_key!r}."
                raise EvaluationExecutionError(msg)
            if isinstance(value, bool) or not isinstance(value, Real):
                msg = f"Metric {metric_key!r} must be numeric, got {value!r}."
                raise EvaluationExecutionError(msg)
            accumulated_metrics[metric_key] = float(value)
