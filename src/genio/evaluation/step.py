from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from genio.core.artifact import Artifact
from genio.core.individual import Individual
from genio.evaluation.task import EvaluationTask


class EvaluationStep(ABC):
    """Abstract evaluation step that creates backend tasks."""

    id: str
    depends_on: tuple[str, ...] = ()
    task_type: type[EvaluationTask] = EvaluationTask

    @abstractmethod
    def create_task(
        self,
        individual: Individual,
        artifacts: Mapping[str, Artifact],
    ) -> EvaluationTask:
        raise NotImplementedError
