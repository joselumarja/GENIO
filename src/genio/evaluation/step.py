from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from genio.artifacts import Artifact
from genio.core.individual import Individual
from genio.evaluation.task import EvaluationTask


class EvaluationStep(ABC):
    """Abstract evaluation step that creates backend tasks."""

    id: str
    depends_on: tuple[str, ...] = ()
    required_artifacts: Mapping[str, type[Artifact]] = MappingProxyType({})
    task_type: type[EvaluationTask] = EvaluationTask

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return structural workflow configuration for checkpoint validation."""

        signature = {
            "id": self.id,
            "depends_on": list(self.depends_on),
            "step_type": f"{type(self).__module__}.{type(self).__qualname__}",
            "task_type": f"{self.task_type.__module__}.{self.task_type.__qualname__}",
        }
        if self.required_artifacts:
            signature["required_artifacts"] = {
                key: f"{artifact_type.__module__}.{artifact_type.__qualname__}"
                for key, artifact_type in sorted(self.required_artifacts.items())
            }
        return signature

    @abstractmethod
    def create_task(
        self,
        individual: Individual,
        artifacts: Mapping[str, Artifact],
    ) -> EvaluationTask:
        """Create a task using only artifacts declared in required_artifacts."""

        raise NotImplementedError
