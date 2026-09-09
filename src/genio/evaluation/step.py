"""Declarative evaluation-step extension contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

from genio.artifacts import Artifact
from genio.core.individual import Individual
from genio.evaluation.task import EvaluationTask


class EvaluationStep(ABC):
    """Describe one node in an evaluation workflow.

    Subclasses expose static dependency information and create an
    :class:`EvaluationTask` for each individual reaching the step.

    Attributes:
        id: Workflow-unique identifier without ``.`` characters. It namespaces
            produced artifacts and metrics.
        depends_on: IDs of steps that must complete first.
        required_artifacts: Mapping from qualified ``step_id.artifact_name`` keys
            to expected artifact classes. Producers must also appear directly in
            ``depends_on``.
        produced_artifacts: Mapping from local artifact names to the concrete
            artifact classes that a successful task may return.
        task_type: Concrete task class that :meth:`create_task` must return.
    """

    id: str
    depends_on: tuple[str, ...] = ()
    required_artifacts: Mapping[str, type[Artifact]] = MappingProxyType({})
    produced_artifacts: Mapping[str, type[Artifact]] = MappingProxyType({})
    task_type: type[EvaluationTask] = cast(type[EvaluationTask], EvaluationTask)

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return structural workflow configuration for checkpoint validation.

        Concrete steps with additional behavior-affecting configuration should
        extend this mapping. A session refuses checkpointing when the base
        implementation has not been overridden.
        """

        signature: dict[str, Any] = {
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
        if self.produced_artifacts:
            signature["produced_artifacts"] = {
                key: f"{artifact_type.__module__}.{artifact_type.__qualname__}"
                for key, artifact_type in sorted(self.produced_artifacts.items())
            }
        return signature

    @abstractmethod
    def create_task(
        self,
        individual: Individual,
        artifacts: Mapping[str, Artifact],
    ) -> EvaluationTask:
        """Create an executable task for one individual.

        Args:
            individual: Candidate being evaluated by this step.
            artifacts: Only the qualified artifacts declared by
                ``required_artifacts`` after executor type validation.

        Returns:
            An instance of the class declared by ``task_type``.
        """

        raise NotImplementedError
