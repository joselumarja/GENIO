from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genio.artifacts import Artifact
    from genio.evaluation.task import EvaluationTask


class EvaluationState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class EvaluationHandle:
    id: str
    task_id: str | None = None
    backend_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    payload: Any = None


class Backend(ABC):
    """Execution mechanism for evaluation tasks."""

    @abstractmethod
    def submit(self, task: EvaluationTask) -> EvaluationHandle:
        raise NotImplementedError

    def submit_batch(self, tasks: Sequence[EvaluationTask]) -> list[EvaluationHandle]:
        return [self.submit(task) for task in tasks]

    @abstractmethod
    def collect(self, handle: EvaluationHandle) -> list[Artifact]:
        raise NotImplementedError

    @abstractmethod
    def status(self, handle: EvaluationHandle) -> EvaluationState:
        raise NotImplementedError

    def error(self, handle: EvaluationHandle) -> str | None:
        return None

    def cancel(self, handle: EvaluationHandle) -> None:
        raise NotImplementedError
