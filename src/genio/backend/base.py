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
    """Enumerate the lifecycle states of an evaluation."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BackendError(RuntimeError):
    """Base error raised by evaluation backends."""


class BackendShutdownError(BackendError):
    """Raised when work is submitted to a backend after shutdown."""


class UnknownEvaluationHandleError(BackendError):
    """Raised when a handle does not belong to a backend."""


@dataclass(frozen=True, slots=True)
class EvaluationHandle:
    """Identify an evaluation submitted to a backend."""

    id: str
    task_id: str | None = None
    backend_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    payload: Any = None


class Backend(ABC):
    """Execution mechanism for evaluation tasks."""

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return execution configuration relevant to resumed task semantics."""

        return {"type": f"{type(self).__module__}.{type(self).__qualname__}"}

    @abstractmethod
    def submit(self, task: EvaluationTask) -> EvaluationHandle:
        """Submit an evaluation task and return its handle."""

        raise NotImplementedError

    def submit_batch(self, tasks: Sequence[EvaluationTask]) -> list[EvaluationHandle]:
        """Submit multiple evaluation tasks and return their handles."""

        return [self.submit(task) for task in tasks]

    @abstractmethod
    def collect(self, handle: EvaluationHandle) -> list[Artifact]:
        """Wait for and return artifacts, re-raising task execution failures."""

        raise NotImplementedError

    def collect_batch(self, handles: Sequence[EvaluationHandle]) -> list[list[Artifact]]:
        """Collect multiple evaluations in handle order."""

        return [self.collect(handle) for handle in handles]

    @abstractmethod
    def status(self, handle: EvaluationHandle) -> EvaluationState:
        """Return the current state of a submitted evaluation."""

        raise NotImplementedError

    def error(self, handle: EvaluationHandle) -> str | None:
        """Return the error reported for a submitted evaluation."""

        return None

    def cancel(self, handle: EvaluationHandle) -> bool:
        """Cancel a pending evaluation and report whether cancellation succeeded."""

        raise NotImplementedError

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Release backend resources and optionally cancel pending evaluations."""

    def __enter__(self) -> "Backend":
        """Return this backend as a managed resource."""

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Shut down backend resources when leaving a context manager."""

        self.shutdown()


__all__ = [
    "Backend",
    "BackendError",
    "BackendShutdownError",
    "EvaluationHandle",
    "EvaluationState",
    "UnknownEvaluationHandleError",
]
