"""Backend interface for scheduling and controlling evaluation tasks."""

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
    """Identify one evaluation submitted to a specific backend.

    Attributes:
        id: Backend-generated handle identifier.
        task_id: Optional identifier of the submitted evaluation task.
        backend_id: Optional identity of the backend that owns the handle.
        metadata: Scheduler-specific submission information.
        payload: Private backend data required to collect or control execution.

    Handles are opaque capabilities. They should only be passed back to the
    backend instance that created them.
    """

    id: str
    task_id: str | None = None
    backend_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    payload: Any = None


class Backend(ABC):
    """Schedule evaluation tasks and expose their lifecycle through handles.

    A backend creates an :class:`genio.ExecutionContext` for each task and may run
    synchronously, on local workers, or through a remote host. Consequently,
    :meth:`submit` is not guaranteed to be non-blocking; callers should rely on
    handles and lifecycle methods rather than a particular scheduling strategy.

    Backends own resources independently from optimization sessions. They can be
    used as context managers to guarantee :meth:`shutdown` is called.
    """

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return execution configuration relevant to resumed task semantics.

        Stateful or configurable backends should extend the base type identity
        with every setting that could change evaluation results after restoration.
        """

        return {"type": f"{type(self).__module__}.{type(self).__qualname__}"}

    @abstractmethod
    def submit(self, task: EvaluationTask) -> EvaluationHandle:
        """Submit an evaluation task and return a backend-owned handle.

        Implementations may finish the task before returning or schedule it
        asynchronously.

        Raises:
            BackendShutdownError: If the backend no longer accepts work.
        """

        raise NotImplementedError

    def submit_batch(self, tasks: Sequence[EvaluationTask]) -> list[EvaluationHandle]:
        """Submit tasks in order and return their corresponding handles.

        The default implementation repeatedly calls :meth:`submit`; it provides
        no all-or-nothing guarantee if a later submission fails.
        """

        return [self.submit(task) for task in tasks]

    @abstractmethod
    def collect(self, handle: EvaluationHandle) -> list[Artifact]:
        """Wait for a submitted task and return its produced artifacts.

        Raises:
            UnknownEvaluationHandleError: If the handle is not owned by this
                backend.
            Exception: The original task failure may be re-raised after execution.
        """

        raise NotImplementedError

    def collect_batch(self, handles: Sequence[EvaluationHandle]) -> list[list[Artifact]]:
        """Collect multiple evaluations and preserve handle order.

        The default implementation waits sequentially even when the tasks are
        already executing concurrently.
        """

        return [self.collect(handle) for handle in handles]

    @abstractmethod
    def status(self, handle: EvaluationHandle) -> EvaluationState:
        """Return the current lifecycle state of a submitted evaluation."""

        raise NotImplementedError

    def error(self, handle: EvaluationHandle) -> str | None:
        """Return a task failure description when the backend provides one."""

        return None

    def cancel(self, handle: EvaluationHandle) -> bool:
        """Request cancellation of an unfinished evaluation.

        Returns:
            Whether this call successfully initiated cancellation. Concrete
            backends define what work can be interrupted.
        """

        raise NotImplementedError

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Release backend resources and optionally cancel pending evaluations.

        Args:
            wait: Wait for running work and worker resources to terminate.
            cancel_futures: Request cancellation of work that has not started.

        The base implementation has no resources to release.
        """

    def cleanup_individual_workspace(self, individual_id: str) -> None:
        """Remove one completed individual's workspace when supported.

        Backends that do not own filesystem workspaces may keep this no-op.
        """

    def __enter__(self) -> "Backend":
        """Return this backend as a managed resource."""

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Shut down backend resources when leaving a context manager."""

        self.shutdown(cancel_futures=exc_type is not None)


__all__ = [
    "Backend",
    "BackendError",
    "BackendShutdownError",
    "EvaluationHandle",
    "EvaluationState",
    "UnknownEvaluationHandleError",
]
