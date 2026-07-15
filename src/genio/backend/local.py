from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from tempfile import mkdtemp
from typing import Any
from uuid import uuid4

from genio.backend.base import (
    Backend,
    BackendError,
    BackendShutdownError,
    EvaluationHandle,
    EvaluationState,
    UnknownEvaluationHandleError,
)
from genio.artifacts import Artifact
from genio.evaluation.task import EvaluationTask, ExecutionContext


class _LocalExecutionBackend(Backend):
    """Share local workspace and execution-context construction."""

    def __init__(
        self,
        base_work_dir: str | Path | None = None,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id or uuid4().hex
        self.metadata = dict(metadata or {})
        self.base_work_dir = self._resolve_base_work_dir(base_work_dir)
        self._shutdown = False

    def create_context(self, task: EvaluationTask) -> ExecutionContext:
        """Create the local execution context for an evaluation task."""

        return ExecutionContext(
            base_work_dir=self.base_work_dir,
            run_id=self.run_id,
            backend_id=self.__class__.__name__,
            metadata={
                **self.metadata,
                "task_id": task.task_id,
                "individual_id": task.individual.id,
                "step_id": task.step_id,
            },
        )

    @staticmethod
    def _resolve_base_work_dir(base_work_dir: str | Path | None) -> Path:
        path = Path(base_work_dir) if base_work_dir is not None else Path(mkdtemp(prefix="genio-"))
        path.mkdir(parents=True, exist_ok=True)
        return path


class LocalBackend(_LocalExecutionBackend):
    """Synchronous backend that runs evaluations in the current process."""

    def __init__(
        self,
        base_work_dir: str | Path | None = None,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            base_work_dir=base_work_dir,
            run_id=run_id,
            metadata=metadata,
        )
        self._artifacts: dict[str, list[Artifact]] = {}
        self._states: dict[str, EvaluationState] = {}
        self._errors: dict[str, str] = {}
        self._exceptions: dict[str, Exception] = {}

    def submit(self, task: EvaluationTask) -> EvaluationHandle:
        """Run an evaluation task synchronously and return its handle."""

        if self._shutdown:
            raise BackendShutdownError("Cannot submit work after backend shutdown.")
        handle = EvaluationHandle(
            id=uuid4().hex,
            task_id=task.task_id,
            backend_id=self.__class__.__name__,
            metadata={"run_id": self.run_id},
            payload=task,
        )
        self._states[handle.id] = EvaluationState.RUNNING
        try:
            context = self.create_context(task)
            artifacts = task.run(context)
            self._artifacts[handle.id] = list(artifacts)
            self._states[handle.id] = EvaluationState.DONE
        except Exception as exc:
            self._errors[handle.id] = f"{type(exc).__name__}: {exc}"
            self._exceptions[handle.id] = exc
            self._states[handle.id] = EvaluationState.FAILED
        return handle

    def collect(self, handle: EvaluationHandle) -> list[Artifact]:
        """Return the artifacts produced for an evaluation handle."""

        self._require_handle(handle)
        if self._states[handle.id] is EvaluationState.FAILED:
            raise self._exceptions[handle.id]
        if self._states[handle.id] is EvaluationState.CANCELLED:
            raise BackendError(f"Evaluation {handle.id!r} was cancelled.")
        return self._artifacts[handle.id]

    def status(self, handle: EvaluationHandle) -> EvaluationState:
        """Return the current state of an evaluation handle."""

        self._require_handle(handle)
        return self._states[handle.id]

    def error(self, handle: EvaluationHandle) -> str | None:
        """Return the error recorded for an evaluation handle."""

        self._require_handle(handle)
        return self._errors.get(handle.id)

    def cancel(self, handle: EvaluationHandle) -> bool:
        """Report that synchronous evaluations cannot be cancelled after submission."""

        self._require_handle(handle)
        return False

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Reject future submissions to this synchronous backend."""

        self._shutdown = True

    def _require_handle(self, handle: EvaluationHandle) -> None:
        if handle.id not in self._states:
            raise UnknownEvaluationHandleError(
                f"Evaluation handle {handle.id!r} does not belong to this backend."
            )
