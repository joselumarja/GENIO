from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any
from uuid import uuid4

from genio.artifacts import Artifact
from genio.backend.base import (
    BackendError,
    BackendShutdownError,
    EvaluationHandle,
    EvaluationState,
    UnknownEvaluationHandleError,
)
from genio.backend.local import _LocalExecutionBackend
from genio.evaluation.task import EvaluationTask, ExecutionContext


@dataclass(slots=True)
class _ParallelEvaluationRecord:
    task: EvaluationTask
    context: ExecutionContext
    workspace_key: tuple[str, str]
    state: EvaluationState = EvaluationState.PENDING
    future: Future[list[Artifact]] | None = None
    artifacts: list[Artifact] | None = None
    exception: BaseException | None = None
    error: str | None = None
    capacity_released: bool = False
    cancel_requested: bool = False


class ParallelLocalBackend(_LocalExecutionBackend):
    """Run local evaluation tasks concurrently in a bounded thread pool."""

    def __init__(
        self,
        *,
        max_workers: int,
        max_pending: int | None = None,
        base_work_dir: str | Path | None = None,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive.")
        if max_pending is not None and max_pending <= 0:
            raise ValueError("max_pending must be positive when provided.")

        super().__init__(
            base_work_dir=base_work_dir,
            run_id=run_id,
            metadata=metadata,
        )
        self.max_workers = max_workers
        self.max_pending = max_pending
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="genio-local",
        )
        self._records: dict[str, _ParallelEvaluationRecord] = {}
        self._active_workspaces: set[tuple[str, str]] = set()
        self._lock = Lock()
        self._capacity = BoundedSemaphore(max_pending) if max_pending is not None else None

    def submit(self, task: EvaluationTask) -> EvaluationHandle:
        """Submit a local task for asynchronous thread-pool execution."""

        if self._capacity is not None:
            self._capacity.acquire()

        try:
            with self._lock:
                if self._shutdown:
                    raise BackendShutdownError("Cannot submit work after backend shutdown.")

                workspace_key = (task.individual.id, task.step_id or "task")
                if workspace_key in self._active_workspaces:
                    raise BackendError(
                        "Cannot run concurrent tasks in the same workspace: "
                        f"{workspace_key!r}."
                    )

                handle = EvaluationHandle(
                    id=uuid4().hex,
                    task_id=task.task_id,
                    backend_id=self.__class__.__name__,
                    metadata={"run_id": self.run_id},
                    payload=task,
                )
                record = _ParallelEvaluationRecord(
                    task=task,
                    context=self.create_context(task),
                    workspace_key=workspace_key,
                )
                self._records[handle.id] = record
                self._active_workspaces.add(workspace_key)
                try:
                    record.future = self._executor.submit(self._run_task, handle.id)
                except Exception:
                    self._records.pop(handle.id, None)
                    self._active_workspaces.discard(workspace_key)
                    raise
                return handle
        except Exception:
            if self._capacity is not None:
                self._capacity.release()
            raise

    def collect(self, handle: EvaluationHandle) -> list[Artifact]:
        """Wait for a local task and return its artifacts or original exception."""

        with self._lock:
            record = self._require_record(handle)
            future = record.future
        assert future is not None

        try:
            artifacts = future.result()
        except CancelledError as exc:
            raise BackendError(f"Evaluation {handle.id!r} was cancelled.") from exc
        return list(artifacts)

    def status(self, handle: EvaluationHandle) -> EvaluationState:
        """Return the current state of a parallel local evaluation."""

        with self._lock:
            return self._require_record(handle).state

    def error(self, handle: EvaluationHandle) -> str | None:
        """Return the captured task error after a failed evaluation."""

        with self._lock:
            return self._require_record(handle).error

    def cancel(self, handle: EvaluationHandle) -> bool:
        """Cancel a queued evaluation or stop its active external commands."""

        with self._lock:
            record = self._require_record(handle)
            if record.state in {
                EvaluationState.DONE,
                EvaluationState.FAILED,
                EvaluationState.CANCELLED,
            } or record.cancel_requested:
                return False
            if (
                record.state is EvaluationState.PENDING
                and record.future is not None
                and record.future.cancel()
            ):
                record.cancel_requested = True
                self._mark_cancelled(record)
                return True
            record.cancel_requested = True
            context = record.context

        context.cancel()
        return True

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Stop accepting work and optionally cancel all unfinished tasks."""

        with self._lock:
            self._shutdown = True
            unfinished_handles = (
                [
                    EvaluationHandle(id=handle_id, backend_id=self.__class__.__name__)
                    for handle_id, record in self._records.items()
                    if record.state in {EvaluationState.PENDING, EvaluationState.RUNNING}
                ]
                if cancel_futures
                else []
            )

        for handle in unfinished_handles:
            self.cancel(handle)
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _run_task(self, handle_id: str) -> list[Artifact]:
        with self._lock:
            record = self._records[handle_id]
            if record.cancel_requested:
                self._mark_cancelled(record)
                cancelled = True
            else:
                record.state = EvaluationState.RUNNING
                cancelled = False

        if cancelled:
            raise CancelledError()

        try:
            artifacts = list(record.task.run(record.context))
        except BaseException as exc:
            with self._lock:
                cancelled = isinstance(exc, CancelledError)
                if cancelled:
                    self._mark_cancelled(record)
                else:
                    record.exception = exc
                    record.error = f"{type(exc).__name__}: {exc}"
                    record.state = EvaluationState.FAILED
                    self._active_workspaces.discard(record.workspace_key)
                    self._release_capacity(record)
            if cancelled:
                raise CancelledError() from exc
            raise

        with self._lock:
            cancelled = record.cancel_requested
            if cancelled:
                self._mark_cancelled(record)
            else:
                record.artifacts = artifacts
                record.state = EvaluationState.DONE
                self._active_workspaces.discard(record.workspace_key)
                self._release_capacity(record)
        if cancelled:
            raise CancelledError()
        return artifacts

    def _require_record(self, handle: EvaluationHandle) -> _ParallelEvaluationRecord:
        try:
            return self._records[handle.id]
        except KeyError as exc:
            raise UnknownEvaluationHandleError(
                f"Evaluation handle {handle.id!r} does not belong to this backend."
            ) from exc

    def _release_capacity(self, record: _ParallelEvaluationRecord) -> None:
        if self._capacity is not None and not record.capacity_released:
            record.capacity_released = True
            self._capacity.release()

    def _mark_cancelled(self, record: _ParallelEvaluationRecord) -> None:
        record.state = EvaluationState.CANCELLED
        self._active_workspaces.discard(record.workspace_key)
        self._release_capacity(record)


__all__ = ["ParallelLocalBackend"]
