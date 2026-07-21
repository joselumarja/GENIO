from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
from genio.backend.ssh import _SSHExecutionBackend, _SSHExecutionContext
from genio.evaluation.task import EvaluationTask


@dataclass(slots=True)
class _ParallelSSHEvaluationRecord:
    task: EvaluationTask
    context: _SSHExecutionContext
    workspace_key: str
    state: EvaluationState = EvaluationState.PENDING
    future: Future[list[Artifact]] | None = None
    artifacts: list[Artifact] | None = None
    exception: BaseException | None = None
    error: str | None = None
    capacity_released: bool = False
    cancel_requested: bool = False


class ParallelSSHBackend(_SSHExecutionBackend):
    """Run SSH-backed evaluation tasks concurrently in a bounded thread pool."""

    def __init__(
        self,
        host: str,
        *,
        max_workers: int,
        max_pending: int | None = None,
        remote_base_work_dir: str | PurePosixPath,
        username: str | None = None,
        local_staging_dir: str | Path | None = None,
        port: int | None = None,
        identity_file: str | Path | None = None,
        ssh_options: Sequence[str] = (),
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        transfer_timeout: float | None = 60.0,
        ssh_executable: str = "ssh",
        rsync_executable: str = "rsync",
    ) -> None:
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers <= 0
        ):
            raise ValueError("max_workers must be positive.")
        if max_pending is not None and (
            isinstance(max_pending, bool)
            or not isinstance(max_pending, int)
            or max_pending <= 0
        ):
            raise ValueError("max_pending must be positive when provided.")

        super().__init__(
            host,
            remote_base_work_dir=remote_base_work_dir,
            username=username,
            local_staging_dir=local_staging_dir,
            port=port,
            identity_file=identity_file,
            ssh_options=ssh_options,
            run_id=run_id,
            metadata=metadata,
            transfer_timeout=transfer_timeout,
            ssh_executable=ssh_executable,
            rsync_executable=rsync_executable,
        )
        self.max_workers = max_workers
        self.max_pending = max_pending
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="genio-ssh",
        )
        self._records: dict[str, _ParallelSSHEvaluationRecord] = {}
        self._active_workspaces: set[str] = set()
        self._lock = Lock()
        self._capacity = BoundedSemaphore(max_pending) if max_pending is not None else None

    def submit(self, task: EvaluationTask) -> EvaluationHandle:
        """Submit an SSH-backed task for asynchronous thread-pool execution."""

        self._acquire_capacity()

        try:
            with self._lock:
                if self._shutdown:
                    raise BackendShutdownError("Cannot submit work after backend shutdown.")

                context = self.create_context(task)
                workspace_key = str(context._remote_path(context.task_dir(task)))
                if workspace_key in self._active_workspaces:
                    raise BackendError(
                        "Cannot run concurrent tasks in the same SSH workspace: "
                        f"{workspace_key!r}."
                    )

                handle = EvaluationHandle(
                    id=uuid4().hex,
                    task_id=task.task_id,
                    backend_id=self.__class__.__name__,
                    metadata=self._handle_metadata(task, context),
                    payload=task,
                )
                record = _ParallelSSHEvaluationRecord(
                    task=task,
                    context=context,
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
        """Wait for an SSH-backed task and return artifacts or its original exception."""

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
        """Return the current state of a parallel SSH evaluation."""

        with self._lock:
            return self._require_record(handle).state

    def error(self, handle: EvaluationHandle) -> str | None:
        """Return the captured error after a failed SSH evaluation."""

        with self._lock:
            return self._require_record(handle).error

    def cancel(self, handle: EvaluationHandle) -> bool:
        """Cancel a queued SSH evaluation or terminate its remote command."""

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
        """Stop accepting work and optionally cancel all unfinished SSH tasks."""

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

    def _acquire_capacity(self) -> None:
        if self._capacity is None:
            return
        while True:
            with self._lock:
                if self._shutdown:
                    raise BackendShutdownError(
                        "Cannot submit work after backend shutdown."
                    )
            if self._capacity.acquire(timeout=0.1):
                return

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
                    if not record.context.workspace_quarantined:
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

    def _require_record(self, handle: EvaluationHandle) -> _ParallelSSHEvaluationRecord:
        try:
            return self._records[handle.id]
        except KeyError as exc:
            raise UnknownEvaluationHandleError(
                f"Evaluation handle {handle.id!r} does not belong to this backend."
            ) from exc

    def _release_capacity(self, record: _ParallelSSHEvaluationRecord) -> None:
        if self._capacity is not None and not record.capacity_released:
            record.capacity_released = True
            self._capacity.release()

    def _mark_cancelled(self, record: _ParallelSSHEvaluationRecord) -> None:
        record.state = EvaluationState.CANCELLED
        if not record.context.workspace_quarantined:
            self._active_workspaces.discard(record.workspace_key)
        self._release_capacity(record)


__all__ = ["ParallelSSHBackend"]
