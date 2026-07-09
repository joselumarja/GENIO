from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from tempfile import mkdtemp
from typing import Any
from uuid import uuid4

from genio.backend.base import Backend, EvaluationHandle, EvaluationState
from genio.core.artifact import Artifact
from genio.evaluation.task import EvaluationTask, ExecutionContext


class LocalBackend(Backend):
    """Synchronous backend that runs evaluations in the current process."""

    def __init__(
        self,
        base_work_dir: str | Path | None = None,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id or uuid4().hex
        self.metadata = dict(metadata or {})
        self.base_work_dir = self._resolve_base_work_dir(base_work_dir)
        self._artifacts: dict[str, list[Artifact]] = {}
        self._states: dict[str, EvaluationState] = {}
        self._errors: dict[str, str] = {}

    def submit(self, task: EvaluationTask) -> EvaluationHandle:
        handle = EvaluationHandle(
            id=task.task_id,
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
            self._errors[handle.id] = str(exc)
            self._states[handle.id] = EvaluationState.FAILED
            raise
        return handle

    def collect(self, handle: EvaluationHandle) -> list[Artifact]:
        return self._artifacts[handle.id]

    def status(self, handle: EvaluationHandle) -> EvaluationState:
        return self._states.get(handle.id, EvaluationState.PENDING)

    def error(self, handle: EvaluationHandle) -> str | None:
        return self._errors.get(handle.id)

    def cancel(self, handle: EvaluationHandle) -> None:
        if self.status(handle) in {EvaluationState.DONE, EvaluationState.FAILED}:
            return
        self._states[handle.id] = EvaluationState.CANCELLED

    def create_context(self, task: EvaluationTask) -> ExecutionContext:
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
