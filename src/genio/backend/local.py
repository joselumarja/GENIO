from __future__ import annotations

from genio.backend.base import Backend, EvaluationHandle, EvaluationState
from genio.core.candidate import Candidate
from genio.core.result import Result
from genio.runner.base import Runner


class LocalBackend(Backend):
    """Synchronous backend that runs evaluations in the current process."""

    def __init__(self, runner: Runner) -> None:
        self._runner = runner
        self._results: dict[str, Result] = {}

    def submit(self, candidate: Candidate) -> EvaluationHandle:
        handle = EvaluationHandle(id=candidate.id)
        try:
            self._results[handle.id] = self._runner.run(candidate)
        except Exception as exc:
            self._results[handle.id] = Result.failed(candidate.id, str(exc))
        return handle

    def collect(self, handle: EvaluationHandle) -> Result:
        return self._results[handle.id]

    def status(self, handle: EvaluationHandle) -> EvaluationState:
        return EvaluationState.DONE if handle.id in self._results else EvaluationState.PENDING
