from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genio.core.candidate import Candidate
    from genio.core.result import Result


class EvaluationState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EvaluationHandle:
    id: str
    payload: Any = None


class Backend(ABC):
    """Execution mechanism for candidate evaluations."""

    @abstractmethod
    def submit(self, candidate: Candidate) -> EvaluationHandle:
        raise NotImplementedError

    def submit_batch(self, candidates: Sequence[Candidate]) -> list[EvaluationHandle]:
        return [self.submit(candidate) for candidate in candidates]

    @abstractmethod
    def collect(self, handle: EvaluationHandle) -> Result:
        raise NotImplementedError

    @abstractmethod
    def status(self, handle: EvaluationHandle) -> EvaluationState:
        raise NotImplementedError
