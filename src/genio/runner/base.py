from __future__ import annotations

from abc import ABC, abstractmethod

from genio.core.candidate import Candidate
from genio.core.result import Result


class Runner(ABC):
    """Evaluation logic executed for one candidate."""

    @abstractmethod
    def run(self, candidate: Candidate) -> Result:
        raise NotImplementedError
