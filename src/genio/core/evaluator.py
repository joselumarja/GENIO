from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from genio.core.candidate import Candidate
from genio.core.result import Result

if TYPE_CHECKING:
    from genio.backend.base import Backend


class Evaluator:
    """Public orchestration API used by optimizers and heuristics."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    def evaluate(self, candidates: Sequence[Candidate]) -> list[Result]:
        handles = self._backend.submit_batch(candidates)
        return [self._backend.collect(handle) for handle in handles]
