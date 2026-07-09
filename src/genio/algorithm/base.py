from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from genio.core.evaluation import Evaluation
from genio.core.individual import Individual

if TYPE_CHECKING:
    from genio.session.optimization import OptimizationSession


class SearchAlgorithm(ABC):
    """Algorithm that proposes individuals and learns from evaluations."""

    @abstractmethod
    def ask(self, session: OptimizationSession) -> Sequence[Individual]:
        raise NotImplementedError

    @abstractmethod
    def tell(self, evaluations: Sequence[Evaluation]) -> None:
        raise NotImplementedError

    @abstractmethod
    def should_stop(self) -> bool:
        raise NotImplementedError

    def best_individuals(self) -> Sequence[Individual]:
        return ()
