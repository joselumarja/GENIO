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
        """Propose the next individuals for evaluation."""
        raise NotImplementedError

    @abstractmethod
    def tell(self, evaluations: Sequence[Evaluation]) -> None:
        """Update the algorithm with completed evaluations."""
        raise NotImplementedError

    @abstractmethod
    def should_stop(self) -> bool:
        """Return whether the search should stop."""
        raise NotImplementedError

    def best_individuals(self) -> Sequence[Individual]:
        """Return the best individuals identified by the algorithm."""
        return ()
