from __future__ import annotations

from collections.abc import Sequence

from genio.algorithm.base import SearchAlgorithm
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual


class GridSearch(SearchAlgorithm):
    """Enumerates the search space in search-index order."""

    def __init__(
        self,
        *,
        max_evaluations: int | None = None,
        batch_size: int = 1,
        start_index: int = 0,
    ) -> None:
        if max_evaluations is not None and max_evaluations < 0:
            raise ValueError("max_evaluations cannot be negative.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if start_index < 0:
            raise ValueError("start_index cannot be negative.")

        self.max_evaluations = max_evaluations
        self.batch_size = batch_size
        self.start_index = start_index
        self._next_index = start_index
        self._asked = 0
        self._exhausted = False
        self._evaluations: list[Evaluation] = []

    def ask(self, session) -> Sequence[Individual]:
        """Return the next batch of individuals in search-index order."""
        if self._exhausted:
            return ()
        if self._next_index >= session.search_space.search_space_size:
            self._exhausted = True
            return ()
        if self.max_evaluations is not None and self._asked >= self.max_evaluations:
            return ()

        remaining_space = session.search_space.search_space_size - self._next_index
        remaining_budget = self.batch_size
        if self.max_evaluations is not None:
            remaining_budget = min(remaining_budget, self.max_evaluations - self._asked)

        size = min(self.batch_size, remaining_space, remaining_budget)
        individuals = tuple(
            session.search_space.from_index(search_index)
            for search_index in range(self._next_index, self._next_index + size)
        )

        self._next_index += size
        self._asked += size
        if self._next_index >= session.search_space.search_space_size:
            self._exhausted = True
        return individuals

    def tell(self, evaluations: Sequence[Evaluation]) -> None:
        """Record completed evaluations."""
        self._evaluations.extend(evaluations)

    def should_stop(self) -> bool:
        """Return whether the space or evaluation budget is exhausted."""
        if self._exhausted:
            return True
        return self.max_evaluations is not None and self._asked >= self.max_evaluations


__all__ = ["GridSearch"]
