from __future__ import annotations

from collections.abc import Sequence
from random import Random

from genio.algorithm.base import SearchAlgorithm
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual


class RandomSearch(SearchAlgorithm):
    """Randomly samples individuals from the search space."""

    def __init__(
        self,
        *,
        max_evaluations: int,
        batch_size: int = 1,
        unique: bool = True,
        balanced: bool = False,
        random: Random | None = None,
    ) -> None:
        if max_evaluations < 0:
            raise ValueError("max_evaluations cannot be negative.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.max_evaluations = max_evaluations
        self.batch_size = batch_size
        self.unique = unique
        self.balanced = balanced
        self.random = random or Random()
        self._asked = 0
        self._evaluations: list[Evaluation] = []

    def ask(self, session) -> Sequence[Individual]:
        remaining = self.max_evaluations - self._asked
        if remaining <= 0:
            return ()

        size = min(self.batch_size, remaining)
        sample_population = (
            session.search_space.sample_balanced_population
            if self.balanced
            else session.search_space.sample_population
        )
        individuals = sample_population(
            size,
            unique=self.unique,
            random=self.random,
        )

        self._asked += len(individuals)
        return tuple(individuals)

    def tell(self, evaluations: Sequence[Evaluation]) -> None:
        self._evaluations.extend(evaluations)

    def should_stop(self) -> bool:
        return self._asked >= self.max_evaluations


__all__ = ["RandomSearch"]
