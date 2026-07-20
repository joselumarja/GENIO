from __future__ import annotations

from collections.abc import Mapping, Sequence
from random import Random
from typing import Any

from genio.algorithm.base import SearchAlgorithm
from genio.checkpoint.codec import (
    decode_evaluation,
    decode_random_state,
    encode_evaluation,
    encode_random_state,
)
from genio.checkpoint.errors import CheckpointFormatError
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual


class RandomSearch(SearchAlgorithm):
    """Randomly samples individuals from the search space."""

    supports_checkpointing = True

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
        """Sample the next batch within the remaining evaluation budget."""
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
        """Record completed evaluations."""
        self._evaluations.extend(evaluations)

    def should_stop(self) -> bool:
        """Return whether the evaluation budget is exhausted."""
        return self._asked >= self.max_evaluations

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return immutable random-search configuration."""

        return {
            "max_evaluations": self.max_evaluations,
            "batch_size": self.batch_size,
            "unique": self.unique,
            "balanced": self.balanced,
        }

    def checkpoint_state(self) -> Mapping[str, Any]:
        """Return budget, RNG and recorded random-search evaluations."""

        return {
            "asked": self._asked,
            "random_state": encode_random_state(self.random.getstate()),
            "evaluations": [encode_evaluation(item) for item in self._evaluations],
        }

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
        search_space,
    ) -> None:
        """Restore random budget and deterministic RNG continuation."""

        if version != self.checkpoint_version:
            raise CheckpointFormatError(
                f"Unsupported RandomSearch checkpoint version {version}."
            )
        try:
            asked = state["asked"]
            random_state = decode_random_state(state["random_state"])
            evaluations = [
                decode_evaluation(item, search_space)
                for item in state.get("evaluations", [])
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointFormatError("Invalid RandomSearch checkpoint state.") from exc
        if (
            isinstance(asked, bool)
            or not isinstance(asked, int)
            or asked < 0
            or asked > self.max_evaluations
        ):
            raise CheckpointFormatError("Invalid RandomSearch asked counter.")
        if asked != len(evaluations):
            raise CheckpointFormatError("RandomSearch checkpoint history is inconsistent.")
        self.random.setstate(random_state)
        self._asked = asked
        self._evaluations = evaluations


__all__ = ["RandomSearch"]
