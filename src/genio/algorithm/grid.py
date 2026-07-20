from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from genio.algorithm.base import SearchAlgorithm
from genio.checkpoint.codec import decode_evaluation, encode_evaluation
from genio.checkpoint.errors import CheckpointFormatError
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual


class GridSearch(SearchAlgorithm):
    """Enumerates the search space in search-index order."""

    supports_checkpointing = True

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

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return immutable grid-search configuration."""

        return {
            "max_evaluations": self.max_evaluations,
            "batch_size": self.batch_size,
            "start_index": self.start_index,
        }

    def checkpoint_state(self) -> Mapping[str, Any]:
        """Return current grid cursor, budget and recorded evaluations."""

        return {
            "next_index": self._next_index,
            "asked": self._asked,
            "exhausted": self._exhausted,
            "evaluations": [encode_evaluation(item) for item in self._evaluations],
        }

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
        search_space,
    ) -> None:
        """Restore a grid-search cursor from a versioned checkpoint."""

        if version != self.checkpoint_version:
            raise CheckpointFormatError(f"Unsupported GridSearch checkpoint version {version}.")
        try:
            next_index = state["next_index"]
            asked = state["asked"]
            exhausted = state["exhausted"]
            evaluations = [
                decode_evaluation(item, search_space)
                for item in state.get("evaluations", [])
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointFormatError("Invalid GridSearch checkpoint state.") from exc
        if (
            isinstance(next_index, bool)
            or not isinstance(next_index, int)
            or isinstance(asked, bool)
            or not isinstance(asked, int)
            or next_index < self.start_index
            or asked < 0
            or not isinstance(exhausted, bool)
        ):
            raise CheckpointFormatError("Invalid GridSearch checkpoint counters.")
        if next_index - self.start_index != asked or asked != len(evaluations):
            raise CheckpointFormatError("GridSearch checkpoint history is inconsistent.")
        if self.max_evaluations is not None and asked > self.max_evaluations:
            raise CheckpointFormatError("GridSearch checkpoint exceeds its budget.")
        if exhausted != (next_index >= search_space.search_space_size):
            raise CheckpointFormatError("GridSearch exhaustion state is inconsistent.")
        self._next_index = next_index
        self._asked = asked
        self._exhausted = exhausted
        self._evaluations = evaluations


__all__ = ["GridSearch"]
