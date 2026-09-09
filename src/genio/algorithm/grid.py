"""Deterministic enumeration of a finite GENIO search space."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

from genio.algorithm.base import SearchAlgorithm
from genio.checkpoint.errors import CheckpointFormatError
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual

if TYPE_CHECKING:
    from genio.objective.runtime import EvaluatedBatch


class GridSearch(SearchAlgorithm):
    """Enumerate individuals in ascending mixed-radix search-index order.

    Enumeration starts at ``start_index`` and advances when individuals are
    proposed by ``ask``. The optional evaluation budget is therefore a limit
    on proposals relative to that starting cursor, rather than an absolute
    search-space index. Evaluations supplied through ``tell`` are retained for
    checkpoint consistency but do not affect enumeration.
    """

    supports_checkpointing = True

    def __init__(
        self,
        *,
        max_evaluations: int | None = None,
        batch_size: int = 1,
        start_index: int = 0,
    ) -> None:
        """Initialize a grid-search cursor.

        Args:
            max_evaluations: Maximum number of individuals to propose, or
                ``None`` to continue until the finite space is exhausted.
            batch_size: Maximum number of consecutive indexes returned by one
                ``ask`` call.
            start_index: First search-space index to propose. Bounds against a
                particular search space are checked when ``ask`` runs.

        Raises:
            ValueError: If the optional budget is negative, ``batch_size`` is
                not positive, or ``start_index`` is negative.
        """
        if max_evaluations is not None and (
            isinstance(max_evaluations, bool)
            or not isinstance(max_evaluations, int)
        ):
            raise ValueError("max_evaluations must be an integer or None.")
        if max_evaluations is not None and max_evaluations < 0:
            raise ValueError("max_evaluations cannot be negative.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if isinstance(start_index, bool) or not isinstance(start_index, int):
            raise ValueError("start_index must be an integer.")
        if start_index < 0:
            raise ValueError("start_index cannot be negative.")

        self.max_evaluations = max_evaluations
        self.batch_size = batch_size
        self.start_index = start_index
        self._next_index = start_index
        self._asked = 0
        self._exhausted = False
        self._evaluations: list[Evaluation] = []

    def ask(self) -> Sequence[Individual]:
        """Return the next consecutive search-index range.

        Returns:
            A tuple ordered by ascending search index. Its size is limited by
            ``batch_size``, the remaining budget, and the remaining space. An
            empty tuple is returned after either limit is reached or when the
            starting cursor lies beyond the space.
        """
        if self._exhausted:
            return ()
        search_space = self.context.search_space
        if self._next_index >= search_space.search_space_size:
            self._exhausted = True
            return ()
        if self.max_evaluations is not None and self._asked >= self.max_evaluations:
            return ()

        remaining_space = search_space.search_space_size - self._next_index
        remaining_budget = self.batch_size
        if self.max_evaluations is not None:
            remaining_budget = min(remaining_budget, self.max_evaluations - self._asked)

        size = min(self.batch_size, remaining_space, remaining_budget)
        individuals = tuple(
            search_space.from_index(search_index)
            for search_index in range(self._next_index, self._next_index + size)
        )

        self._next_index += size
        self._asked += size
        if self._next_index >= search_space.search_space_size:
            self._exhausted = True
        return individuals

    def tell(self, batch: EvaluatedBatch) -> None:
        """Append completed evaluations to checkpointed history.

        The batch evaluations are stored in their existing order; they do not
        change the grid cursor and are not matched to the latest proposal here.

        Args:
            batch: Objective-aware batch whose evaluations are recorded.
        """
        self._evaluations.extend(tuple(batch.evaluations))

    def should_stop(self) -> bool:
        """Report whether space exhaustion is known or the budget is spent.

        Returns:
            ``True`` once ``ask`` has reached the end of the search space, or
            once the configured proposal budget has been reached.

        Note:
            A ``start_index`` beyond the space is recognized as exhausted by
            the first ``ask`` call because the constructor has no search-space
            instance against which to check it.
        """
        if self._exhausted:
            return True
        return self.max_evaluations is not None and self._asked >= self.max_evaluations

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return configuration that must match when restoring state.

        Returns:
            The optional proposal budget, batch size, and starting index.
        """

        return {
            "max_evaluations": self.max_evaluations,
            "batch_size": self.batch_size,
            "start_index": self.start_index,
        }

    def checkpoint_state(self) -> Mapping[str, Any]:
        """Serialize the cursor, proposal count, and exhaustion state.

        Returns:
            JSON-compatible state sufficient to continue grid traversal.
        """

        return {
            "next_index": self._next_index,
            "asked": self._asked,
            "exhausted": self._exhausted,
        }

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        search_space,
        evaluated_batches: Sequence[EvaluatedBatch] = (),
    ) -> None:
        """Restore a grid-search cursor using session-owned history.

        Args:
            state: Encoded cursor, counters, and exhaustion flag.
            search_space: Search space used to validate the exhaustion flag.
            evaluated_batches: Authoritative evaluation history from the session.

        Raises:
            CheckpointFormatError: If values are invalid, the
                cursor disagrees with the proposal/history counts, the budget
                is exceeded, or exhaustion disagrees with the search-space
                size.
        """

        if set(state) != {"next_index", "asked", "exhausted"}:
            raise CheckpointFormatError("Invalid GridSearch checkpoint fields.")
        try:
            next_index = state["next_index"]
            asked = state["asked"]
            exhausted = state["exhausted"]
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointFormatError("Invalid GridSearch checkpoint state.") from exc
        evaluations = [
            evaluation
            for batch in evaluated_batches
            for evaluation in batch.evaluations
        ]
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
