"""Random search implementation for finite GENIO search spaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from random import Random
from typing import Any, TYPE_CHECKING

from genio.algorithm.base import SearchAlgorithm
from genio.checkpoint.codec import (
    decode_random_state,
    encode_random_state,
)
from genio.checkpoint.errors import CheckpointFormatError
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual

if TYPE_CHECKING:
    from genio.objective.runtime import EvaluatedBatch


class RandomSearch(SearchAlgorithm):
    """Sample independent batches until a proposal budget is exhausted.

    The algorithm counts individuals when ``ask`` proposes them, not when
    ``tell`` records their evaluations. Optional uniqueness is local to one
    returned batch; indexes sampled by earlier calls are not excluded from
    later calls. Balanced sampling chooses a stage group uniformly for each
    scenario slot and then chooses an alternative within that group, while
    design genes remain uniformly sampled from their domains.

    Completed evaluations are retained for checkpoint restoration, but they do
    not influence subsequent random proposals.
    """

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
        """Initialize a budgeted random search.

        Args:
            max_evaluations: Maximum number of individuals that may be
                proposed. Zero creates an already-stopped search.
            batch_size: Maximum number of individuals returned by one
                ``ask`` call. The final batch may be smaller.
            unique: If ``True``, reject duplicate search indexes within each
                batch. This does not enforce uniqueness across batches.
            balanced: If ``True``, use stage-balanced slot sampling instead of
                uniform sampling over each slot's alternatives.
            random: Pseudo-random generator to use. A new unseeded generator
                is created when omitted.

        Raises:
            ValueError: If ``max_evaluations`` is negative or ``batch_size``
                is not positive.
        """
        if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int):
            raise ValueError("max_evaluations must be an integer.")
        if max_evaluations < 0:
            raise ValueError("max_evaluations cannot be negative.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not isinstance(unique, bool):
            raise ValueError("unique must be a boolean.")
        if not isinstance(balanced, bool):
            raise ValueError("balanced must be a boolean.")

        self.max_evaluations = max_evaluations
        self.batch_size = batch_size
        self.unique = unique
        self.balanced = balanced
        self.random = random or Random()
        self._asked = 0
        self._evaluations: list[Evaluation] = []

    def ask(self) -> Sequence[Individual]:
        """Sample the next batch within the remaining proposal budget.

        Returns:
            A tuple containing at most ``batch_size`` individuals, shortened
            to the remaining budget, or an empty tuple after exhaustion.

        Raises:
            ValueError: If unique sampling requests more individuals than are
                available in the search space for this batch.
        """
        remaining = self.max_evaluations - self._asked
        if remaining <= 0:
            return ()

        size = min(self.batch_size, remaining)
        sample_population = (
            self.context.search_space.sample_balanced_population
            if self.balanced
            else self.context.search_space.sample_population
        )
        individuals = sample_population(
            size,
            unique=self.unique,
            random=self.random,
        )

        self._asked += len(individuals)
        return tuple(individuals)

    def tell(self, batch: EvaluatedBatch) -> None:
        """Append completed evaluations to checkpointed history.

        The batch evaluations are stored in their existing order; this method
        does not validate them against the last proposal or alter sampling.

        Args:
            batch: Objective-aware batch whose evaluations are recorded.
        """
        self._evaluations.extend(tuple(batch.evaluations))

    def should_stop(self) -> bool:
        """Report whether the proposal counter reached its budget.

        Returns:
            ``True`` after ``max_evaluations`` individuals have been returned
            by ``ask``.
        """
        return self._asked >= self.max_evaluations

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return configuration that must match when restoring state.

        Returns:
            The proposal budget, batch size, uniqueness mode, and balancing
            mode.
        """

        return {
            "max_evaluations": self.max_evaluations,
            "batch_size": self.batch_size,
            "unique": self.unique,
            "balanced": self.balanced,
        }

    def checkpoint_state(self) -> Mapping[str, Any]:
        """Serialize the proposal counter and deterministic RNG state.

        Returns:
            JSON-compatible state that preserves deterministic continuation.
        """

        return {
            "asked": self._asked,
            "random_state": encode_random_state(self.random.getstate()),
        }

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        search_space,
        evaluated_batches: Sequence[EvaluatedBatch] = (),
    ) -> None:
        """Restore random-search history and deterministic RNG continuation.

        Args:
            state: Encoded proposal counter and random state.
            search_space: Search space owned by the restoring session.
            evaluated_batches: Authoritative evaluation history from the session.

        Raises:
            CheckpointFormatError: If values are invalid, the counter exceeds
                the configured budget, or history is inconsistent.
            ValueError: If the decoded object is not a valid ``Random`` state.
        """

        del search_space
        if set(state) != {"asked", "random_state"}:
            raise CheckpointFormatError("Invalid RandomSearch checkpoint fields.")
        try:
            asked = state["asked"]
            random_state = decode_random_state(state["random_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointFormatError("Invalid RandomSearch checkpoint state.") from exc
        evaluations = [
            evaluation
            for batch in evaluated_batches
            for evaluation in batch.evaluations
        ]
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
