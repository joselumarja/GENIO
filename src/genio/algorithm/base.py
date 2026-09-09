"""Common ask/tell interface for search algorithms.

An optimization session repeatedly asks an algorithm for a batch of
individuals, evaluates that batch externally, and tells the algorithm about
the resulting evaluations. Concrete algorithms own their search state and may
add stricter ordering requirements to this basic protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from genio.checkpoint.errors import CheckpointNotSupportedError
from genio.core.individual import Individual

if TYPE_CHECKING:
    from genio.objective.base import ObjectiveSchema
    from genio.objective.runtime import EvaluatedBatch
    from genio.search_space.space import SearchSpace


@dataclass(frozen=True, slots=True)
class SearchContext:
    """Immutable configuration shared with a search algorithm.

    Attributes:
        search_space: Space from which the algorithm proposes individuals.
        objective_schema: Immutable names, directions, and normalization bounds.
        has_normalizer: Whether normalized objective views are available.
        has_scalarizer: Whether aggregate objective scores are available.
        normalization_scope: Configured normalization scope, when applicable.
    """

    search_space: SearchSpace
    objective_schema: ObjectiveSchema | None = None
    has_normalizer: bool = False
    has_scalarizer: bool = False
    normalization_scope: str | None = None


class SearchAlgorithm(ABC):
    """Propose individuals and consume their completed evaluations.

    The normal lifecycle is ``configure(context)``, ``should_stop()``, ``ask()``,
    external evaluation of every returned individual, and ``tell(batch)``. The evaluated
    batch supplied to ``tell`` should correspond to the latest proposal. An
    empty proposal indicates that the algorithm has nothing to evaluate even
    if a preceding ``should_stop`` call returned ``False``.

    Checkpoint-capable implementations expose immutable configuration through
    ``checkpoint_signature`` and mutable state through ``checkpoint_state``.
    Callers restore that state only after validating the signature and the
    configured component signatures.

    Attributes:
        supports_checkpointing: Whether the checkpoint methods are
            implemented by the concrete algorithm.
    """

    supports_checkpointing = False
    _context: SearchContext

    def configure(self, context: SearchContext) -> None:
        """Attach the immutable search context to this algorithm.

        Reusing the exact same context is idempotent. Replacing it would mix
        algorithm state from different searches and is therefore rejected.

        Args:
            context: Search configuration owned by the active session.

        Raises:
            RuntimeError: If the algorithm was already configured with a
                different context object.
        """

        if hasattr(self, "_context") and self._context is not context:
            raise RuntimeError(
                f"{type(self).__name__} is already configured with a different context."
            )
        self._context = context

    @property
    def context(self) -> SearchContext:
        """Return the configured search context.

        Raises:
            RuntimeError: If ``configure`` has not been called.
        """

        if not hasattr(self, "_context"):
            raise RuntimeError(f"{type(self).__name__} has not been configured.")
        return self._context

    @abstractmethod
    def ask(self) -> Sequence[Individual]:
        """Propose the next batch of individuals for external evaluation.

        Args:
        Returns:
            The next batch in algorithm-defined order, or an empty sequence
            when no proposal is available.

        Raises:
            NotImplementedError: Always raised by the base implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def tell(self, batch: EvaluatedBatch) -> None:
        """Commit the evaluated form of the latest proposed batch.

        Args:
            batch: Objective-aware results for the latest batch returned by
                ``ask``.

        Raises:
            NotImplementedError: Always raised by the base implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def should_stop(self) -> bool:
        """Report whether the algorithm has reached its stopping condition.

        Returns:
            ``True`` when the session should request no further batches.

        Raises:
            NotImplementedError: Always raised by the base implementation.
        """
        raise NotImplementedError

    def best_individuals(self) -> Sequence[Individual]:
        """Return the best individuals identified by the algorithm.

        Returns:
            An algorithm-defined ordered collection. Algorithms that do not
            track a best set return an empty sequence.
        """
        return ()

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return immutable configuration for compatibility validation.

        Returns:
            A mapping that identifies configuration affecting continuation.

        Raises:
            CheckpointNotSupportedError: Always raised unless a subclass
                implements checkpointing.
        """

        raise CheckpointNotSupportedError(
            f"{type(self).__name__} does not support checkpointing."
        )

    def checkpoint_state(self) -> Mapping[str, Any]:
        """Return JSON-compatible mutable state at a checkpoint-safe boundary.

        Returns:
            State sufficient for the concrete algorithm to continue.

        Raises:
            CheckpointNotSupportedError: Always raised unless a subclass
                implements checkpointing.
        """

        raise CheckpointNotSupportedError(
            f"{type(self).__name__} does not support checkpointing."
        )

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        search_space: SearchSpace,
        evaluated_batches: Sequence[EvaluatedBatch] = (),
    ) -> None:
        """Restore mutable state after external compatibility validation.

        Args:
            state: JSON-decoded mutable state produced by
                ``checkpoint_state``.
            search_space: Search space with which restored individuals and
                algorithm state must be associated.
            evaluated_batches: Objective-aware history already restored by the
                session. Algorithms may ignore it when their native state does
                not depend on objective vectors.

        Raises:
            CheckpointNotSupportedError: Always raised unless a subclass
                implements checkpointing.
        """

        raise CheckpointNotSupportedError(
            f"{type(self).__name__} does not support checkpointing."
        )


__all__ = ["SearchAlgorithm", "SearchContext"]
