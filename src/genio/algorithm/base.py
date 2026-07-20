from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

from genio.checkpoint.errors import CheckpointNotSupportedError
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual

if TYPE_CHECKING:
    from genio.search_space.space import SearchSpace
    from genio.session.optimization import OptimizationSession


class SearchAlgorithm(ABC):
    """Algorithm that proposes individuals and learns from evaluations."""

    checkpoint_version = 1
    supports_checkpointing = False

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

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return immutable configuration used to validate checkpoint compatibility."""

        raise CheckpointNotSupportedError(
            f"{type(self).__name__} does not support checkpointing."
        )

    def checkpoint_state(self) -> Mapping[str, Any]:
        """Return JSON-compatible mutable algorithm state at a safe boundary."""

        raise CheckpointNotSupportedError(
            f"{type(self).__name__} does not support checkpointing."
        )

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        version: int,
        search_space: SearchSpace,
    ) -> None:
        """Restore mutable algorithm state after compatibility validation."""

        raise CheckpointNotSupportedError(
            f"{type(self).__name__} does not support checkpointing."
        )
