from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any

from genio.checkpoint.errors import CheckpointNotSupportedError
from genio.checkpoint.codec import qualified_name
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual
from genio.core.proposal import Proposal
from genio.core.search_result import SearchResult

if TYPE_CHECKING:
    from genio.session.optimization import OptimizationSession


class StatisticsCollector(ABC):
    """Hook object notified during an optimization session."""

    checkpoint_version = 1
    supports_checkpointing = False

    def on_session_started(self, session: OptimizationSession) -> None:
        """Handle the start of an optimization session."""
        pass

    def on_batch_started(
        self,
        batch_index: int,
        individuals: Sequence[Individual],
    ) -> None:
        """Handle the start of an evaluation batch."""
        pass

    def on_proposals_generated(self, proposals: Sequence[Proposal]) -> None:
        """Handle individuals proposed for evaluation in one batch."""

        pass

    def on_evaluation_completed(self, evaluation: Evaluation) -> None:
        """Handle the completion of an individual evaluation."""
        pass

    def on_batch_completed(
        self,
        batch_index: int,
        evaluations: Sequence[Evaluation],
    ) -> None:
        """Handle the completion of an evaluation batch."""
        pass

    def on_session_completed(self, result: SearchResult) -> None:
        """Handle the completion of an optimization session."""
        pass

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of the collected statistics."""
        return {}

    def checkpoint_state(self) -> dict[str, Any]:
        """Return JSON-compatible collector state at a completed batch boundary."""

        raise CheckpointNotSupportedError(
            f"{type(self).__name__} does not support checkpointing."
        )

    def checkpoint_signature(self) -> dict[str, Any]:
        """Return immutable collector configuration for compatibility checks."""

        return {"type": qualified_name(self)}

    def restore_checkpoint_state(
        self,
        state: dict[str, Any],
        *,
        session: OptimizationSession,
        evaluations: Sequence[Evaluation],
        completed: bool,
    ) -> None:
        """Restore collector state before a resumed session continues."""

        raise CheckpointNotSupportedError(
            f"{type(self).__name__} does not support checkpointing."
        )


class InMemoryStatistics(StatisticsCollector):
    """Simple statistics collector useful as the default implementation."""

    supports_checkpointing = True

    def __init__(self) -> None:
        self.evaluations: list[Evaluation] = []
        self.batches: list[tuple[int, tuple[Evaluation, ...]]] = []

    def on_evaluation_completed(self, evaluation: Evaluation) -> None:
        """Record a completed individual evaluation."""
        self.evaluations.append(evaluation)

    def on_batch_completed(
        self,
        batch_index: int,
        evaluations: Sequence[Evaluation],
    ) -> None:
        """Record a completed evaluation batch."""
        self.batches.append((batch_index, tuple(evaluations)))

    def snapshot(self) -> dict[str, Any]:
        """Return counts of recorded evaluations and batches."""
        return {
            "evaluations": len(self.evaluations),
            "batches": len(self.batches),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Return the completed batch indexes represented by this collector."""

        return {"batch_indexes": [batch_index for batch_index, _ in self.batches]}

    def restore_checkpoint_state(
        self,
        state: dict[str, Any],
        *,
        session: OptimizationSession,
        evaluations: Sequence[Evaluation],
        completed: bool,
    ) -> None:
        """Rebuild in-memory statistics from committed session evaluations."""

        self.evaluations = list(evaluations)
        by_batch: dict[int, list[Evaluation]] = {}
        for evaluation in evaluations:
            batch_index = evaluation.metadata.get("batch_index")
            if not isinstance(batch_index, int):
                raise ValueError("Checkpoint evaluation has no integer batch_index.")
            by_batch.setdefault(batch_index, []).append(evaluation)
        expected_indexes = list(state.get("batch_indexes", []))
        if expected_indexes != sorted(by_batch):
            raise ValueError("Checkpoint statistics batch indexes are inconsistent.")
        self.batches = [
            (batch_index, tuple(by_batch[batch_index]))
            for batch_index in expected_indexes
        ]
