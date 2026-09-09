"""Observation hooks for optimization-session lifecycle events."""

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
    from genio.objective.runtime import EvaluatedBatch
    from genio.session.optimization import OptimizationSession


class StatisticsCollector(ABC):
    """Observe optimization lifecycle events without controlling the search.

    For each normal batch, callbacks occur in this order:
    ``on_batch_started``, ``on_proposals_generated``, one
    ``on_evaluation_completed`` per result, and ``on_batch_completed``. Session
    start and completion callbacks surround those batches. Restored collectors
    receive ``restore_checkpoint_state`` instead of replaying historical hooks.

    Hook implementations should raise on unrecoverable persistence errors; a batch
    is not committed to session history until its completion hooks return.
    """

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

    def on_evaluated_batch(self, batch: EvaluatedBatch) -> None:
        """Handle validated objective values before a batch is committed."""

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
        evaluated_batches: Sequence[EvaluatedBatch],
        completed: bool,
    ) -> None:
        """Restore collector state before a resumed session continues."""

        raise CheckpointNotSupportedError(
            f"{type(self).__name__} does not support checkpointing."
        )


class InMemoryStatistics(StatisticsCollector):
    """Record completed evaluations and batches in memory.

    This is the default collector used by :class:`genio.OptimizationSession`. It
    intentionally exposes only aggregate counts through :meth:`snapshot` while
    retaining full objects on ``evaluations`` and ``batches`` for direct access.
    """

    supports_checkpointing = True

    def __init__(self) -> None:
        """Initialize empty evaluation and batch histories."""
        self.evaluations: list[Evaluation] = []
        self.batches: list[tuple[int, tuple[Evaluation, ...]]] = []
        self.evaluated_batches: list[EvaluatedBatch] = []

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

    def on_evaluated_batch(self, batch: EvaluatedBatch) -> None:
        """Record one objective-aware batch."""

        self.evaluated_batches.append(batch)

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
        evaluated_batches: Sequence[EvaluatedBatch],
        completed: bool,
    ) -> None:
        """Rebuild in-memory statistics from committed session evaluations."""

        self.evaluations = list(evaluations)
        self.evaluated_batches = list(evaluated_batches)
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
