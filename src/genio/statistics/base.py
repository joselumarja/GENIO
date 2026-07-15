from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any

from genio.core.evaluation import Evaluation
from genio.core.individual import Individual
from genio.core.proposal import Proposal
from genio.core.search_result import SearchResult

if TYPE_CHECKING:
    from genio.session.optimization import OptimizationSession


class StatisticsCollector(ABC):
    """Hook object notified during an optimization session."""

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


class InMemoryStatistics(StatisticsCollector):
    """Simple statistics collector useful as the default implementation."""

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
