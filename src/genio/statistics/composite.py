"""Composition of independent statistics collectors behind one hook object."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from genio.checkpoint.codec import qualified_name
from genio.checkpoint.errors import CheckpointNotSupportedError
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual
from genio.core.proposal import Proposal
from genio.core.search_result import SearchResult
from genio.objective.runtime import EvaluatedBatch
from genio.statistics.base import StatisticsCollector

if TYPE_CHECKING:
    from genio.session.optimization import OptimizationSession


class CompositeStatisticsCollector(StatisticsCollector):
    """Forward lifecycle events to an ordered collection of collectors.

    Collectors are invoked in declaration order. Event delivery is fail-fast: if
    one collector raises, later collectors do not receive that event and the
    original exception propagates to the optimization session.

    Args:
        collectors: Non-empty ordered sequence of distinct collector instances.

    Raises:
        TypeError: If ``collectors`` is not a sequence of statistics collectors.
        ValueError: If the sequence is empty or repeats the same instance.
    """

    def __init__(self, collectors: Sequence[StatisticsCollector]) -> None:
        if isinstance(collectors, (str, bytes)) or not isinstance(
            collectors, Sequence
        ):
            raise TypeError("collectors must be a sequence of StatisticsCollector.")
        normalized = tuple(collectors)
        if not normalized:
            raise ValueError("collectors must not be empty.")
        if any(not isinstance(collector, StatisticsCollector) for collector in normalized):
            raise TypeError("collectors must contain only StatisticsCollector instances.")
        if len({id(collector) for collector in normalized}) != len(normalized):
            raise ValueError("collectors must not repeat the same instance.")
        self.collectors = normalized
        self.supports_checkpointing = all(
            collector.supports_checkpointing for collector in self.collectors
        )

    def on_session_started(self, session: OptimizationSession) -> None:
        """Forward session start in collector order."""

        for collector in self.collectors:
            collector.on_session_started(session)

    def on_batch_started(
        self,
        batch_index: int,
        individuals: Sequence[Individual],
    ) -> None:
        """Forward batch start in collector order."""

        for collector in self.collectors:
            collector.on_batch_started(batch_index, individuals)

    def on_proposals_generated(self, proposals: Sequence[Proposal]) -> None:
        """Forward generated proposals in collector order."""

        for collector in self.collectors:
            collector.on_proposals_generated(proposals)

    def on_evaluation_completed(self, evaluation: Evaluation) -> None:
        """Forward one completed evaluation in collector order."""

        for collector in self.collectors:
            collector.on_evaluation_completed(evaluation)

    def on_evaluated_batch(self, batch: EvaluatedBatch) -> None:
        """Forward objective-aware batch data in collector order."""

        for collector in self.collectors:
            collector.on_evaluated_batch(batch)

    def on_batch_completed(
        self,
        batch_index: int,
        evaluations: Sequence[Evaluation],
    ) -> None:
        """Forward batch completion in collector order."""

        for collector in self.collectors:
            collector.on_batch_completed(batch_index, evaluations)

    def on_session_completed(self, result: SearchResult) -> None:
        """Forward session completion in collector order."""

        for collector in self.collectors:
            collector.on_session_completed(result)

    def snapshot(self) -> dict[str, Any]:
        """Return collision-free child snapshots in declaration order."""

        return {
            "collectors": [
                {
                    "type": qualified_name(collector),
                    "snapshot": collector.snapshot(),
                }
                for collector in self.collectors
            ]
        }

    def checkpoint_signature(self) -> dict[str, Any]:
        """Return ordered child configuration signatures."""

        return {
            "type": qualified_name(self),
            "collectors": [
                collector.checkpoint_signature() for collector in self.collectors
            ],
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Return ordered child states when every collector supports checkpointing."""

        self._require_checkpoint_support()
        return {
            "collectors": [
                collector.checkpoint_state() for collector in self.collectors
            ]
        }

    def restore_checkpoint_state(
        self,
        state: dict[str, Any],
        *,
        session: OptimizationSession,
        evaluations: Sequence[Evaluation],
        evaluated_batches: Sequence[EvaluatedBatch],
        completed: bool,
    ) -> None:
        """Restore each child from its ordered state entry."""

        self._require_checkpoint_support()
        if set(state) != {"collectors"}:
            raise ValueError(
                "Composite statistics state must contain exactly 'collectors'."
            )
        child_states = state["collectors"]
        if isinstance(child_states, (str, bytes)) or not isinstance(
            child_states, Sequence
        ):
            raise TypeError("Composite collector states must be a sequence.")
        if len(child_states) != len(self.collectors):
            raise ValueError("Composite collector state count is inconsistent.")
        for index, (collector, child_state) in enumerate(
            zip(self.collectors, child_states, strict=True)
        ):
            if not isinstance(child_state, Mapping):
                raise TypeError(
                    f"Composite collector state {index} must be a mapping."
                )
            collector.restore_checkpoint_state(
                dict(child_state),
                session=session,
                evaluations=evaluations,
                evaluated_batches=evaluated_batches,
                completed=completed,
            )

    def _require_checkpoint_support(self) -> None:
        unsupported = [
            qualified_name(collector)
            for collector in self.collectors
            if not collector.supports_checkpointing
        ]
        if unsupported:
            raise CheckpointNotSupportedError(
                "Composite statistics checkpointing is not supported by: "
                f"{unsupported!r}."
            )


__all__ = ["CompositeStatisticsCollector"]
