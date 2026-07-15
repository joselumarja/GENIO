from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any
from uuid import uuid4

from genio.algorithm.base import SearchAlgorithm
from genio.backend.base import Backend
from genio.cache import ArtifactCache
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual
from genio.core.proposal import Proposal
from genio.core.search_result import SearchResult
from genio.evaluation.executor import EvaluationExecutor
from genio.evaluation.workflow import EvaluationWorkflow
from genio.search_space.space import SearchSpace
from genio.statistics.base import InMemoryStatistics, StatisticsCollector


class OptimizationSession:
    """Coordinates search space, algorithm, backend execution and statistics."""

    def __init__(
        self,
        search_space: SearchSpace,
        algorithm: SearchAlgorithm,
        backend: Backend,
        evaluation_workflow: EvaluationWorkflow,
        statistics: StatisticsCollector | None = None,
        id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        artifact_cache: ArtifactCache | None = None,
    ) -> None:
        self.id = id or search_space.scenario_id
        self.run_id = run_id or uuid4().hex
        self.search_space = search_space
        self.algorithm = algorithm
        self.backend = backend
        self.evaluation_workflow = evaluation_workflow
        self.artifact_cache = artifact_cache
        self.evaluation_executor = EvaluationExecutor(
            self.evaluation_workflow,
            backend,
            artifact_cache=artifact_cache,
        )
        self.statistics = statistics or InMemoryStatistics()
        self.metadata = metadata or {}
        self._next_proposal_sequence = 0

    def run(self) -> SearchResult:
        """Run the optimization loop and return its search result."""
        evaluations: list[Evaluation] = []
        batch_index = 0
        if self.artifact_cache is not None:
            self.artifact_cache.clear()
        self.statistics.on_session_started(self)

        while not self.algorithm.should_stop():
            individuals = tuple(self.algorithm.ask(self))
            if not individuals:
                break

            self.statistics.on_batch_started(batch_index, individuals)
            batch_evaluations = self.evaluate(individuals, batch_index=batch_index)
            evaluations.extend(batch_evaluations)

            for evaluation in batch_evaluations:
                self.statistics.on_evaluation_completed(evaluation)
            self.algorithm.tell(batch_evaluations)
            self.statistics.on_batch_completed(batch_index, batch_evaluations)
            batch_index += 1

        result = SearchResult(
            session_id=self.id,
            evaluations=tuple(evaluations),
            run_id=self.run_id,
            best_individuals=tuple(self.algorithm.best_individuals()),
            statistics=self.statistics.snapshot(),
        )
        self.statistics.on_session_completed(result)
        return replace(result, statistics=self.statistics.snapshot())

    def evaluate(
        self,
        individuals: Sequence[Individual],
        *,
        batch_index: int | None = None,
    ) -> list[Evaluation]:
        """Evaluate a batch of individuals and wrap their results."""
        proposals = self._create_proposals(individuals, batch_index=batch_index)
        self.statistics.on_proposals_generated(proposals)
        results = self.evaluation_executor.evaluate_many(
            tuple(proposal.individual for proposal in proposals)
        )
        return [
            Evaluation(
                individual=proposal.individual,
                result=result,
                metadata=proposal.evaluation_metadata(),
            )
            for proposal, result in zip(proposals, results, strict=True)
        ]

    def _create_proposals(
        self,
        individuals: Sequence[Individual],
        *,
        batch_index: int | None,
    ) -> tuple[Proposal, ...]:
        proposals = tuple(
            Proposal(
                proposal_id=f"{self.run_id}:{self._next_proposal_sequence + position:06d}",
                proposal_sequence=self._next_proposal_sequence + position,
                batch_index=batch_index,
                batch_position=position,
                individual=individual,
            )
            for position, individual in enumerate(individuals)
        )
        self._next_proposal_sequence += len(proposals)
        return proposals
