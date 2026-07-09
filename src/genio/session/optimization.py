from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from genio.algorithm.base import SearchAlgorithm
from genio.backend.base import Backend
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.id = id or search_space.scenario_id
        self.search_space = search_space
        self.algorithm = algorithm
        self.backend = backend
        self.evaluation_workflow = evaluation_workflow
        self.evaluation_executor = EvaluationExecutor(self.evaluation_workflow, backend)
        self.statistics = statistics or InMemoryStatistics()
        self.metadata = metadata or {}

    def run(self) -> SearchResult:
        evaluations: list[Evaluation] = []
        batch_index = 0
        self.statistics.on_session_started(self)

        while not self.algorithm.should_stop():
            individuals = tuple(self.algorithm.ask(self))
            if not individuals:
                break

            self.statistics.on_batch_started(batch_index, individuals)
            batch_evaluations = self.evaluate(individuals, batch_index=batch_index)
            evaluations.extend(batch_evaluations)
            self.algorithm.tell(batch_evaluations)

            for evaluation in batch_evaluations:
                self.statistics.on_evaluation_completed(evaluation)
            self.statistics.on_batch_completed(batch_index, batch_evaluations)
            batch_index += 1

        result = SearchResult(
            session_id=self.id,
            evaluations=tuple(evaluations),
            best_individuals=tuple(self.algorithm.best_individuals()),
            statistics=self.statistics.snapshot(),
        )
        self.statistics.on_session_completed(result)
        return result

    def evaluate(
        self,
        individuals: Sequence[Individual],
        *,
        batch_index: int | None = None,
    ) -> list[Evaluation]:
        results = self.evaluation_executor.evaluate_many(individuals)
        return [
            Evaluation(
                individual=individual,
                result=result,
                metadata={"batch_index": batch_index} if batch_index is not None else {},
            )
            for individual, result in zip(individuals, results, strict=True)
        ]
