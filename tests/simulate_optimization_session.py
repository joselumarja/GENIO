from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Mapping

from genio import EvaluationHandle
from genio import EvaluationState
from genio import EvaluationStep
from genio import EvaluationTask
from genio import EvaluationWorkflow
from genio import LocalBackend
from genio import OptimizationSession
from genio import RandomSearch
from genio import SearchSpace
from genio.core import Individual
from genio.artifacts import Artifact


ROOT = Path(__file__).resolve().parents[1]


class DummyTask(EvaluationTask):
    def run(self, context):
        return []


class DummyStep(EvaluationStep):
    id = "dummy_step"
    task_type = DummyTask

    def create_task(
        self,
        individual: Individual,
        artifacts: Mapping[str, Artifact],
    ) -> EvaluationTask:
        return DummyTask(individual=individual, step_id=self.id)


class DummyBackend(LocalBackend):
    pass


def build_session() -> OptimizationSession:
    search_space = SearchSpace(
        ROOT / "search_space/tests/simple_threshold_pipeline.json",
        ROOT / "search_space/stages/definitions",
    )

    algorithm = RandomSearch(
        max_evaluations=5,
        batch_size=2,
        unique=False,
        balanced=True,
        random=Random(0),
    )

    backend = DummyBackend(ROOT / "tmp/simulated_optimization_session")
    workflow = EvaluationWorkflow(steps=(DummyStep(),))

    return OptimizationSession(
        search_space=search_space,
        algorithm=algorithm,
        backend=backend,
        evaluation_workflow=workflow,
        id="simulated_optimization_session",
        metadata={"purpose": "object aggregation example"},
    )


def main() -> None:
    session = build_session()

    print("OptimizationSession")
    print(f"  id: {session.id}")
    print(f"  search_space: {session.search_space.scenario_id}")
    print(f"  algorithm: {type(session.algorithm).__name__}")
    print(f"  backend: {type(session.backend).__name__}")
    print(f"  workflow_steps: {[step.id for step in session.evaluation_workflow.steps]}")

    result = session.run()

    print("SearchResult")
    print(f"  session_id: {result.session_id}")
    print(f"  evaluations: {len(result.evaluations)}")
    print(f"  best_individuals: {len(result.best_individuals)}")


if __name__ == "__main__":
    main()
