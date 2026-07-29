from __future__ import annotations

from dataclasses import dataclass

import pytest

from genio import (
    Evaluation,
    EvaluationStep,
    EvaluationTask,
    EvaluationWorkflow,
    LocalBackend,
    MetricArtifact,
    MetricObjective,
    NSGA2Search,
    ObjectiveSet,
    OptimizationDirection,
    OptimizationSession,
    Result,
    SearchSpace,
    StageChoice,
)
from genio.artifacts import Artifact
from genio.evaluation.task import ExecutionContext
from genio.search_space import SearchScenarioSpec, SlotSpec


class DummySession:
    def __init__(self, search_space: SearchSpace) -> None:
        self.search_space = search_space


def make_search_space() -> SearchSpace:
    return SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="nsga2_test",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(
                        StageChoice(slot=0, stage="a"),
                        StageChoice(slot=0, stage="b"),
                    ),
                ),
                SlotSpec(
                    index=1,
                    alternatives=(
                        StageChoice(slot=1, stage="c"),
                        StageChoice(slot=1, stage="d"),
                    ),
                ),
            ),
            design_spaces={"hls": {"npc": (1, 2)}},
        )
    )


def objectives() -> ObjectiveSet:
    return ObjectiveSet(
        (
            MetricObjective("quality", OptimizationDirection.MAXIMIZE),
            MetricObjective("latency", OptimizationDirection.MINIMIZE),
        )
    )


def evaluation(individual, *, quality: float, latency: float) -> Evaluation:
    return Evaluation(
        individual,
        Result.success(
            individual.id,
            metrics={"quality": quality, "latency": latency},
        ),
    )


INITIAL_POPULATION = (
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 1, 1),
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"population_size": 0}, "population_size"),
        ({"max_generations": -1}, "max_generations"),
        ({"crossover_probability": 1.1}, "crossover_probability"),
        ({"mutation_probability": -0.1}, "mutation_probability"),
        ({"seed": -1}, "seed"),
    ),
)
def test_nsga2_validates_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        NSGA2Search(objectives=objectives(), **kwargs)


def test_nsga2_requires_multiple_objectives() -> None:
    with pytest.raises(ValueError, match="at least two"):
        NSGA2Search(
            objectives=MetricObjective(
                "quality",
                OptimizationDirection.MAXIMIZE,
            )
        )


def test_nsga2_rejects_non_integer_initial_genotypes() -> None:
    with pytest.raises(ValueError, match="only integers"):
        NSGA2Search(
            objectives=objectives(),
            population_size=1,
            initial_population=((0.5, 0, 0),),
        )


def test_nsga2_materializes_initial_integer_population() -> None:
    algorithm = NSGA2Search(
        objectives=objectives(),
        population_size=4,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        seed=3,
    )

    population = tuple(algorithm.ask(DummySession(make_search_space())))

    assert tuple(individual.genotype for individual in population) == INITIAL_POPULATION
    assert all(
        individual.metadata["algorithm"]["name"] == "nsga2"
        and individual.metadata["algorithm"]["generation"] == 1
        and individual.metadata["algorithm"]["proposal_origin"] == "initialization"
        for individual in population
    )


def test_nsga2_returns_successful_pareto_front() -> None:
    algorithm = NSGA2Search(
        objectives=objectives(),
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    population = tuple(algorithm.ask(DummySession(make_search_space())))
    values = (
        (1.0, 4.0),
        (2.0, 3.0),
        (1.5, 1.0),
        (0.0, 5.0),
    )

    algorithm.tell(
        tuple(
            evaluation(individual, quality=quality, latency=latency)
            for individual, (quality, latency) in zip(
                population,
                values,
                strict=True,
            )
        )
    )

    assert {individual.id for individual in algorithm.best_individuals()} == {
        population[1].id,
        population[2].id,
    }
    assert algorithm.should_stop()


def test_nsga2_treats_failed_evaluations_as_infeasible() -> None:
    algorithm = NSGA2Search(
        objectives=objectives(),
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    population = tuple(algorithm.ask(DummySession(make_search_space())))
    evaluations = [
        evaluation(individual, quality=float(index), latency=float(4 - index))
        for index, individual in enumerate(population)
    ]
    evaluations[-1] = Evaluation(
        population[-1],
        Result.failed(
            population[-1].id,
            "failed",
            metrics={"quality": 1000.0, "latency": 0.0},
        ),
    )

    algorithm.tell(evaluations)

    assert population[-1] not in algorithm.best_individuals()
    assert algorithm.best_individuals()


def test_nsga2_generates_valid_categorical_offspring() -> None:
    search_space = make_search_space()
    session = DummySession(search_space)
    algorithm = NSGA2Search(
        objectives=objectives(),
        population_size=4,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        mutation_probability=1.0,
        seed=11,
    )
    first = tuple(algorithm.ask(session))
    algorithm.tell(
        tuple(
            evaluation(
                individual,
                quality=float(index),
                latency=float(index),
            )
            for index, individual in enumerate(first)
        )
    )

    second = tuple(algorithm.ask(session))

    assert len(second) == 4
    assert all(
        search_space.to_genotype(individual) == individual.genotype
        and individual.metadata["algorithm"]["generation"] == 2
        and individual.metadata["algorithm"]["proposal_origin"] == "offspring"
        for individual in second
    )


def test_nsga2_checkpoint_replay_preserves_next_generation() -> None:
    search_space = make_search_space()
    session = DummySession(search_space)
    original = NSGA2Search(
        objectives=objectives(),
        population_size=4,
        max_generations=3,
        initial_population=INITIAL_POPULATION,
        seed=17,
    )
    first = tuple(original.ask(session))
    original.tell(
        tuple(
            evaluation(
                individual,
                quality=float(index),
                latency=float(3 - index),
            )
            for index, individual in enumerate(first)
        )
    )
    state = original.checkpoint_state()
    expected = tuple(individual.genotype for individual in original.ask(session))

    restored = NSGA2Search(
        objectives=objectives(),
        population_size=4,
        max_generations=3,
        initial_population=INITIAL_POPULATION,
        seed=17,
    )
    restored.restore_checkpoint_state(
        state,
        version=restored.checkpoint_version,
        search_space=search_space,
    )
    actual = tuple(individual.genotype for individual in restored.ask(session))

    assert actual == expected


@dataclass(frozen=True, slots=True)
class ObjectiveTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        del context
        assert self.individual.search_index is not None
        index = float(self.individual.search_index)
        return [
            MetricArtifactImpl(
                name="objectives",
                producer=self.step_id or "objectives",
                individual_id=self.individual.id,
                values={"quality": index, "latency": -index},
            )
        ]


@dataclass(frozen=True, slots=True)
class MetricArtifactImpl(MetricArtifact):
    values: dict[str, float]

    def load(self):
        return (self.values,)

    def metrics(self):
        return self.values


@dataclass(frozen=True, slots=True)
class ObjectiveStep(EvaluationStep):
    id: str = "objectives"
    task_type: type[EvaluationTask] = ObjectiveTask

    def create_task(self, individual, artifacts):
        del artifacts
        return ObjectiveTask(individual=individual, step_id=self.id)


def test_nsga2_runs_through_optimization_session(tmp_path) -> None:
    algorithm = NSGA2Search(
        objectives=ObjectiveSet(
            (
                MetricObjective(
                    "objectives.quality",
                    OptimizationDirection.MAXIMIZE,
                ),
                MetricObjective(
                    "objectives.latency",
                    OptimizationDirection.MINIMIZE,
                ),
            )
        ),
        population_size=4,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        seed=5,
    )

    result = OptimizationSession(
        search_space=make_search_space(),
        algorithm=algorithm,
        backend=LocalBackend(base_work_dir=tmp_path),
        evaluation_workflow=EvaluationWorkflow((ObjectiveStep(),)),
    ).run()

    assert len(result.evaluations) == 8
    assert result.best_individuals
