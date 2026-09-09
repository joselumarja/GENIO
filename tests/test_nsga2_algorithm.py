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
from genio.algorithm.base import SearchContext
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


def configure_algorithm(
    algorithm: NSGA2Search,
    *,
    search_space: SearchSpace | None = None,
    objective_set: ObjectiveSet | None = None,
):
    search_space = search_space or make_search_space()
    objective_set = objective_set or objectives()
    algorithm.configure(
        SearchContext(
            search_space=search_space,
            objective_schema=objective_set.schema,
            has_normalizer=objective_set.normalizer is not None,
            has_scalarizer=objective_set.scalarizer is not None,
        )
    )
    return DummySession(search_space), objective_set.bind()


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
        NSGA2Search(**kwargs)


def test_nsga2_requires_multiple_objectives() -> None:
    algorithm = NSGA2Search()
    with pytest.raises(ValueError, match="at least two"):
        configure_algorithm(
            algorithm,
            objective_set=ObjectiveSet(
                (
                    MetricObjective(
                        "quality",
                        OptimizationDirection.MAXIMIZE,
                    ),
                )
            ),
        )
    configure_algorithm(algorithm)
    assert algorithm.context.objective_schema == objectives().schema


def test_nsga2_requires_objective_schema_during_configuration() -> None:
    algorithm = NSGA2Search()
    context = SearchContext(search_space=make_search_space())

    with pytest.raises(ValueError, match="objective schema"):
        algorithm.configure(context)


def test_nsga2_rejects_non_integer_initial_genotypes() -> None:
    with pytest.raises(ValueError, match="only integers"):
        NSGA2Search(
            population_size=1,
            initial_population=((0.5, 0, 0),),
        )


def test_nsga2_materializes_initial_integer_population() -> None:
    algorithm = NSGA2Search(
        population_size=4,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        seed=3,
    )
    session, _ = configure_algorithm(algorithm)

    population = tuple(algorithm.ask())

    assert tuple(individual.genotype for individual in population) == INITIAL_POPULATION
    assert all(
        individual.metadata["algorithm"]["name"] == "nsga2"
        and individual.metadata["algorithm"]["generation"] == 1
        and individual.metadata["algorithm"]["proposal_origin"] == "initialization"
        for individual in population
    )


@pytest.mark.parametrize("balanced_initialization", (False, True))
def test_nsga2_fills_unique_random_initial_population(
    balanced_initialization,
) -> None:
    algorithm = NSGA2Search(
        population_size=8,
        max_generations=1,
        eliminate_duplicates=True,
        balanced_initialization=balanced_initialization,
        seed=1,
    )
    session, _ = configure_algorithm(algorithm)

    population = tuple(algorithm.ask())

    assert len(population) == 8
    assert len({individual.genotype for individual in population}) == 8


def test_nsga2_allows_population_larger_than_space_with_duplicates() -> None:
    algorithm = NSGA2Search(
        population_size=9,
        max_generations=1,
        eliminate_duplicates=False,
        balanced_initialization=False,
        seed=2,
    )
    session, _ = configure_algorithm(algorithm)

    population = tuple(algorithm.ask())

    assert len(population) == 9
    assert len({individual.genotype for individual in population}) <= 8


def test_nsga2_returns_successful_pareto_front() -> None:
    algorithm = NSGA2Search(
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    session, runtime = configure_algorithm(algorithm)
    population = tuple(algorithm.ask())
    values = (
        (1.0, 4.0),
        (2.0, 3.0),
        (1.5, 1.0),
        (0.0, 5.0),
    )

    evaluations = tuple(
        evaluation(individual, quality=quality, latency=latency)
        for individual, (quality, latency) in zip(
            population,
            values,
            strict=True,
        )
    )
    algorithm.tell(runtime.evaluate_batch(tuple(reversed(evaluations))))

    assert {individual.id for individual in algorithm.best_individuals()} == {
        population[1].id,
        population[2].id,
    }
    assert algorithm.should_stop()


def test_nsga2_treats_failed_evaluations_as_infeasible() -> None:
    algorithm = NSGA2Search(
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    session, runtime = configure_algorithm(algorithm)
    population = tuple(algorithm.ask())
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

    algorithm.tell(runtime.evaluate_batch(evaluations))

    assert population[-1] not in algorithm.best_individuals()
    assert algorithm.best_individuals()


def test_nsga2_treats_invalid_objectives_as_infeasible() -> None:
    algorithm = NSGA2Search(
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    session, runtime = configure_algorithm(algorithm)
    population = tuple(algorithm.ask())
    evaluations = [
        evaluation(individual, quality=float(index), latency=float(4 - index))
        for index, individual in enumerate(population)
    ]
    evaluations[-1] = Evaluation(
        population[-1],
        Result.success(
            population[-1].id,
            metrics={"quality": 1000.0},
        ),
    )

    algorithm.tell(runtime.evaluate_batch(evaluations))

    assert population[-1] not in algorithm.best_individuals()
    assert algorithm.best_individuals()


def test_nsga2_generates_valid_categorical_offspring() -> None:
    search_space = make_search_space()
    algorithm = NSGA2Search(
        population_size=4,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        mutation_probability=1.0,
        seed=11,
    )
    session, runtime = configure_algorithm(algorithm, search_space=search_space)
    first = tuple(algorithm.ask())
    algorithm.tell(
        runtime.evaluate_batch(
            tuple(
                evaluation(
                    individual,
                    quality=float(index),
                    latency=float(index),
                )
                for index, individual in enumerate(first)
            )
        )
    )

    second = tuple(algorithm.ask())

    assert len(second) == 4
    assert all(
        search_space.to_genotype(individual) == individual.genotype
        and individual.metadata["algorithm"]["generation"] == 2
        and individual.metadata["algorithm"]["proposal_origin"] == "offspring"
        for individual in second
    )


def test_nsga2_checkpoint_replay_preserves_next_generation() -> None:
    search_space = make_search_space()
    original = NSGA2Search(
        population_size=4,
        max_generations=3,
        initial_population=INITIAL_POPULATION,
        seed=17,
    )
    session, runtime = configure_algorithm(original, search_space=search_space)
    first = tuple(original.ask())
    evaluated_batch = runtime.evaluate_batch(
        tuple(
            evaluation(
                individual,
                quality=float(index),
                latency=float(3 - index),
            )
            for index, individual in enumerate(first)
        ),
        batch_index=0,
    )
    original.tell(evaluated_batch)
    state = original.checkpoint_state()
    assert state == {"exhausted": False}
    expected = tuple(individual.genotype for individual in original.ask())

    restored = NSGA2Search(
        population_size=4,
        max_generations=3,
        initial_population=INITIAL_POPULATION,
        seed=17,
    )
    configure_algorithm(restored, search_space=search_space)
    restored.restore_checkpoint_state(
        state,
        search_space=search_space,
        evaluated_batches=(evaluated_batch,),
    )
    actual = tuple(individual.genotype for individual in restored.ask())

    assert actual == expected


def test_nsga2_requires_configuration_before_checkpoint_restore() -> None:
    algorithm = NSGA2Search(population_size=4)

    with pytest.raises(RuntimeError, match="not been configured"):
        algorithm.restore_checkpoint_state(
            {"exhausted": False},
            search_space=make_search_space(),
        )


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
    produced_artifacts = {"objectives": MetricArtifactImpl}

    def create_task(self, individual, artifacts):
        del artifacts
        return ObjectiveTask(individual=individual, step_id=self.id)


def test_nsga2_runs_through_optimization_session(tmp_path) -> None:
    objective_set = ObjectiveSet(
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
    )
    algorithm = NSGA2Search(
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
        objective_set=objective_set,
    ).run()

    assert len(result.evaluations) == 8
    assert result.best_individuals
