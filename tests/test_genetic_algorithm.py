from __future__ import annotations

from dataclasses import dataclass
from random import Random

import pytest

from genio import (
    Evaluation,
    EvaluationStep,
    EvaluationTask,
    EvaluationWorkflow,
    GeneticSearch,
    LocalBackend,
    MetricArtifact,
    MetricObjective,
    Objective,
    ObjectiveSet,
    OptimizationDirection,
    OptimizationSession,
    Result,
    SearchSpace,
    StageChoice,
)
from genio.search_space import SearchScenarioSpec, SlotSpec


class DummySession:
    def __init__(self, search_space: SearchSpace) -> None:
        self.search_space = search_space


def make_search_space() -> SearchSpace:
    return SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="genetic_test",
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


def score_objective(metric: str = "score") -> MetricObjective:
    return MetricObjective(metric, OptimizationDirection.MAXIMIZE)


def make_evaluation(individual, **metrics: float) -> Evaluation:
    return Evaluation(
        individual=individual,
        result=Result.success(individual.id, metrics=metrics),
    )


def make_failed_evaluation(individual, **metrics: float) -> Evaluation:
    return Evaluation(
        individual=individual,
        result=Result.failed(individual.id, "failed", metrics=metrics),
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
        ({"population_size": 3}, "population_size"),
        ({"population_size": True}, "population_size"),
        ({"max_generations": -1}, "max_generations"),
        ({"max_generations": 1.5}, "max_generations"),
        ({"start_generation": 0}, "start_generation"),
        ({"mutation_probability": -0.1}, "mutation_probability"),
        ({"mutation_probability": 1.1}, "mutation_probability"),
    ),
)
def test_genetic_search_validates_scalar_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        GeneticSearch(objectives=score_objective(), **kwargs)


def test_genetic_search_validates_weights_and_initial_population() -> None:
    objectives = ObjectiveSet(
        (
            MetricObjective("quality", OptimizationDirection.MAXIMIZE),
            MetricObjective("latency", OptimizationDirection.MINIMIZE),
        )
    )

    with pytest.raises(ValueError, match="weights must match"):
        GeneticSearch(objectives=objectives, weights={"quality": 1.0})
    with pytest.raises(ValueError, match="finite and non-negative"):
        GeneticSearch(
            objectives=objectives,
            weights={"quality": 1.0, "latency": -1.0},
        )
    with pytest.raises(ValueError, match="At least one"):
        GeneticSearch(
            objectives=objectives,
            weights={"quality": 0.0, "latency": 0.0},
        )
    with pytest.raises(ValueError, match="initial_population"):
        GeneticSearch(
            objectives=objectives,
            population_size=4,
            initial_population=INITIAL_POPULATION[:2],
        )
    with pytest.raises(ValueError, match="initial_population is required"):
        GeneticSearch(
            objectives=objectives,
            population_size=4,
            start_generation=2,
        )

    algorithm = GeneticSearch(
        objectives=objectives,
        weights={"quality": 1e308, "latency": 1e308},
    )
    assert algorithm.weights == (0.5, 0.5)


def test_genetic_search_materializes_provided_initial_population() -> None:
    session = DummySession(make_search_space())
    algorithm = GeneticSearch(
        objectives=score_objective(),
        population_size=4,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        random=Random(7),
    )

    population = tuple(algorithm.ask(session))

    assert [individual.genotype for individual in population] == list(INITIAL_POPULATION)
    assert len({individual.id for individual in population}) == 4
    assert [
        individual.metadata["algorithm"]["population_index"]
        for individual in population
    ] == [0, 1, 2, 3]
    assert all(
        individual.metadata["algorithm"] == {
            "generation": 1,
            "population_index": position,
            "proposal_origin": "initial_population",
            "parent_ids": [],
            "mutation_applied": False,
            "mutation_changed": False,
        }
        for position, individual in enumerate(population)
    )
    with pytest.raises(RuntimeError, match=r"tell\(\)"):
        algorithm.ask(session)


def test_genetic_search_tell_validates_and_reorders_pending_generation() -> None:
    space = make_search_space()
    session = DummySession(space)
    algorithm = GeneticSearch(
        objectives=score_objective(),
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    population = tuple(algorithm.ask(session))
    evaluations = tuple(
        make_evaluation(individual, score=float(position))
        for position, individual in enumerate(population)
    )

    with pytest.raises(ValueError, match="Expected 4"):
        algorithm.tell(evaluations[:3])
    with pytest.raises(ValueError, match="Duplicate"):
        algorithm.tell((evaluations[0], evaluations[0], evaluations[2], evaluations[3]))

    unexpected = space.from_genotype((0, 0, 0))
    with pytest.raises(ValueError, match="unexpected"):
        algorithm.tell((*evaluations[:3], make_evaluation(unexpected, score=9.0)))

    mismatched_result = Evaluation(
        individual=population[0],
        result=Result.success("another-individual", metrics={"score": 0.0}),
    )
    with pytest.raises(ValueError, match="Result individual ID"):
        algorithm.tell((mismatched_result, *evaluations[1:]))

    algorithm.tell(tuple(reversed(evaluations)))

    assert algorithm.should_stop()
    assert algorithm.ask(session) == ()
    assert algorithm.best_individuals() == (population[3],)
    assert algorithm.generation_best_individuals() == (population[3],)
    assert algorithm.generation_fitnesses() == (
        {
            population[0].id: 0.0,
            population[1].id: pytest.approx(1 / 3),
            population[2].id: pytest.approx(2 / 3),
            population[3].id: 1.0,
        },
    )
    with pytest.raises(RuntimeError, match="pending generation"):
        algorithm.tell(evaluations)


def test_genetic_search_uses_median_roulette_and_full_replacement() -> None:
    session = DummySession(make_search_space())
    algorithm = GeneticSearch(
        objectives=score_objective(),
        population_size=4,
        mutation_probability=0.0,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        random=Random(4),
    )
    first_generation = tuple(algorithm.ask(session))
    algorithm.tell(
        tuple(
            make_evaluation(individual, score=float(position))
            for position, individual in enumerate(first_generation)
        )
    )

    second_generation = tuple(algorithm.ask(session))
    eligible_parent_ids = {first_generation[2].id, first_generation[3].id}

    assert len(second_generation) == 4
    assert not ({individual.id for individual in first_generation} & {
        individual.id for individual in second_generation
    })
    assert all(len(individual.genotype) == 3 for individual in second_generation)
    assert all(
        set(individual.metadata["algorithm"]["parent_ids"]).issubset(
            eligible_parent_ids
        )
        for individual in second_generation
    )
    assert all(
        individual.metadata["algorithm"]["proposal_origin"] == "crossover"
        and individual.metadata["algorithm"]["generation"] == 2
        for individual in second_generation
    )


def test_genetic_search_applies_weighted_maximize_and_minimize_objectives() -> None:
    session = DummySession(make_search_space())
    objectives = ObjectiveSet(
        (
            MetricObjective("quality", OptimizationDirection.MAXIMIZE),
            MetricObjective("latency", OptimizationDirection.MINIMIZE),
        )
    )
    algorithm = GeneticSearch(
        objectives=objectives,
        weights={"quality": 0.5, "latency": 0.5},
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    population = tuple(algorithm.ask(session))
    values = (
        {"quality": 0.0, "latency": 40.0},
        {"quality": 1.0, "latency": 40.0},
        {"quality": 0.0, "latency": 100.0},
        {"quality": 1.0, "latency": 100.0},
    )

    algorithm.tell(
        tuple(
            make_evaluation(individual, **metrics)
            for individual, metrics in zip(population, values, strict=True)
        )
    )

    assert algorithm.best_individuals() == (population[1],)


def test_genetic_search_does_not_bias_fitness_with_constant_objective() -> None:
    session = DummySession(make_search_space())
    algorithm = GeneticSearch(
        objectives=MetricObjective("latency", OptimizationDirection.MINIMIZE),
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    population = tuple(algorithm.ask(session))

    algorithm.tell(
        tuple(make_evaluation(individual, latency=10.0) for individual in population)
    )

    assert tuple(algorithm.generation_fitnesses()[0].values()) == (0.0, 0.0, 0.0, 0.0)


def test_genetic_search_mutates_one_gene_after_crossover() -> None:
    session = DummySession(make_search_space())
    algorithm = GeneticSearch(
        objectives=score_objective(),
        population_size=4,
        mutation_probability=1.0,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        random=Random(11),
    )
    first_generation = tuple(algorithm.ask(session))
    algorithm.tell(
        tuple(
            make_evaluation(individual, score=float(position))
            for position, individual in enumerate(first_generation)
        )
    )

    second_generation = tuple(algorithm.ask(session))

    assert all(
        individual.metadata["algorithm"]["proposal_origin"] == "crossover"
        and individual.metadata["algorithm"]["mutation_applied"] is True
        and isinstance(
            individual.metadata["algorithm"]["mutation_changed"],
            bool,
        )
        and len(individual.metadata["algorithm"]["parent_ids"]) == 2
        for individual in second_generation
    )


def test_genetic_search_resumes_from_configured_generation() -> None:
    session = DummySession(make_search_space())
    algorithm = GeneticSearch(
        objectives=score_objective(),
        population_size=4,
        mutation_probability=0.0,
        max_generations=4,
        start_generation=3,
        initial_population=INITIAL_POPULATION,
        random=Random(9),
    )

    third_generation = tuple(algorithm.ask(session))
    algorithm.tell(
        tuple(
            make_evaluation(individual, score=float(position))
            for position, individual in enumerate(third_generation)
        )
    )
    fourth_generation = tuple(algorithm.ask(session))

    assert all(
        individual.metadata["algorithm"]["generation"] == 3
        for individual in third_generation
    )
    assert all(
        individual.metadata["algorithm"]["generation"] == 4
        for individual in fourth_generation
    )
    assert algorithm.should_stop()


def test_genetic_search_restarts_after_all_failed_generation() -> None:
    session = DummySession(make_search_space())
    algorithm = GeneticSearch(
        objectives=score_objective(),
        population_size=4,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        random=Random(3),
    )
    first_generation = tuple(algorithm.ask(session))

    algorithm.tell(
        tuple(make_failed_evaluation(individual, score=1000.0) for individual in first_generation)
    )
    second_generation = tuple(algorithm.ask(session))

    assert algorithm.best_individuals() == ()
    assert algorithm.generation_best_individuals() == ()
    assert all(
        individual.metadata["algorithm"]["proposal_origin"] == "restart"
        and individual.metadata["algorithm"]["parent_ids"] == []
        for individual in second_generation
    )


def test_genetic_search_ignores_partial_metrics_from_failed_evaluations() -> None:
    session = DummySession(make_search_space())
    algorithm = GeneticSearch(
        objectives=score_objective(),
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    population = tuple(algorithm.ask(session))

    algorithm.tell(
        (
            make_failed_evaluation(population[0], score=1000.0),
            make_evaluation(population[1], score=1.0),
            make_evaluation(population[2], score=2.0),
            make_evaluation(population[3], score=3.0),
        )
    )

    assert algorithm.best_individuals() == (population[3],)


class FailOnceObjective(Objective):
    def __init__(self, fail_on_call: int) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    @property
    def name(self) -> str:
        return "score"

    @property
    def direction(self) -> OptimizationDirection:
        return OptimizationDirection.MAXIMIZE

    def value(self, evaluation: Evaluation) -> float:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("objective failed")
        return float(evaluation.result.metrics["score"])


def test_genetic_search_tell_is_atomic_when_objective_fails() -> None:
    session = DummySession(make_search_space())
    objective = FailOnceObjective(fail_on_call=5)
    algorithm = GeneticSearch(
        objectives=objective,
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    population = tuple(algorithm.ask(session))
    evaluations = tuple(
        make_evaluation(individual, score=float(position))
        for position, individual in enumerate(population)
    )

    with pytest.raises(RuntimeError, match="objective failed"):
        algorithm.tell(evaluations)

    assert algorithm.generation_best_individuals() == ()
    assert algorithm.generation_fitnesses() == ()
    algorithm.tell(evaluations)
    assert algorithm.generation_best_individuals() == (population[3],)


def test_genetic_search_handles_zero_generations_and_invalid_initial_genotype() -> None:
    session = DummySession(make_search_space())
    stopped = GeneticSearch(
        objectives=score_objective(),
        population_size=4,
        max_generations=0,
    )
    assert stopped.ask(session) == ()

    invalid = GeneticSearch(
        objectives=score_objective(),
        population_size=4,
        max_generations=1,
        initial_population=((0, 0, 9), *INITIAL_POPULATION[1:]),
    )
    with pytest.raises(ValueError, match="out of range"):
        invalid.ask(session)


def test_genetic_search_is_deterministic_with_seeded_random() -> None:
    first_space = make_search_space()
    second_space = make_search_space()
    first = GeneticSearch(
        objectives=score_objective(),
        population_size=4,
        max_generations=2,
        random=Random(21),
    )
    second = GeneticSearch(
        objectives=score_objective(),
        population_size=4,
        max_generations=2,
        random=Random(21),
    )

    first_initial = tuple(first.ask(DummySession(first_space)))
    second_initial = tuple(second.ask(DummySession(second_space)))
    assert [item.genotype for item in first_initial] == [
        item.genotype for item in second_initial
    ]

    first.tell(
        tuple(
            make_evaluation(individual, score=float(individual.search_index))
            for individual in first_initial
        )
    )
    second.tell(
        tuple(
            make_evaluation(individual, score=float(individual.search_index))
            for individual in second_initial
        )
    )

    assert [item.genotype for item in first.ask(DummySession(first_space))] == [
        item.genotype for item in second.ask(DummySession(second_space))
    ]


def test_genetic_search_cannot_be_reused_with_another_search_space() -> None:
    first_space = make_search_space()
    algorithm = GeneticSearch(
        objectives=score_objective(),
        population_size=4,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
    )
    first_generation = tuple(algorithm.ask(DummySession(first_space)))
    algorithm.tell(
        tuple(
            make_evaluation(individual, score=float(position))
            for position, individual in enumerate(first_generation)
        )
    )

    with pytest.raises(RuntimeError, match="another SearchSpace"):
        algorithm.ask(DummySession(make_search_space()))


@dataclass(frozen=True, slots=True)
class GeneMetricArtifact(MetricArtifact):
    value: float = 0.0

    def load(self):
        return ()

    def metrics(self):
        return {"score": self.value}


class GeneMetricTask(EvaluationTask):
    def run(self, context):
        assert self.individual.genotype is not None
        return [
            GeneMetricArtifact(
                name="gene-score",
                producer="fitness",
                individual_id=self.individual.id,
                value=float(self.individual.genotype[0]),
            )
        ]


class GeneMetricStep(EvaluationStep):
    id = "fitness"
    task_type = GeneMetricTask

    def create_task(self, individual, artifacts):
        return GeneMetricTask(individual=individual, step_id=self.id)


def test_genetic_search_runs_complete_generations_in_optimization_session(tmp_path) -> None:
    algorithm = GeneticSearch(
        objectives=MetricObjective(
            "fitness.score",
            OptimizationDirection.MAXIMIZE,
        ),
        population_size=4,
        mutation_probability=0.0,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        random=Random(6),
    )
    session = OptimizationSession(
        search_space=make_search_space(),
        algorithm=algorithm,
        backend=LocalBackend(base_work_dir=tmp_path),
        evaluation_workflow=EvaluationWorkflow((GeneMetricStep(),)),
    )

    result = session.run()

    assert len(result.evaluations) == 8
    assert [evaluation.metadata["batch_index"] for evaluation in result.evaluations] == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]
    assert [
        evaluation.individual.metadata["algorithm"]["generation"]
        for evaluation in result.evaluations
    ] == [1, 1, 1, 1, 2, 2, 2, 2]
    assert len(result.best_individuals) == 1
    assert result.best_individuals[0].genotype[0] == 1
