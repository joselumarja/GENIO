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
from genio.algorithm.base import SearchContext
from genio.objective import (
    EvaluatedBatch,
    MinMaxNormalizer,
    NormalizationScope,
    ObjectiveRuntime,
    WeightedMeanScalarizer,
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


def score_objective_set(metric: str = "score") -> ObjectiveSet:
    return ObjectiveSet(
        (score_objective(metric),),
        scalarizer=WeightedMeanScalarizer(),
    )


def configure_algorithm(
    algorithm: GeneticSearch,
    session: DummySession,
    objective_set: ObjectiveSet | None = None,
) -> ObjectiveRuntime:
    configured_objectives = objective_set or score_objective_set()
    normalization_scope = getattr(configured_objectives.normalizer, "scope", None)
    algorithm.configure(
        SearchContext(
            search_space=session.search_space,
            objective_schema=configured_objectives.schema,
            has_normalizer=configured_objectives.normalizer is not None,
            has_scalarizer=configured_objectives.scalarizer is not None,
            normalization_scope=(
                normalization_scope.value if normalization_scope is not None else None
            ),
        )
    )
    return ObjectiveRuntime(configured_objectives)


def tell_evaluations(
    algorithm: GeneticSearch,
    runtime: ObjectiveRuntime,
    evaluations,
    *,
    batch_index: int | None = None,
) -> EvaluatedBatch:
    batch = runtime.evaluate_batch(tuple(evaluations), batch_index=batch_index)
    algorithm.tell(batch)
    return batch


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
        ({"max_generations": True}, "max_generations"),
        ({"max_generations": 1.5}, "max_generations"),
        ({"start_generation": 0}, "start_generation"),
        ({"start_generation": True}, "start_generation"),
        ({"mutation_probability": -0.1}, "mutation_probability"),
        ({"mutation_probability": 1.1}, "mutation_probability"),
        ({"balanced_initialization": 1}, "balanced_initialization"),
    ),
)
def test_genetic_search_validates_scalar_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        GeneticSearch(**kwargs)


def test_genetic_search_validates_objective_context() -> None:
    space = make_search_space()
    algorithm = GeneticSearch()
    with pytest.raises(ValueError, match="objective schema"):
        algorithm.configure(SearchContext(search_space=space))

    objective_set = ObjectiveSet((score_objective(),), scalarizer=None)
    with pytest.raises(ValueError, match="scalarizer"):
        algorithm.configure(
            SearchContext(
                search_space=space,
                objective_schema=objective_set.schema,
            )
        )

    valid_objectives = score_objective_set()
    valid_context = SearchContext(
        search_space=space,
        objective_schema=valid_objectives.schema,
        has_scalarizer=True,
    )
    algorithm.configure(valid_context)
    assert algorithm.context is valid_context


@pytest.mark.parametrize(
    "scope",
    (NormalizationScope.BATCH, NormalizationScope.CUMULATIVE),
)
def test_genetic_search_rejects_variable_normalization_across_generations(
    scope,
) -> None:
    objective_set = ObjectiveSet(
        (score_objective(),),
        normalizer=MinMaxNormalizer(scope),
        scalarizer=WeightedMeanScalarizer(),
    )

    with pytest.raises(ValueError, match=rf"{scope.value.upper()} normalization"):
        configure_algorithm(
            GeneticSearch(max_generations=2),
            DummySession(make_search_space()),
            objective_set,
        )


def test_objective_set_validates_and_normalizes_genetic_weights() -> None:
    objective_definitions = (
        MetricObjective("quality", OptimizationDirection.MAXIMIZE),
        MetricObjective("latency", OptimizationDirection.MINIMIZE),
    )

    with pytest.raises(ValueError, match="exactly match"):
        ObjectiveSet(
            objective_definitions,
            scalarizer=WeightedMeanScalarizer({"quality": 1.0}),
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        WeightedMeanScalarizer({"quality": 1.0, "latency": -1.0})
    with pytest.raises(ValueError, match="At least one"):
        WeightedMeanScalarizer({"quality": 0.0, "latency": 0.0})

    scalarizer = WeightedMeanScalarizer(
        {"quality": 1e308, "latency": 1e308}
    )
    objective_set = ObjectiveSet(
        objective_definitions,
        scalarizer=scalarizer,
    )

    assert dict(objective_set.scalarizer.weights) == {
        "quality": 0.5,
        "latency": 0.5,
    }


def test_genetic_search_validates_initial_population() -> None:
    with pytest.raises(ValueError, match="initial_population"):
        GeneticSearch(
            population_size=4,
            initial_population=INITIAL_POPULATION[:2],
        )
    with pytest.raises(ValueError, match="initial_population is required"):
        GeneticSearch(
            population_size=4,
            start_generation=2,
        )
    with pytest.raises(ValueError, match="only integers"):
        GeneticSearch(
            population_size=4,
            initial_population=((True, 0, 0), *INITIAL_POPULATION[1:]),
        )


def test_genetic_search_materializes_provided_initial_population() -> None:
    session = DummySession(make_search_space())
    algorithm = GeneticSearch(
        population_size=4,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        random=Random(7),
    )
    configure_algorithm(algorithm, session)

    population = tuple(algorithm.ask())

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
        algorithm.ask()


def test_genetic_search_tell_validates_and_reorders_pending_generation() -> None:
    space = make_search_space()
    session = DummySession(space)
    algorithm = GeneticSearch(
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    runtime = configure_algorithm(algorithm, session)
    population = tuple(algorithm.ask())
    evaluations = tuple(
        make_evaluation(individual, score=float(position))
        for position, individual in enumerate(population)
    )

    with pytest.raises(ValueError, match="Expected 4"):
        tell_evaluations(algorithm, runtime, evaluations[:3])
    with pytest.raises(ValueError, match="Duplicate"):
        tell_evaluations(
            algorithm,
            runtime,
            (evaluations[0], evaluations[0], evaluations[2], evaluations[3]),
        )

    unexpected = space.from_genotype((0, 0, 0))
    with pytest.raises(ValueError, match="unexpected"):
        tell_evaluations(
            algorithm,
            runtime,
            (*evaluations[:3], make_evaluation(unexpected, score=9.0)),
        )

    tell_evaluations(algorithm, runtime, reversed(evaluations))

    assert algorithm.should_stop()
    assert algorithm.ask() == ()
    assert algorithm.best_individuals() == (population[3],)
    assert algorithm.generation_best_individuals() == (population[3],)
    assert algorithm.generation_fitnesses() == (
        {
            population[0].id: 0.0,
            population[1].id: 1.0,
            population[2].id: 2.0,
            population[3].id: 3.0,
        },
    )
    with pytest.raises(RuntimeError, match="pending generation"):
        tell_evaluations(algorithm, runtime, evaluations)


def test_genetic_search_uses_median_roulette_and_full_replacement() -> None:
    session = DummySession(make_search_space())
    algorithm = GeneticSearch(
        population_size=4,
        mutation_probability=0.0,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        random=Random(4),
    )
    runtime = configure_algorithm(algorithm, session)
    first_generation = tuple(algorithm.ask())
    tell_evaluations(
        algorithm,
        runtime,
        (
            make_evaluation(individual, score=float(position))
            for position, individual in enumerate(first_generation)
        ),
    )

    second_generation = tuple(algorithm.ask())
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
        ),
        scalarizer=WeightedMeanScalarizer(
            {"quality": 0.5, "latency": 0.5}
        ),
    )
    algorithm = GeneticSearch(
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    runtime = configure_algorithm(algorithm, session, objectives)
    population = tuple(algorithm.ask())
    values = (
        {"quality": 0.0, "latency": 40.0},
        {"quality": 1.0, "latency": 40.0},
        {"quality": 0.0, "latency": 100.0},
        {"quality": 1.0, "latency": 100.0},
    )

    tell_evaluations(
        algorithm,
        runtime,
        (
            make_evaluation(individual, **metrics)
            for individual, metrics in zip(population, values, strict=True)
        ),
    )

    assert algorithm.best_individuals() == (population[1],)


def test_genetic_search_does_not_bias_fitness_with_constant_objective() -> None:
    session = DummySession(make_search_space())
    algorithm = GeneticSearch(
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    runtime = configure_algorithm(
        algorithm,
        session,
        ObjectiveSet(
            (MetricObjective("latency", OptimizationDirection.MINIMIZE),),
            scalarizer=WeightedMeanScalarizer(),
        ),
    )
    population = tuple(algorithm.ask())

    tell_evaluations(
        algorithm,
        runtime,
        (make_evaluation(individual, latency=10.0) for individual in population),
    )

    assert tuple(algorithm.generation_fitnesses()[0].values()) == (
        -10.0,
        -10.0,
        -10.0,
        -10.0,
    )


def test_genetic_search_mutates_one_gene_after_crossover() -> None:
    session = DummySession(make_search_space())
    algorithm = GeneticSearch(
        population_size=4,
        mutation_probability=1.0,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        random=Random(11),
    )
    runtime = configure_algorithm(algorithm, session)
    first_generation = tuple(algorithm.ask())
    tell_evaluations(
        algorithm,
        runtime,
        (
            make_evaluation(individual, score=float(position))
            for position, individual in enumerate(first_generation)
        ),
    )

    second_generation = tuple(algorithm.ask())

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
        population_size=4,
        mutation_probability=0.0,
        max_generations=4,
        start_generation=3,
        initial_population=INITIAL_POPULATION,
        random=Random(9),
    )
    runtime = configure_algorithm(algorithm, session)

    third_generation = tuple(algorithm.ask())
    tell_evaluations(
        algorithm,
        runtime,
        (
            make_evaluation(individual, score=float(position))
            for position, individual in enumerate(third_generation)
        ),
    )
    fourth_generation = tuple(algorithm.ask())

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
        population_size=4,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        random=Random(3),
    )
    runtime = configure_algorithm(algorithm, session)
    first_generation = tuple(algorithm.ask())

    tell_evaluations(
        algorithm,
        runtime,
        (
            make_failed_evaluation(individual, score=1000.0)
            for individual in first_generation
        ),
    )
    second_generation = tuple(algorithm.ask())

    assert algorithm.best_individuals() == ()
    assert algorithm.generation_best_individuals() == ()
    assert all(
        individual.metadata["algorithm"]["proposal_origin"] == "restart"
        and individual.metadata["algorithm"]["parent_ids"] == []
        for individual in second_generation
    )


def test_genetic_search_assigns_zero_fitness_to_failed_and_invalid_evaluations() -> None:
    session = DummySession(make_search_space())
    algorithm = GeneticSearch(
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    runtime = configure_algorithm(algorithm, session)
    population = tuple(algorithm.ask())

    tell_evaluations(
        algorithm,
        runtime,
        (
            make_evaluation(population[0], unrelated=1000.0),
            make_failed_evaluation(population[1], score=1000.0),
            make_evaluation(population[2], score=2.0),
            make_evaluation(population[3], score=3.0),
        ),
    )

    assert algorithm.best_individuals() == (population[3],)
    assert algorithm.generation_fitnesses() == (
        {
            population[0].id: 0.0,
            population[1].id: 0.0,
            population[2].id: 2.0,
            population[3].id: 3.0,
        },
    )


def test_genetic_search_roulette_supports_negative_scores() -> None:
    session = DummySession(make_search_space())
    algorithm = GeneticSearch(
        population_size=4,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        mutation_probability=0.0,
        random=Random(4),
    )
    runtime = configure_algorithm(algorithm, session)
    population = tuple(algorithm.ask())
    evaluated_batch = tell_evaluations(
        algorithm,
        runtime,
        (
            make_evaluation(individual, score=float(position - 4))
            for position, individual in enumerate(population)
        ),
        batch_index=0,
    )

    state = algorithm.checkpoint_state()
    assert "evaluations" not in state
    restored = GeneticSearch(
        population_size=4,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
        mutation_probability=0.0,
        random=Random(999),
    )
    configure_algorithm(restored, session)
    restored.restore_checkpoint_state(
        state,
        search_space=session.search_space,
        evaluated_batches=(evaluated_batch,),
    )
    assert restored.generation_fitnesses() == algorithm.generation_fitnesses()

    offspring = tuple(algorithm.ask())
    eligible_parent_ids = {population[2].id, population[3].id}
    assert all(
        set(individual.metadata["algorithm"]["parent_ids"]).issubset(
            eligible_parent_ids
        )
        for individual in offspring
    )
    assert [item.genotype for item in restored.ask()] == [
        item.genotype for item in offspring
    ]


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


def test_objective_runtime_failure_leaves_genetic_generation_pending() -> None:
    session = DummySession(make_search_space())
    objective = FailOnceObjective(fail_on_call=1)
    algorithm = GeneticSearch(
        population_size=4,
        max_generations=1,
        initial_population=INITIAL_POPULATION,
    )
    runtime = configure_algorithm(
        algorithm,
        session,
        ObjectiveSet(
            (objective,),
            scalarizer=WeightedMeanScalarizer(),
        ),
    )
    population = tuple(algorithm.ask())
    evaluations = tuple(
        make_evaluation(individual, score=float(position))
        for position, individual in enumerate(population)
    )

    with pytest.raises(RuntimeError, match="objective failed"):
        runtime.evaluate_batch(evaluations)

    assert algorithm.generation_best_individuals() == ()
    assert algorithm.generation_fitnesses() == ()
    tell_evaluations(algorithm, runtime, evaluations)
    assert algorithm.generation_best_individuals() == (population[3],)
    assert objective.calls == 5


def test_genetic_search_handles_zero_generations_and_invalid_initial_genotype() -> None:
    session = DummySession(make_search_space())
    stopped = GeneticSearch(
        population_size=4,
        max_generations=0,
    )
    configure_algorithm(stopped, session)
    assert stopped.ask() == ()

    invalid = GeneticSearch(
        population_size=4,
        max_generations=1,
        initial_population=((0, 0, 9), *INITIAL_POPULATION[1:]),
    )
    configure_algorithm(invalid, session)
    with pytest.raises(ValueError, match="out of range"):
        invalid.ask()


def test_genetic_search_is_deterministic_with_seeded_random() -> None:
    first_space = make_search_space()
    second_space = make_search_space()
    first = GeneticSearch(
        population_size=4,
        max_generations=2,
        random=Random(21),
    )
    second = GeneticSearch(
        population_size=4,
        max_generations=2,
        random=Random(21),
    )
    first_session = DummySession(first_space)
    second_session = DummySession(second_space)
    first_runtime = configure_algorithm(first, first_session)
    second_runtime = configure_algorithm(second, second_session)

    first_initial = tuple(first.ask())
    second_initial = tuple(second.ask())
    assert [item.genotype for item in first_initial] == [
        item.genotype for item in second_initial
    ]

    tell_evaluations(
        first,
        first_runtime,
        (
            make_evaluation(individual, score=float(individual.search_index))
            for individual in first_initial
        ),
    )
    tell_evaluations(
        second,
        second_runtime,
        (
            make_evaluation(individual, score=float(individual.search_index))
            for individual in second_initial
        ),
    )

    assert [item.genotype for item in first.ask()] == [
        item.genotype for item in second.ask()
    ]


def test_genetic_search_cannot_be_reused_with_another_search_space() -> None:
    first_space = make_search_space()
    algorithm = GeneticSearch(
        population_size=4,
        max_generations=2,
        initial_population=INITIAL_POPULATION,
    )
    first_session = DummySession(first_space)
    runtime = configure_algorithm(algorithm, first_session)
    first_generation = tuple(algorithm.ask())
    tell_evaluations(
        algorithm,
        runtime,
        (
            make_evaluation(individual, score=float(position))
            for position, individual in enumerate(first_generation)
        ),
    )

    different_objectives = score_objective_set()
    with pytest.raises(RuntimeError, match="different context"):
        algorithm.configure(
            SearchContext(
                search_space=make_search_space(),
                objective_schema=different_objectives.schema,
                has_scalarizer=True,
            )
        )


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
    produced_artifacts = {"gene-score": GeneMetricArtifact}

    def create_task(self, individual, artifacts):
        return GeneMetricTask(individual=individual, step_id=self.id)


def test_genetic_search_runs_complete_generations_in_optimization_session(tmp_path) -> None:
    objective_set = ObjectiveSet(
        (
            MetricObjective(
                "fitness.score",
                OptimizationDirection.MAXIMIZE,
            ),
        ),
        scalarizer=WeightedMeanScalarizer(),
    )
    algorithm = GeneticSearch(
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
        objective_set=objective_set,
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
