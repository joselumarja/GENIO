from __future__ import annotations

from random import Random

import pytest

from genio import Evaluation, GridSearch, RandomSearch, Result, SearchSpace, StageChoice
from genio.algorithm.base import SearchAlgorithm, SearchContext
from genio.objective import (
    EvaluatedBatch,
    EvaluatedIndividual,
    MetricObjective,
    ObjectiveEvaluationStatus,
    ObjectiveSet,
)
from genio.search_space import SearchScenarioSpec, SlotSpec


class DummySession:
    def __init__(self, search_space: SearchSpace) -> None:
        self.search_space = search_space


def make_search_space() -> SearchSpace:
    return SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="algorithm_test",
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
        )
    )


def make_search_context(search_space: SearchSpace | None = None) -> SearchContext:
    objective_set = ObjectiveSet((MetricObjective("score", "maximize"),))
    return SearchContext(
        search_space=search_space or make_search_space(),
        objective_schema=objective_set.schema,
    )


def test_algorithm_context_requires_configuration() -> None:
    algorithm = RandomSearch(max_evaluations=1)

    with pytest.raises(RuntimeError, match="has not been configured"):
        _ = algorithm.context


def test_algorithm_configure_is_idempotent_for_same_context() -> None:
    algorithm = RandomSearch(max_evaluations=1)
    context = make_search_context()

    algorithm.configure(context)
    algorithm.configure(context)

    assert algorithm.context is context


def test_algorithm_configure_rejects_different_context() -> None:
    algorithm = RandomSearch(max_evaluations=1)
    context = make_search_context()
    different_context = SearchContext(
        search_space=context.search_space,
        objective_schema=context.objective_schema,
    )

    algorithm.configure(context)

    with pytest.raises(RuntimeError, match="different context"):
        algorithm.configure(different_context)

    assert algorithm.context is context


@pytest.mark.parametrize(
    "algorithm",
    (GridSearch(max_evaluations=1), RandomSearch(max_evaluations=1)),
)
def test_search_algorithms_record_evaluated_batch(
    algorithm: SearchAlgorithm,
) -> None:
    session = DummySession(make_search_space())
    algorithm.configure(make_search_context(session.search_space))
    (individual,) = algorithm.ask()
    evaluation = Evaluation(
        individual=individual,
        result=Result.success(individual.id, {"score": 1.0}),
    )
    batch = EvaluatedBatch(
        items=(
            EvaluatedIndividual(
                evaluation=evaluation,
                objective_values=None,
                status=ObjectiveEvaluationStatus.NOT_CONFIGURED,
            ),
        ),
        objective_names=(),
        batch_index=0,
    )

    algorithm.tell(batch)

    assert algorithm._evaluations == [evaluation]
    assert "evaluations" not in algorithm.checkpoint_state()


def test_grid_search_enumerates_indexes_in_order() -> None:
    session = DummySession(make_search_space())
    algorithm = GridSearch(max_evaluations=3, batch_size=2)
    algorithm.configure(make_search_context(session.search_space))

    first_batch = algorithm.ask()
    second_batch = algorithm.ask()

    assert [individual.search_index for individual in first_batch] == [0, 1]
    assert [individual.search_index for individual in second_batch] == [2]
    assert algorithm.should_stop()


def test_grid_search_stops_when_space_is_exhausted() -> None:
    session = DummySession(make_search_space())
    algorithm = GridSearch(batch_size=10)
    algorithm.configure(make_search_context(session.search_space))

    batch = algorithm.ask()

    assert [individual.search_index for individual in batch] == [0, 1, 2, 3]
    assert algorithm.should_stop()
    assert algorithm.ask() == ()


def test_random_search_respects_max_evaluations() -> None:
    session = DummySession(make_search_space())
    algorithm = RandomSearch(
        max_evaluations=3,
        batch_size=2,
        unique=True,
        random=Random(0),
    )
    algorithm.configure(make_search_context(session.search_space))

    first_batch = algorithm.ask()
    second_batch = algorithm.ask()
    indexes = [
        individual.search_index
        for individual in (*first_batch, *second_batch)
    ]

    assert len(indexes) == 3
    assert algorithm.should_stop()


def test_random_search_does_not_track_global_uniqueness() -> None:
    session = DummySession(make_search_space())
    algorithm = RandomSearch(
        max_evaluations=10,
        batch_size=3,
        unique=False,
        random=Random(0),
    )
    algorithm.configure(make_search_context(session.search_space))

    batches = []
    while not algorithm.should_stop():
        batches.append(algorithm.ask())
    indexes = [individual.search_index for batch in batches for individual in batch]

    assert len(indexes) == 10
    assert len(set(indexes)) <= 4


def test_algorithm_constructor_validation() -> None:
    with pytest.raises(ValueError, match="max_evaluations"):
        RandomSearch(max_evaluations=-1)

    with pytest.raises(ValueError, match="batch_size"):
        GridSearch(batch_size=0)

    with pytest.raises(ValueError, match="start_index"):
        GridSearch(start_index=-1)


@pytest.mark.parametrize(
    ("constructor", "kwargs", "message"),
    (
        (RandomSearch, {"max_evaluations": True}, "max_evaluations"),
        (RandomSearch, {"max_evaluations": 1.5}, "max_evaluations"),
        (RandomSearch, {"max_evaluations": 1, "batch_size": True}, "batch_size"),
        (RandomSearch, {"max_evaluations": 1, "unique": 1}, "unique"),
        (RandomSearch, {"max_evaluations": 1, "balanced": 0}, "balanced"),
        (GridSearch, {"max_evaluations": True}, "max_evaluations"),
        (GridSearch, {"batch_size": 1.5}, "batch_size"),
        (GridSearch, {"start_index": True}, "start_index"),
    ),
)
def test_random_and_grid_reject_bool_and_non_integer_configuration(
    constructor,
    kwargs,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        constructor(**kwargs)
