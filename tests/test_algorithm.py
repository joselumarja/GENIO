from __future__ import annotations

from random import Random

import pytest

from genio import GridSearch, RandomSearch, SearchSpace, StageChoice
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


def test_grid_search_enumerates_indexes_in_order() -> None:
    session = DummySession(make_search_space())
    algorithm = GridSearch(max_evaluations=3, batch_size=2)

    first_batch = algorithm.ask(session)
    second_batch = algorithm.ask(session)

    assert [individual.search_index for individual in first_batch] == [0, 1]
    assert [individual.search_index for individual in second_batch] == [2]
    assert algorithm.should_stop()


def test_grid_search_stops_when_space_is_exhausted() -> None:
    session = DummySession(make_search_space())
    algorithm = GridSearch(batch_size=10)

    batch = algorithm.ask(session)

    assert [individual.search_index for individual in batch] == [0, 1, 2, 3]
    assert algorithm.should_stop()
    assert algorithm.ask(session) == ()


def test_random_search_respects_max_evaluations() -> None:
    session = DummySession(make_search_space())
    algorithm = RandomSearch(
        max_evaluations=3,
        batch_size=2,
        unique=True,
        random=Random(0),
    )

    first_batch = algorithm.ask(session)
    second_batch = algorithm.ask(session)
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

    batches = []
    while not algorithm.should_stop():
        batches.append(algorithm.ask(session))
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
