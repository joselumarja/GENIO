from random import Random

from genio import SearchScenarioSpec
from genio import SearchSpace
from genio import SlotSpec
from genio import StageChoice


def make_search_space() -> SearchSpace:
    return SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="design_space_test",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(
                        StageChoice(slot=0, stage="threshold"),
                        StageChoice(slot=0, stage="adaptive_threshold"),
                    ),
                ),
            ),
            design_spaces={
                "hls": {
                    "pipeline_ii": (1, 2),
                },
                "system": {
                    "memory_size": (32768, 65536),
                },
            },
        )
    )


def test_design_space_extends_genotype_and_index() -> None:
    search_space = make_search_space()

    individual = search_space.from_genotype((1, 0, 1), id="design_001")

    assert search_space.slot_lengths == (2,)
    assert search_space.design_lengths == (2, 2)
    assert search_space.genotype_lengths == (2, 2, 2)
    assert search_space.search_space_size == 8
    assert individual.stage_sequence() == ("adaptive_threshold",)
    assert individual.design == {
        "hls": {"pipeline_ii": 1},
        "system": {"memory_size": 65536},
    }
    assert individual.search_index == 5
    assert search_space.index_to_genotype(5) == (1, 0, 1)


def test_design_space_roundtrips_from_individual() -> None:
    search_space = make_search_space()
    individual = search_space.from_genotype((0, 1, 1))

    assert search_space.to_genotype(individual) == (0, 1, 1)
    assert search_space.to_index(individual) == individual.search_index


def test_design_space_is_sampled_with_pipeline() -> None:
    search_space = make_search_space()

    individual = search_space.sample(random=Random(0))

    assert len(individual.genotype) == 3
    assert set(individual.design) == {"hls", "system"}
