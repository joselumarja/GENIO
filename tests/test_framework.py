from genio import (
    Candidate,
    Evaluator,
    LocalBackend,
    Result,
    ResultStatus,
    Runner,
    SearchScenarioSpec,
    SearchSpace,
    SlotSpec,
    StageChoice,
)


class SumRunner(Runner):
    def run(self, candidate: Candidate) -> Result:
        return Result.success(candidate.id, {"score": sum(candidate.parameters.values())})


def test_evaluator_returns_results_for_candidates():
    evaluator = Evaluator(LocalBackend(SumRunner()))

    results = evaluator.evaluate([
        Candidate("candidate_001", {"a": 1, "b": 2}),
        Candidate("candidate_002", {"a": 3, "b": 4}),
    ])

    assert [result.candidate_id for result in results] == ["candidate_001", "candidate_002"]
    assert [result.status for result in results] == [ResultStatus.SUCCESS, ResultStatus.SUCCESS]
    assert [result.metrics["score"] for result in results] == [3, 7]


def test_individual_keeps_concrete_stage_choices():
    from genio import Individual

    individual = Individual.from_slots(
        id="individual_001",
        scenario="simple_threshold_pipeline",
        genotype=(0, 0),
        search_index=0,
        slots=[
            StageChoice(slot=0, stage="bgr_to_gray"),
            StageChoice(
                slot=1,
                stage="threshold",
                parameters={"threshold": 120, "maxval": 255},
            ),
        ],
    )

    assert individual.stage_sequence() == ("bgr_to_gray", "threshold")
    assert individual.genotype == (0, 0)
    assert individual.search_index == 0
    assert individual.parameters_by_slot() == {
        0: {},
        1: {"threshold": 120, "maxval": 255},
    }


def test_search_space_maps_genotype_to_individual_and_index():
    search_space = SearchSpace(
        SearchScenarioSpec(
            id="simple_threshold_pipeline",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(
                        StageChoice(slot=0, stage="nop"),
                        StageChoice(slot=0, stage="bgr_to_gray"),
                    ),
                ),
                SlotSpec(
                    index=1,
                    alternatives=(
                        StageChoice(
                            slot=1,
                            stage="threshold",
                            parameters={"threshold": 80, "maxval": 255},
                        ),
                        StageChoice(
                            slot=1,
                            stage="threshold",
                            parameters={"threshold": 120, "maxval": 255},
                        ),
                        StageChoice(slot=1, stage="otsu_threshold"),
                    ),
                ),
            ),
        )
    )

    individual = search_space.from_genotype((1, 2), id="individual_001")

    assert search_space.slot_lengths == (2, 3)
    assert search_space.cardinality == 6
    assert individual.scenario == "simple_threshold_pipeline"
    assert individual.genotype == (1, 2)
    assert individual.search_index == 5
    assert individual.stage_sequence() == ("bgr_to_gray", "otsu_threshold")
    assert search_space.index_to_genotype(5) == (1, 2)
    assert search_space.genotype_to_index((1, 2)) == 5
    assert search_space.to_index(individual) == 5


def test_search_space_rebuilds_individual_from_slots():
    search_space = SearchSpace(
        SearchScenarioSpec(
            id="two_slot_space",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(
                        StageChoice(slot=0, stage="nop"),
                        StageChoice(slot=0, stage="resize"),
                    ),
                ),
                SlotSpec(
                    index=1,
                    alternatives=(
                        StageChoice(slot=1, stage="erode", parameters={"K_ROWS": 3}),
                        StageChoice(slot=1, stage="dilate", parameters={"K_ROWS": 3}),
                    ),
                ),
            ),
        )
    )

    individual = search_space.from_slots(
        [
            StageChoice(slot=0, stage="resize"),
            StageChoice(slot=1, stage="erode", parameters={"K_ROWS": 3}),
        ],
        id="child_001",
    )

    assert individual.genotype == (1, 0)
    assert individual.search_index == 2
    assert search_space.from_index(2, id="decoded").genotype == (1, 0)
