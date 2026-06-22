from pathlib import Path
from random import Random

import pytest

from genio import SearchSpace, StageChoice


ROOT = Path(__file__).resolve().parents[1]
TESTS_PATH = ROOT / "search_space/tests"
DEFINITIONS_PATH = ROOT / "search_space/stages/definitions"


def test_search_space_expands_simple_threshold_pipeline():
    search_space = SearchSpace(
        TESTS_PATH / "simple_threshold_pipeline.json",
        DEFINITIONS_PATH,
    )

    assert search_space.scenario_id == "simple_threshold_pipeline"
    assert search_space.slot_lengths == (3, 19, 10)
    assert search_space.search_space_size == 570

    slot_0 = search_space.scenario.slots[0]
    slot_1 = search_space.scenario.slots[1]
    slot_2 = search_space.scenario.slots[2]

    assert [choice.stage for choice in slot_0.alternatives] == [
        "nop",
        "bgr_to_gray",
        "rgb_to_gray",
    ]
    assert slot_1.alternatives[0] == StageChoice(
        slot=1,
        stage="threshold",
        parameters={
            "threshold": 40,
            "maxval": 255,
            "threshold_type": "binary",
        },
    )
    assert slot_1.alternatives[17] == StageChoice(
        slot=1,
        stage="threshold",
        parameters={
            "threshold": 200,
            "maxval": 255,
            "threshold_type": "binary_inv",
        },
    )
    assert slot_1.alternatives[18] == StageChoice(
        slot=1,
        stage="otsu_threshold",
        parameters={"maxval": 255},
    )
    assert slot_2.alternatives[0] == StageChoice(slot=2, stage="nop")
    assert all(
        choice.parameters["kernel_rows"] == choice.parameters["kernel_cols"]
        for choice in slot_2.alternatives
        if choice.stage == "erode"
    )


def test_search_space_roundtrips_genotype_index_and_slots():
    search_space = SearchSpace(
        TESTS_PATH / "simple_threshold_pipeline.json",
        DEFINITIONS_PATH,
    )

    genotype = (2, 18, 9)
    individual = search_space.from_genotype(genotype, id="debug_001")

    assert individual.genotype == genotype
    assert individual.search_index == 569
    assert search_space.index_to_genotype(individual.search_index) == genotype
    assert search_space.to_genotype(individual) == genotype
    assert search_space.to_index(individual) == individual.search_index

    rebuilt = search_space.from_slots(individual.slots, id="rebuilt_001")

    assert rebuilt.genotype == genotype
    assert rebuilt.search_index == individual.search_index
    assert rebuilt.slots == individual.slots


def test_search_space_rejects_invalid_genotype_and_index():
    search_space = SearchSpace(
        TESTS_PATH / "simple_threshold_pipeline.json",
        DEFINITIONS_PATH,
    )

    with pytest.raises(ValueError, match="Expected genotype length"):
        search_space.from_genotype((0, 0))

    with pytest.raises(ValueError, match="Gene 19 for slot 1 is out of range"):
        search_space.from_genotype((0, 19, 0))

    with pytest.raises(ValueError, match="search_index 570 is out of range"):
        search_space.from_index(570)


def test_search_space_constraint_evaluator_supports_membership():
    assert SearchSpace._evaluate_constraint(
        "@mode in ['nearest', 'linear']",
        {"@mode": "linear"},
    )
    assert SearchSpace._evaluate_constraint(
        "@mode not in ('nearest', 'linear')",
        {"@mode": "area"},
    )
    assert SearchSpace._evaluate_constraint(
        "@value in {1, 3, 5}",
        {"@value": 3},
    )
    assert not SearchSpace._evaluate_constraint(
        "@mode in ['nearest', 'linear']",
        {"@mode": "area"},
    )


def test_search_space_constraint_evaluator_supports_safe_functions():
    assert SearchSpace._evaluate_constraint(
        "abs(@a - @b) <= 2",
        {"@a": 10, "@b": 12},
    )
    assert SearchSpace._evaluate_constraint(
        "min(@rows, @cols) >= 128",
        {"@rows": 240, "@cols": 320},
    )
    assert SearchSpace._evaluate_constraint(
        "max(@rows, @cols) <= 640",
        {"@rows": 240, "@cols": 640},
    )
    assert SearchSpace._evaluate_constraint(
        "round(@scale) == 2",
        {"@scale": 2.1},
    )


def test_search_space_constraint_evaluator_supports_domain_functions():
    assert SearchSpace._evaluate_constraint("is_odd(@value)", {"@value": 5})
    assert not SearchSpace._evaluate_constraint("is_odd(@value)", {"@value": 4})

    assert SearchSpace._evaluate_constraint("is_even(@value)", {"@value": 4})
    assert not SearchSpace._evaluate_constraint("is_even(@value)", {"@value": 5})

    assert SearchSpace._evaluate_constraint(
        "is_power_of_two(@value)",
        {"@value": 8},
    )
    assert not SearchSpace._evaluate_constraint(
        "is_power_of_two(@value)",
        {"@value": 0},
    )
    assert not SearchSpace._evaluate_constraint(
        "is_power_of_two(@value)",
        {"@value": 12},
    )

    assert SearchSpace._evaluate_constraint(
        "divisible_by(@cols, 8)",
        {"@cols": 640},
    )
    assert not SearchSpace._evaluate_constraint(
        "divisible_by(@cols, 0)",
        {"@cols": 640},
    )
    assert not SearchSpace._evaluate_constraint(
        "divisible_by(@cols, 8)",
        {"@cols": 642},
    )

    assert SearchSpace._evaluate_constraint(
        "square(@rows, @cols)",
        {"@rows": 5, "@cols": 5},
    )
    assert not SearchSpace._evaluate_constraint(
        "square(@rows, @cols)",
        {"@rows": 5, "@cols": 7},
    )


def test_search_space_constraint_evaluator_rejects_unsupported_functions():
    with pytest.raises(ValueError, match="Unsupported constraint function"):
        SearchSpace._evaluate_constraint(
            "sum([@a, @b]) == 3",
            {"@a": 1, "@b": 2},
        )

    with pytest.raises(ValueError, match="Unsupported constraint function"):
        SearchSpace._evaluate_constraint(
            "within(@value, 0, 10)",
            {"@value": 5},
        )

    with pytest.raises(ValueError, match="Only direct function calls"):
        SearchSpace._evaluate_constraint(
            "@name.lower() == 'linear'",
            {"@name": "LINEAR"},
        )


def test_search_space_rejects_slots_not_in_space():
    search_space = SearchSpace(
        TESTS_PATH / "simple_threshold_pipeline.json",
        DEFINITIONS_PATH,
    )

    with pytest.raises(ValueError, match="not a valid alternative"):
        search_space.from_slots(
            [
                StageChoice(slot=0, stage="bgr_to_gray"),
                StageChoice(
                    slot=1,
                    stage="threshold",
                    parameters={
                        "threshold": 999,
                        "maxval": 255,
                        "threshold_type": "binary",
                    },
                ),
                StageChoice(slot=2, stage="nop"),
            ]
        )


def test_search_space_samples_unique_population_deterministically():
    search_space = SearchSpace(
        TESTS_PATH / "simple_threshold_pipeline.json",
        DEFINITIONS_PATH,
    )

    population = search_space.sample_population(
        10,
        unique=True,
        random=Random(7),
    )

    indexes = [individual.search_index for individual in population]

    assert len(population) == 10
    assert len(set(indexes)) == 10
    assert indexes == [236, 391, 415, 398, 11, 321, 28, 209, 79, 189]


def test_search_space_loads_all_current_test_files():
    expected_slot_lengths = {
        "color_threshold_pipeline.json": (2, 30),
        "geometric_transform_pipeline.json": (5,),
        "lut_and_model_pipeline.json": (2, 3, 1),
        "resize_filter_pipeline.json": (27, 12),
        "simple_threshold_pipeline.json": (3, 19, 10),
    }

    for test_file_name, slot_lengths in expected_slot_lengths.items():
        search_space = SearchSpace(TESTS_PATH / test_file_name, DEFINITIONS_PATH)

        assert search_space.slot_lengths == slot_lengths
        assert search_space.search_space_size > 0
        assert search_space.from_index(0).genotype == tuple(0 for _ in slot_lengths)
