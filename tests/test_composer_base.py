from pathlib import Path

import pytest

from genio import (
    Composer,
    Individual,
    StageChoice,
    StageDefinitionNotFoundError,
)


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS_PATH = ROOT / "search_space/stages/definitions"


class DummyComposer(Composer):
    def compose(self, individual: Individual) -> dict:
        return {
            "metadata": self.artifact_metadata(individual),
            "stages": [
                {
                    "slot": choice.slot,
                    "stage": choice.stage,
                    "parameters": dict(choice.parameters),
                    "definition_id": definition["id"],
                }
                for choice, definition in self.active_stage_definitions(individual)
            ],
        }


def test_base_composer_skips_nop_and_composes_non_nop_steps():
    composer = DummyComposer(DEFINITIONS_PATH)
    individual = Individual.from_slots(
        id="individual_001",
        scenario="simple_threshold_pipeline",
        genotype=(0, 1),
        search_index=1,
        slots=[
            StageChoice(slot=0, stage="nop"),
            StageChoice(
                slot=1,
                stage="threshold",
                parameters={
                    "threshold": 120,
                    "maxval": 255,
                    "threshold_type": "binary",
                },
            ),
        ],
    )

    artifact = composer.compose(individual)

    assert artifact["metadata"] == {
        "composer": "DummyComposer",
        "scenario": "simple_threshold_pipeline",
        "search_index": 1,
    }
    assert artifact["stages"] == [
        {
            "slot": 1,
            "stage": "threshold",
            "parameters": {
                "threshold": 120,
                "maxval": 255,
                "threshold_type": "binary",
            },
            "definition_id": "threshold",
        }
    ]


def test_base_composer_exposes_active_choices_without_nop():
    composer = DummyComposer(DEFINITIONS_PATH)
    individual = Individual.from_slots(
        id="individual_001",
        scenario="simple_threshold_pipeline",
        slots=[
            StageChoice(slot=0, stage="nop"),
            StageChoice(
                slot=1,
                stage="threshold",
                parameters={
                    "threshold": 120,
                    "maxval": 255,
                    "threshold_type": "binary",
                },
            ),
        ],
    )

    assert list(composer.active_choices(individual)) == [
        StageChoice(
            slot=1,
            stage="threshold",
            parameters={
                "threshold": 120,
                "maxval": 255,
                "threshold_type": "binary",
            },
        )
    ]


def test_base_composer_exposes_active_stage_definitions():
    composer = DummyComposer(DEFINITIONS_PATH)
    individual = Individual.from_slots(
        id="individual_001",
        scenario="simple_threshold_pipeline",
        slots=[
            StageChoice(
                slot=0,
                stage="threshold",
                parameters={
                    "threshold": 120,
                    "maxval": 255,
                    "threshold_type": "binary",
                },
            ),
        ],
    )

    active = list(composer.active_stage_definitions(individual))

    assert active[0][0] == StageChoice(
        slot=0,
        stage="threshold",
        parameters={
            "threshold": 120,
            "maxval": 255,
            "threshold_type": "binary",
        },
    )
    assert active[0][1]["id"] == "threshold"


def test_base_composer_raises_for_unknown_stage():
    composer = DummyComposer(DEFINITIONS_PATH)
    individual = Individual.from_slots(
        id="individual_001",
        scenario="unknown_stage_pipeline",
        slots=[StageChoice(slot=0, stage="missing_stage")],
    )

    with pytest.raises(StageDefinitionNotFoundError, match="missing_stage"):
        composer.compose(individual)
