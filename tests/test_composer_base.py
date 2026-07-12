from pathlib import Path

import pytest

from genio import (
    Composer,
    HLSExecutionPackage,
    Individual,
    PythonExecutionPackage,
    PythonImagePipelineComposer,
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


def test_python_execution_package_materializes_files(tmp_path):
    package = PythonExecutionPackage(
        entrypoint="pipeline.py:run",
        files={
            "pipeline.py": "def run(image):\n    return image\n",
            "config/settings.py": "VALUE = 1\n",
        },
        requirements=("numpy", "opencv-python"),
        metadata={"scenario": "test"},
    )

    package_dir = package.materialize(tmp_path / "package")

    assert package_dir == tmp_path / "package"
    assert (package_dir / "pipeline.py").read_text(encoding="utf-8") == (
        "def run(image):\n    return image\n"
    )
    assert (package_dir / "config/settings.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert (package_dir / "requirements.txt").read_text(encoding="utf-8") == (
        "numpy\nopencv-python\n"
    )


def test_hls_execution_package_materializes_files_and_metadata(tmp_path):
    package = HLSExecutionPackage(
        entrypoint="hls_config.cfg",
        files={
            "hls_config.cfg": "part=xczu9eg-ffvb1156-2-e\n\n[hls]\nsyn.top=image_pipeline_top\n",
            "src/pipeline.cpp": "void image_pipeline_top() {}\n",
            "include/pipeline.hpp": "#pragma once\n",
        },
        metadata={
            "package_type": "hls_image_pipeline",
            "top_function": "image_pipeline_top",
            "source_files": ["src/pipeline.cpp"],
        },
    )

    package_dir = package.materialize(tmp_path / "package")

    assert package_dir == tmp_path / "package"
    assert package.entrypoint == "hls_config.cfg"
    assert (package_dir / "hls_config.cfg").read_text(encoding="utf-8").startswith(
        "part=xczu9eg-ffvb1156-2-e"
    )
    assert (package_dir / "src/pipeline.cpp").read_text(encoding="utf-8") == (
        "void image_pipeline_top() {}\n"
    )
    assert '"top_function": "image_pipeline_top"' in (
        package_dir / "metadata.json"
    ).read_text(encoding="utf-8")


def test_python_image_pipeline_composer_generates_execution_package():
    composer = PythonImagePipelineComposer(DEFINITIONS_PATH)
    individual = Individual.from_slots(
        id="individual_001",
        scenario="simple_threshold_pipeline",
        genotype=(0, 1),
        search_index=1,
        slots=[
            StageChoice(slot=0, stage="bgr_to_gray"),
            StageChoice(
                slot=1,
                stage="threshold",
                parameters={
                    "threshold": 120,
                    "maxval": 255,
                    "threshold_type": "binary_inv",
                },
            ),
        ],
    )

    package = composer.compose(individual)
    source = package.files["pipeline.py"]

    assert package.entrypoint == "pipeline.py:run"
    assert package.requirements == ("opencv-python", "numpy")
    assert package.metadata["implementation_source"] == "opencv"
    assert "def run(image):" in source
    assert "stage_0_output = cv.cvtColor(current, cv.COLOR_BGR2GRAY)" in source
    assert (
        "_, stage_1_output = cv.threshold(current, 120, 255, cv.THRESH_BINARY_INV)"
        in source
    )
    assert "return current" in source


def test_python_image_pipeline_composer_maps_morphology_shape_enums():
    composer = PythonImagePipelineComposer(DEFINITIONS_PATH)
    individual = Individual.from_slots(
        id="individual_001",
        scenario="morphology_pipeline",
        slots=[
            StageChoice(
                slot=0,
                stage="erode",
                parameters={
                    "kernel_shape": "rect",
                    "kernel_rows": 3,
                    "kernel_cols": 3,
                    "iterations": 1,
                },
            ),
        ],
    )

    package = composer.compose(individual)
    source = package.files["pipeline.py"]

    assert "kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))" in source
    assert "'rect'" not in source
