from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from genio import Composer
from genio import ExecutionPackage
from genio import ExecutionContext
from genio import ImageFunctionalMetricsArtifact
from genio import PythonImageFunctionalEvaluationStep
from genio import PythonImageFunctionalTask
from genio import PythonExecutionPackage
from genio import PythonImagePipelineComposer
from genio import SearchScenarioSpec
from genio import SearchSpace
from genio import SlotSpec
from genio import StageChoice


ROOT = Path(__file__).resolve().parents[1]
TESTS_PATH = ROOT / "search_space/tests"
DEFINITIONS_PATH = ROOT / "search_space/stages/definitions"

TEST = "insect_segmentation_pipeline.json"

def test_python_image_functional_step_creates_task() -> None:
    """search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="image_functional_test",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(StageChoice(slot=0, stage="nop"),),
                ),
            ),
        )
    )"""

    search_space = SearchSpace(TESTS_PATH / TEST, DEFINITIONS_PATH)
    individual = search_space.from_index(0)
    step = PythonImageFunctionalEvaluationStep(
        images_path=Path("images"),
        references_path=Path("masks"),
        metrics=("mask_iou",),
    )

    task = step.create_task(individual, artifacts={})

    assert isinstance(task, PythonImageFunctionalTask)
    assert task.step_id == step.id
    assert task.images_path == Path("images")
    assert task.references_path == Path("masks")
    assert task.metrics == ("mask_iou",)


def test_image_functional_metrics_artifact_exposes_metrics() -> None:
    artifact = ImageFunctionalMetricsArtifact(
        name="image_functional_metrics",
        producer="python_image_functional",
        individual_id="individual_001",
        values={"iou": 0.75, "f1": 0.85},
        per_sample_values={
            "image_001.png": {"iou": 0.7, "f1": 0.82},
            "image_002.png": {"iou": 0.8, "f1": 0.88},
        },
    )

    assert artifact.metrics() == {"iou": 0.75, "f1": 0.85}
    assert artifact.load() == ({"iou": 0.75, "f1": 0.85},)


class DummyComposer(Composer):
    def compose(self, individual):
        return PythonExecutionPackage(
            entrypoint="pipeline.py:run",
            files={"pipeline.py": "def run(image):\n    return image\n"},
            requirements=("numpy",),
            metadata={"source": "dummy"},
        )


class InvalidExecutionPackage(ExecutionPackage):
    @property
    def entrypoint(self):
        return "invalid"

    @property
    def files(self):
        return {}

    def materialize(self, target_dir):
        return target_dir


class InvalidPackageComposer(Composer):
    def compose(self, individual):
        return InvalidExecutionPackage()


def make_individual():
    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="image_functional_test",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(StageChoice(slot=0, stage="nop"),),
                ),
            ),
        )
    )
    return search_space.from_index(0)


def test_python_image_functional_task_requires_composer(tmp_path) -> None:
    images_path = tmp_path / "images"
    images_path.mkdir()
    (images_path / "sample.png").write_bytes(b"")
    task = PythonImageFunctionalTask(
        individual=make_individual(),
        images_path=images_path,
    )

    with pytest.raises(ValueError, match="requires a composer"):
        task.run(ExecutionContext(base_work_dir=tmp_path))


def test_python_image_functional_task_requires_existing_images_path(tmp_path) -> None:
    task = PythonImageFunctionalTask(
        individual=make_individual(),
        composer=DummyComposer(tmp_path),
        images_path=tmp_path / "missing_images",
    )

    with pytest.raises(ValueError, match="images_path"):
        task.run(ExecutionContext(base_work_dir=tmp_path))


def test_python_image_functional_task_rejects_unknown_metrics(tmp_path) -> None:
    images_path = tmp_path / "images"
    references_path = tmp_path / "references"
    images_path.mkdir()
    references_path.mkdir()
    (images_path / "sample.png").write_bytes(b"")
    task = PythonImageFunctionalTask(
        individual=make_individual(),
        composer=DummyComposer(tmp_path),
        images_path=images_path,
        references_path=references_path,
        metrics=("unknown_metric",),
    )

    with pytest.raises(ValueError, match="Unsupported image functional metrics"):
        task.run(ExecutionContext(base_work_dir=tmp_path))


def test_python_image_functional_task_reaches_pending_runtime_after_validation(tmp_path) -> None:
    images_path = tmp_path / "images"
    references_path = tmp_path / "references"
    images_path.mkdir()
    references_path.mkdir()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert cv.imwrite(str(images_path / "sample.png"), image)
    assert cv.imwrite(str(references_path / "sample.png"), image)

    individual = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="image_functional_test",
            slots=(
                SlotSpec(index=0, alternatives=(StageChoice(slot=0, stage="bgr_to_gray"),)),
            ),
        )
    ).from_index(0)
    task = PythonImageFunctionalTask(
        individual=individual,
        composer=PythonImagePipelineComposer(DEFINITIONS_PATH),
        images_path=images_path,
        references_path=references_path,
        metrics=("mask_iou",),
    )

    artifacts = task.run(ExecutionContext(base_work_dir=tmp_path))

    package_dir = tmp_path / task.individual.id / "task" / "package"
    pipeline_source = (package_dir / "pipeline.py").read_text(encoding="utf-8")

    assert "def run(image):" in pipeline_source
    assert "cv.cvtColor(current, cv.COLOR_BGR2GRAY)" in pipeline_source
    assert "return current" in pipeline_source
    assert (package_dir / "requirements.txt").read_text(encoding="utf-8") == (
        "opencv-python\nnumpy\n"
    )
    assert (package_dir / "package_metadata.json").is_file()
    dataset_manifest = (package_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    assert "sample.png" in dataset_manifest
    output_path = tmp_path / task.individual.id / "task" / "artifacts" / "outputs" / "sample.png"
    assert output_path.is_file()
    execution_manifest = tmp_path / task.individual.id / "task" / "artifacts" / "execution_manifest.json"
    assert execution_manifest.is_file()
    assert len(artifacts) == 1
    assert artifacts[0].metrics() == {"mask_iou": 1.0}


def test_python_image_functional_task_computes_mask_and_instance_metrics(tmp_path) -> None:
    images_path = tmp_path / "images"
    references_path = tmp_path / "references"
    images_path.mkdir()
    references_path.mkdir()
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    mask[6:8, 6:8] = 255
    assert cv.imwrite(str(images_path / "sample.png"), mask)
    assert cv.imwrite(str(references_path / "sample.png"), mask)
    task = PythonImageFunctionalTask(
        individual=make_individual(),
        composer=DummyComposer(tmp_path),
        images_path=images_path,
        references_path=references_path,
        metrics=(
            "mask_iou",
            "mask_f1",
            "mask_precision",
            "mask_recall",
            "mask_accuracy",
            "mask_specificity",
            "mask_fpr",
            "mask_fnr",
            "mask_balanced_accuracy",
            "instance_precision",
            "instance_recall",
            "instance_f1",
            "mean_box_iou",
            "count_error",
        ),
    )

    artifacts = task.run(ExecutionContext(base_work_dir=tmp_path))
    metrics = artifacts[0].metrics()

    assert metrics == {
        "count_error": 0.0,
        "instance_f1": 1.0,
        "instance_precision": 1.0,
        "instance_recall": 1.0,
        "mask_accuracy": 1.0,
        "mask_balanced_accuracy": 1.0,
        "mask_f1": 1.0,
        "mask_fnr": 0.0,
        "mask_fpr": 0.0,
        "mask_iou": 1.0,
        "mask_precision": 1.0,
        "mask_recall": 1.0,
        "mask_specificity": 1.0,
        "mean_box_iou": 1.0,
    }


def test_python_image_functional_task_computes_pixel_error_rates(tmp_path) -> None:
    images_path = tmp_path / "images"
    references_path = tmp_path / "references"
    images_path.mkdir()
    references_path.mkdir()
    prediction = np.zeros((2, 2), dtype=np.uint8)
    reference = np.zeros((2, 2), dtype=np.uint8)
    prediction[0, 0] = 255
    prediction[0, 1] = 255
    reference[0, 0] = 255
    reference[1, 0] = 255
    assert cv.imwrite(str(images_path / "sample.png"), prediction)
    assert cv.imwrite(str(references_path / "sample.png"), reference)
    task = PythonImageFunctionalTask(
        individual=make_individual(),
        composer=DummyComposer(tmp_path),
        images_path=images_path,
        references_path=references_path,
        metrics=(
            "mask_accuracy",
            "mask_balanced_accuracy",
            "mask_fnr",
            "mask_fpr",
            "mask_iou",
            "mask_precision",
            "mask_recall",
            "mask_specificity",
        ),
    )

    metrics = task.run(ExecutionContext(base_work_dir=tmp_path))[0].metrics()

    assert metrics == {
        "mask_accuracy": 0.5,
        "mask_balanced_accuracy": 0.5,
        "mask_fnr": 0.5,
        "mask_fpr": 0.5,
        "mask_iou": 1.0 / 3.0,
        "mask_precision": 0.5,
        "mask_recall": 0.5,
        "mask_specificity": 0.5,
    }


def test_python_image_functional_task_reports_pipeline_failures(tmp_path) -> None:
    images_path = tmp_path / "images"
    images_path.mkdir()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert cv.imwrite(str(images_path / "sample.png"), image)

    class FailingComposer(Composer):
        def compose(self, individual):
            return PythonExecutionPackage(
                entrypoint="pipeline.py:run",
                files={"pipeline.py": "def run(image):\n    raise RuntimeError('boom')\n"},
            )

    task = PythonImageFunctionalTask(
        individual=make_individual(),
        composer=FailingComposer(tmp_path),
        images_path=images_path,
    )

    with pytest.raises(RuntimeError, match="failed for samples"):
        task.run(ExecutionContext(base_work_dir=tmp_path))


def test_python_image_functional_task_rejects_missing_references(tmp_path) -> None:
    images_path = tmp_path / "images"
    references_path = tmp_path / "references"
    images_path.mkdir()
    references_path.mkdir()
    (images_path / "sample.png").write_bytes(b"")
    task = PythonImageFunctionalTask(
        individual=make_individual(),
        composer=DummyComposer(tmp_path),
        images_path=images_path,
        references_path=references_path,
        metrics=("mask_iou",),
    )

    with pytest.raises(ValueError, match="Missing reference images"):
        task.run(ExecutionContext(base_work_dir=tmp_path))


def test_python_image_functional_task_requires_python_execution_package(tmp_path) -> None:
    images_path = tmp_path / "images"
    images_path.mkdir()
    (images_path / "sample.png").write_bytes(b"")
    task = PythonImageFunctionalTask(
        individual=make_individual(),
        composer=InvalidPackageComposer(tmp_path),
        images_path=images_path,
    )

    with pytest.raises(TypeError, match="PythonExecutionPackage"):
        task.run(ExecutionContext(base_work_dir=tmp_path))


def test_python_image_functional_task_uses_real_insect_segmentation_search_space(tmp_path) -> None:
    #images_path = tmp_path / "images"
    #references_path = tmp_path / "references"
    images_path = "/home/joselu/Universidad/Doctorado/Datasets/Olive_Fly/Images"
    references_path = "/home/joselu/Universidad/Doctorado/Datasets/Olive_Fly/Masks"
    #images_path.mkdir()
    #references_path.mkdir()
    #image = np.zeros((8, 8, 3), dtype=np.uint8)
    #assert cv.imwrite(str(images_path / "sample.png"), image)
    #assert cv.imwrite(str(references_path / "sample.png"), image)

    search_space = SearchSpace(TESTS_PATH / TEST, DEFINITIONS_PATH)
    individual = search_space.from_index(0)
    task = PythonImageFunctionalTask(
        individual=individual,
        composer=PythonImagePipelineComposer(DEFINITIONS_PATH),
        images_path=images_path,
        references_path=references_path,
        metrics=("count_error",
            "instance_f1",
            "instance_precision",
            "instance_recall",
            "mask_accuracy",
            "mask_balanced_accuracy",
            "mask_f1",
            "mask_fnr",
            "mask_fpr",
            "mask_iou",
            "mask_precision",
            "mask_recall",
            "mask_specificity",
            "mean_box_iou",),
    )

    assert search_space.scenario_id == "insect_segmentation_pipeline"
    assert individual.scenario == "insect_segmentation_pipeline"
    artifacts = task.run(ExecutionContext(base_work_dir=tmp_path))

    package_dir = tmp_path / individual.id / "task" / "package"
    assert (package_dir / "pipeline.py").is_file()
    assert (package_dir / "dataset_manifest.json").is_file()
    assert (
        tmp_path / individual.id / "task" / "artifacts" / "execution_manifest.json"
    ).is_file()
    assert artifacts[0].metrics()["mask_iou"] >= 0.0
