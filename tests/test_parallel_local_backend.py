from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which
from threading import Barrier, Event, Lock

import cv2 as cv
import numpy as np
import pytest

from genio import (
    Artifact,
    BackendError,
    BackendShutdownError,
    EvaluationExecutor,
    EvaluationHandle,
    EvaluationState,
    EvaluationStep,
    EvaluationTask,
    EvaluationWorkflow,
    ExecutionContext,
    GridSearch,
    HLSImagePipelineComposer,
    HLSImagePipelineSynthesisEvaluationStep,
    Individual,
    LocalBackend,
    LFUArtifactCache,
    MetricArtifact,
    OptimizationSession,
    ParallelLocalBackend,
    PythonImageFunctionalEvaluationStep,
    PythonImagePipelineComposer,
    ResultStatus,
    SearchScenarioSpec,
    SearchSpace,
    SlotSpec,
    StageChoice,
    UnknownEvaluationHandleError,
)


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS_PATH = ROOT / "search_space/stages/definitions"
HLS_TEMPLATES_PATH = ROOT / "hls_templates/vitis_vision_image_pipeline"
VITIS_LIBRARIES_PATH = ROOT / "Vitis_Libraries"
REQUIRES_VITIS = pytest.mark.skipif(
    which("v++") is None,
    reason="Vitis v++ is required for HLS integration tests.",
)
REQUIRES_VITIS_LIBRARIES = pytest.mark.skipif(
    not VITIS_LIBRARIES_PATH.is_dir(),
    reason="Vitis_Libraries is required for image-pipeline synthesis tests.",
)


@dataclass(frozen=True, slots=True)
class ValueArtifact(Artifact):
    value: object = None

    def load(self):
        return (self.value,)


@dataclass(frozen=True, slots=True)
class ValuesMetricArtifact(MetricArtifact):
    values: Mapping[str, float] = field(default_factory=dict)

    def load(self):
        return (self.values,)

    def metrics(self):
        return self.values


class ConcurrencyProbe:
    def __init__(self, parties: int) -> None:
        self.barrier = Barrier(parties)
        self.lock = Lock()
        self.active = 0
        self.max_active = 0

    def run(self) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.barrier.wait(timeout=5)
        finally:
            with self.lock:
                self.active -= 1


class ProbeTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        self.metadata["probe"].run()
        return []


class BlockingTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        self.metadata["started"].set()
        if not self.metadata["release"].wait(timeout=5):
            raise TimeoutError("Blocking task was not released.")
        return []


class FailingTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        raise ValueError("intentional task failure")


class WorkspaceTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        marker = context.write_text(context.task_dir(self, "marker.txt"), self.individual.id)
        return [
            ValueArtifact(
                name="workspace",
                producer="test",
                individual_id=self.individual.id,
                value=marker,
            )
        ]


class ProbeStep(EvaluationStep):
    id = "probe"
    task_type = ProbeTask

    def __init__(self, probe: ConcurrencyProbe) -> None:
        self.probe = probe

    def create_task(self, individual: Individual, artifacts):
        return ProbeTask(
            individual=individual,
            step_id=self.id,
            metadata={"probe": self.probe},
        )


class BaselineTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        return [
            ValuesMetricArtifact(
                name="metrics",
                producer="test",
                individual_id=self.individual.id,
                values={"score": 1.0},
            )
        ]


class BaselineStep(EvaluationStep):
    id = "baseline"
    task_type = BaselineTask

    def create_task(self, individual: Individual, artifacts):
        return BaselineTask(individual=individual, step_id=self.id)


class ConditionalTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        if self.individual.id == "bad":
            raise ValueError("candidate rejected")
        return [
            ValuesMetricArtifact(
                name="metrics",
                producer="test",
                individual_id=self.individual.id,
                values={"score": 2.0},
            )
        ]


class ConditionalStep(EvaluationStep):
    id = "quality"
    depends_on = ("baseline",)
    task_type = ConditionalTask

    def create_task(self, individual: Individual, artifacts):
        return ConditionalTask(individual=individual, step_id=self.id)


class RecordingTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        with self.metadata["lock"]:
            self.metadata["executed"].append(self.individual.id)
        return []


class RecordingStep(EvaluationStep):
    id = "downstream"
    depends_on = ("quality",)
    task_type = RecordingTask

    def __init__(self, executed: list[str], lock: Lock) -> None:
        self.executed = executed
        self.lock = lock

    def create_task(self, individual: Individual, artifacts):
        return RecordingTask(
            individual=individual,
            step_id=self.id,
            metadata={"executed": self.executed, "lock": self.lock},
        )


def make_individual(identifier: str) -> Individual:
    return Individual.from_slots(
        id=identifier,
        scenario="parallel_backend_test",
        slots=[StageChoice(slot=0, stage="nop")],
    )


def test_parallel_local_backend_runs_tasks_concurrently(tmp_path) -> None:
    probe = ConcurrencyProbe(parties=2)
    tasks = [
        ProbeTask(
            individual=make_individual(identifier),
            step_id="probe",
            metadata={"probe": probe},
        )
        for identifier in ("first", "second")
    ]

    with ParallelLocalBackend(max_workers=2, base_work_dir=tmp_path) as backend:
        handles = backend.submit_batch(tasks)
        assert len({handle.id for handle in handles}) == 2
        assert backend.collect_batch(handles) == [[], []]
        assert all(backend.status(handle) is EvaluationState.DONE for handle in handles)

    assert probe.max_active == 2


def test_parallel_local_backend_exposes_failure_and_original_exception(tmp_path) -> None:
    with ParallelLocalBackend(max_workers=1, base_work_dir=tmp_path) as backend:
        handle = backend.submit(FailingTask(individual=make_individual("failed"), step_id="fail"))

        with pytest.raises(ValueError, match="intentional task failure"):
            backend.collect(handle)

        assert backend.status(handle) is EvaluationState.FAILED
        assert backend.error(handle) == "ValueError: intentional task failure"


def test_local_backend_returns_failed_handle_and_raises_on_collect(tmp_path) -> None:
    backend = LocalBackend(base_work_dir=tmp_path)
    handle = backend.submit(FailingTask(individual=make_individual("failed"), step_id="fail"))

    assert backend.status(handle) is EvaluationState.FAILED
    assert backend.error(handle) == "ValueError: intentional task failure"
    with pytest.raises(ValueError, match="intentional task failure"):
        backend.collect(handle)


def test_parallel_local_backend_cancels_only_queued_tasks(tmp_path) -> None:
    started = Event()
    release = Event()
    backend = ParallelLocalBackend(max_workers=1, base_work_dir=tmp_path)
    running_handle = backend.submit(
        BlockingTask(
            individual=make_individual("running"),
            step_id="block",
            metadata={"started": started, "release": release},
        )
    )
    assert started.wait(timeout=2)
    queued_handle = backend.submit(
        BlockingTask(
            individual=make_individual("queued"),
            step_id="block",
            metadata={"started": Event(), "release": Event()},
        )
    )

    try:
        assert backend.status(running_handle) is EvaluationState.RUNNING
        assert backend.status(queued_handle) is EvaluationState.PENDING
        assert backend.cancel(running_handle) is False
        assert backend.cancel(queued_handle) is True
        assert backend.status(queued_handle) is EvaluationState.CANCELLED
        with pytest.raises(BackendError, match="cancelled"):
            backend.collect(queued_handle)
    finally:
        release.set()
        backend.shutdown()

    assert backend.status(running_handle) is EvaluationState.DONE


def test_parallel_local_backend_rejects_duplicate_active_workspace(tmp_path) -> None:
    started = Event()
    release = Event()
    task = BlockingTask(
        individual=make_individual("duplicate"),
        step_id="same",
        metadata={"started": started, "release": release},
    )
    backend = ParallelLocalBackend(max_workers=2, base_work_dir=tmp_path)
    first_handle = backend.submit(task)
    assert started.wait(timeout=2)

    try:
        with pytest.raises(BackendError, match="same workspace"):
            backend.submit(task)
    finally:
        release.set()
        backend.collect(first_handle)
        backend.shutdown()


def test_parallel_local_backend_isolates_workspaces(tmp_path) -> None:
    tasks = [
        WorkspaceTask(individual=make_individual(identifier), step_id="workspace")
        for identifier in ("first", "second")
    ]

    with ParallelLocalBackend(max_workers=2, base_work_dir=tmp_path) as backend:
        artifacts = backend.collect_batch(backend.submit_batch(tasks))

    paths = [task_artifacts[0].load()[0] for task_artifacts in artifacts]
    assert paths == [
        tmp_path / "first" / "workspace" / "marker.txt",
        tmp_path / "second" / "workspace" / "marker.txt",
    ]
    assert [path.read_text(encoding="utf-8") for path in paths] == ["first", "second"]


def test_parallel_local_backend_rejects_unknown_handle_and_submission_after_shutdown(
    tmp_path,
) -> None:
    backend = ParallelLocalBackend(max_workers=1, base_work_dir=tmp_path)
    with pytest.raises(UnknownEvaluationHandleError):
        backend.status(EvaluationHandle(id="unknown"))

    backend.shutdown()
    with pytest.raises(BackendShutdownError):
        backend.submit(WorkspaceTask(individual=make_individual("late"), step_id="workspace"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"max_workers": 0}, "max_workers"),
        ({"max_workers": 1, "max_pending": 0}, "max_pending"),
    ),
)
def test_parallel_local_backend_validates_capacity(tmp_path, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        ParallelLocalBackend(base_work_dir=tmp_path, **kwargs)


def test_evaluation_executor_uses_parallel_backend_for_batch(tmp_path) -> None:
    probe = ConcurrencyProbe(parties=2)
    workflow = EvaluationWorkflow((ProbeStep(probe),))
    individuals = (make_individual("first"), make_individual("second"))

    with ParallelLocalBackend(max_workers=2, base_work_dir=tmp_path) as backend:
        results = EvaluationExecutor(workflow, backend).evaluate_many(individuals)

    assert [result.individual_id for result in results] == ["first", "second"]
    assert all(result.status is ResultStatus.SUCCESS for result in results)
    assert probe.max_active == 2


def test_optimization_session_reaches_parallelism_through_algorithm_batch(tmp_path) -> None:
    probe = ConcurrencyProbe(parties=2)
    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="parallel_session",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(
                        StageChoice(slot=0, stage="nop"),
                        StageChoice(slot=0, stage="threshold"),
                    ),
                ),
            ),
        )
    )

    with ParallelLocalBackend(max_workers=2, base_work_dir=tmp_path) as backend:
        result = OptimizationSession(
            search_space=search_space,
            algorithm=GridSearch(max_evaluations=2, batch_size=2),
            backend=backend,
            evaluation_workflow=EvaluationWorkflow((ProbeStep(probe),)),
        ).run()

    assert len(result.evaluations) == 2
    assert all(
        evaluation.result.status is ResultStatus.SUCCESS
        for evaluation in result.evaluations
    )
    assert probe.max_active == 2


def test_parallel_backend_runs_isolated_python_functional_evaluations(tmp_path) -> None:
    images_path = tmp_path / "images"
    references_path = tmp_path / "references"
    images_path.mkdir()
    references_path.mkdir()
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    assert cv.imwrite(str(images_path / "sample.png"), mask)
    assert cv.imwrite(str(references_path / "sample.png"), mask)
    workflow = EvaluationWorkflow(
        (
            PythonImageFunctionalEvaluationStep(
                composer=PythonImagePipelineComposer(DEFINITIONS_PATH),
                images_path=images_path,
                references_path=references_path,
                metrics=("mask_iou",),
            ),
        )
    )
    individuals = (make_individual("first"), make_individual("second"))
    work_dir = tmp_path / "work"

    with ParallelLocalBackend(max_workers=2, base_work_dir=work_dir) as backend:
        results = EvaluationExecutor(workflow, backend).evaluate_many(individuals)

    assert [result.metrics for result in results] == [
        {"python_image_functional.mask_iou": 1.0},
        {"python_image_functional.mask_iou": 1.0},
    ]
    assert (
        work_dir
        / "first/python_image_functional/artifacts/outputs/sample.png"
    ).is_file()
    assert (
        work_dir
        / "second/python_image_functional/artifacts/outputs/sample.png"
    ).is_file()


def test_python_functional_cache_ignores_unrelated_design_domains(tmp_path) -> None:
    images_path = tmp_path / "images"
    references_path = tmp_path / "references"
    images_path.mkdir()
    references_path.mkdir()
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    assert cv.imwrite(str(images_path / "sample.png"), mask)
    assert cv.imwrite(str(references_path / "sample.png"), mask)
    workflow = EvaluationWorkflow(
        (
            PythonImageFunctionalEvaluationStep(
                composer=PythonImagePipelineComposer(DEFINITIONS_PATH),
                images_path=images_path,
                references_path=references_path,
                metrics=("mask_iou",),
            ),
        )
    )
    individuals = tuple(
        Individual.from_slots(
            id=identifier,
            scenario="cached_python",
            slots=[StageChoice(slot=0, stage="nop")],
            design={"hls": {"clock": clock}, "system": {"mcu": mcu}},
        )
        for identifier, clock, mcu in (
            ("source_python", 5, "small"),
            ("cached_python", 10, "large"),
        )
    )
    work_dir = tmp_path / "work"
    cache = LFUArtifactCache({"python_image_functional": 2})

    with ParallelLocalBackend(max_workers=2, base_work_dir=work_dir) as backend:
        results = EvaluationExecutor(workflow, backend, cache).evaluate_many(individuals)

    assert [result.metrics for result in results] == [
        {"python_image_functional.mask_iou": 1.0},
        {"python_image_functional.mask_iou": 1.0},
    ]
    assert [
        result.metadata["cache"]["python_image_functional"]["status"]
        for result in results
    ] == ["miss", "coalesced"]
    assert (
        work_dir
        / "source_python/python_image_functional/artifacts/outputs/sample.png"
    ).is_file()
    assert not (
        work_dir
        / "cached_python/python_image_functional/artifacts/outputs/sample.png"
    ).exists()


@pytest.mark.hls_integration
@pytest.mark.slow
@REQUIRES_VITIS
@REQUIRES_VITIS_LIBRARIES
def test_parallel_backend_runs_two_real_hls_syntheses(tmp_path) -> None:
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=HLS_TEMPLATES_PATH,
        rows=64,
        cols=128,
        image_type="XF_8UC1",
    )
    workflow = EvaluationWorkflow(
        (
            HLSImagePipelineSynthesisEvaluationStep(
                composer=composer,
                part="xa7a100tcsg324-1I",
            ),
        )
    )
    individuals = tuple(
        Individual.from_slots(
            id=identifier,
            scenario="parallel_hls",
            slots=[
                StageChoice(
                    slot=0,
                    stage="convert_scale_abs",
                    parameters={"alpha": 1.0, "beta": 0.0},
                )
            ],
        )
        for identifier in ("first_hls", "second_hls")
    )

    with ParallelLocalBackend(
        max_workers=2,
        base_work_dir=tmp_path,
        metadata={"vitis_libraries_path": str(VITIS_LIBRARIES_PATH.resolve())},
    ) as backend:
        results = EvaluationExecutor(workflow, backend).evaluate_many(individuals)

    assert all(result.status is ResultStatus.SUCCESS for result in results)
    assert all(
        "hls_image_pipeline_synthesis.hls_synthesis.lut" in result.metrics
        for result in results
    )
    assert (
        tmp_path
        / "first_hls/hls_image_pipeline_synthesis/package/work/hls/syn/verilog/top.v"
    ).is_file()
    assert (
        tmp_path
        / "second_hls/hls_image_pipeline_synthesis/package/work/hls/syn/verilog/top.v"
    ).is_file()


@pytest.mark.hls_integration
@pytest.mark.slow
@REQUIRES_VITIS
@REQUIRES_VITIS_LIBRARIES
def test_hls_artifact_cache_reuses_one_real_synthesis(tmp_path) -> None:
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=HLS_TEMPLATES_PATH,
        rows=64,
        cols=128,
        image_type="XF_8UC1",
    )
    workflow = EvaluationWorkflow(
        (
            HLSImagePipelineSynthesisEvaluationStep(
                composer=composer,
                part="xa7a100tcsg324-1I",
            ),
        )
    )
    individuals = tuple(
        Individual.from_slots(
            id=identifier,
            scenario="cached_hls",
            slots=[
                StageChoice(
                    slot=0,
                    stage="convert_scale_abs",
                    parameters={"alpha": 1.0, "beta": 0.0},
                )
            ],
            design={
                "hls": {"clock": 5, "flow_target": "vivado"},
                "system": {"mcu": mcu},
            },
        )
        for identifier, mcu in (("source_hls", "small"), ("cached_hls", "large"))
    )
    cache = LFUArtifactCache({"hls_image_pipeline_synthesis": 2})

    with ParallelLocalBackend(
        max_workers=2,
        base_work_dir=tmp_path,
        metadata={"vitis_libraries_path": str(VITIS_LIBRARIES_PATH.resolve())},
    ) as backend:
        results = EvaluationExecutor(workflow, backend, cache).evaluate_many(individuals)

    assert all(result.status is ResultStatus.SUCCESS for result in results)
    assert [
        result.metadata["cache"]["hls_image_pipeline_synthesis"]["status"]
        for result in results
    ] == ["miss", "coalesced"]
    assert results[0].metrics == results[1].metrics
    assert (
        tmp_path
        / "source_hls/hls_image_pipeline_synthesis/package/work/hls/syn/verilog/top.v"
    ).is_file()
    assert not (
        tmp_path
        / "cached_hls/hls_image_pipeline_synthesis/package/work/hls/syn/verilog/top.v"
    ).exists()
    assert cache.snapshot()["namespaces"]["hls_image_pipeline_synthesis"][
        "executions_avoided"
    ] == 1


def test_evaluation_executor_isolates_failures_and_skips_downstream_work(tmp_path) -> None:
    executed: list[str] = []
    lock = Lock()
    workflow = EvaluationWorkflow(
        (
            BaselineStep(),
            ConditionalStep(),
            RecordingStep(executed, lock),
        )
    )
    individuals = (make_individual("bad"), make_individual("good"))

    with ParallelLocalBackend(max_workers=2, base_work_dir=tmp_path) as backend:
        results = EvaluationExecutor(workflow, backend).evaluate_many(individuals)

    assert [result.individual_id for result in results] == ["bad", "good"]
    assert results[0].status is ResultStatus.FAILED
    assert results[0].metrics == {"baseline.score": 1.0}
    assert "ValueError: candidate rejected" in str(results[0].error)
    assert results[1].status is ResultStatus.SUCCESS
    assert results[1].metrics == {
        "baseline.score": 1.0,
        "quality.score": 2.0,
    }
    assert executed == ["good"]
