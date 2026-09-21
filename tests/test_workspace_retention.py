from __future__ import annotations

from pathlib import Path

import pytest

from genio import (
    EvaluationStep,
    EvaluationTask,
    EvaluationWorkflow,
    GridSearch,
    HLSRTLArtifact,
    Individual,
    LFUArtifactCache,
    LocalBackend,
    OptimizationSession,
    SearchScenarioSpec,
    SearchSpace,
    SlotSpec,
    StageChoice,
)


class FileArtifactTask(EvaluationTask):
    def cache_inputs(self):
        return {"semantic": "shared"}

    def run(self, context):
        self.metadata["calls"].append(self.individual.id)
        rtl_path = context.artifact_path(self, "generated.v")
        context.write_text(rtl_path, "module generated; endmodule\n")
        return [
            HLSRTLArtifact(
                name="rtl",
                producer=self.step_id,
                individual_id=self.individual.id,
                origin="test",
                top_function="generated",
                verilog_paths=(rtl_path,),
            )
        ]


class FileArtifactStep(EvaluationStep):
    id = "file_artifact"
    task_type = FileArtifactTask
    produced_artifacts = {"rtl": HLSRTLArtifact}

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def create_task(self, individual, artifacts):
        return FileArtifactTask(
            individual=individual,
            step_id=self.id,
            metadata={"calls": self.calls},
        )


def search_space(size: int) -> SearchSpace:
    return SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="workspace_retention",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=tuple(
                        StageChoice(slot=0, stage=f"stage-{index}")
                        for index in range(size)
                    ),
                ),
            ),
        )
    )


def run_session(
    tmp_path: Path,
    *,
    size: int,
    keep_all_individuals: bool,
    cache: LFUArtifactCache | None,
    calls: list[str],
):
    backend = LocalBackend(base_work_dir=tmp_path / "work")
    return OptimizationSession(
        search_space=search_space(size),
        algorithm=GridSearch(max_evaluations=size, batch_size=1),
        backend=backend,
        evaluation_workflow=EvaluationWorkflow((FileArtifactStep(calls),)),
        artifact_cache=cache,
        keep_all_individuals=keep_all_individuals,
    ).run(), backend


def test_session_cleans_completed_workspaces_but_cache_keeps_payloads(tmp_path) -> None:
    calls: list[str] = []
    cache = LFUArtifactCache(
        {"file_artifact": 1},
        storage_dir=tmp_path / "artifact_cache",
    )

    (result, backend) = run_session(
        tmp_path,
        size=2,
        keep_all_individuals=False,
        cache=cache,
        calls=calls,
    )

    assert len(result.evaluations) == 2
    assert len(calls) == 1
    assert all(
        not backend.base_work_dir.joinpath(evaluation.individual.id).exists()
        for evaluation in result.evaluations
    )
    key = cache.build_key("file_artifact", {"semantic": "shared"})
    entry = cache.get("file_artifact", key)
    assert entry is not None
    cached_artifact = entry.artifacts[0]
    assert isinstance(cached_artifact, HLSRTLArtifact)
    assert cached_artifact.verilog_paths[0].is_file()
    assert cached_artifact.verilog_paths[0].is_relative_to(cache.storage_dir)


def test_session_keeps_workspaces_when_requested(tmp_path) -> None:
    calls: list[str] = []

    (result, backend) = run_session(
        tmp_path,
        size=1,
        keep_all_individuals=True,
        cache=None,
        calls=calls,
    )

    workspace = backend.base_work_dir / result.evaluations[0].individual.id
    assert workspace.is_dir()
    assert (workspace / "file_artifact/artifacts/generated.v").is_file()


def test_session_rejects_non_boolean_workspace_retention(tmp_path) -> None:
    with pytest.raises(TypeError, match="keep_all_individuals"):
        OptimizationSession(
            search_space=search_space(1),
            algorithm=GridSearch(max_evaluations=1),
            backend=LocalBackend(base_work_dir=tmp_path),
            evaluation_workflow=EvaluationWorkflow(()),
            keep_all_individuals=1,  # type: ignore[arg-type]
        )


def test_session_rejects_cache_storage_inside_workspace(tmp_path) -> None:
    backend = LocalBackend(base_work_dir=tmp_path / "work")
    cache = LFUArtifactCache(
        {"file_artifact": 1},
        storage_dir=backend.base_work_dir / "cache",
    )

    with pytest.raises(ValueError, match="outside the backend workspace"):
        OptimizationSession(
            search_space=search_space(1),
            algorithm=GridSearch(max_evaluations=1),
            backend=backend,
            evaluation_workflow=EvaluationWorkflow(()),
            artifact_cache=cache,
        )


def test_execution_context_rejects_workspace_escape(tmp_path) -> None:
    task = FileArtifactTask(
        individual=Individual.from_slots(
            id="../escape",
            scenario="workspace_retention",
            slots=(),
        ),
        step_id="file_artifact",
        metadata={"calls": []},
    )
    context = LocalBackend(base_work_dir=tmp_path).create_context(task)

    with pytest.raises(ValueError, match="safe workspace path segment"):
        context.task_dir(task)
