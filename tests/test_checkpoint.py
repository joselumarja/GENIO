from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random

import pytest

from genio import (
    Artifact,
    CheckpointCompatibilityError,
    CheckpointFormatError,
    CheckpointNotSupportedError,
    CheckpointPolicy,
    CheckpointStateError,
    CSVStatisticsCollector,
    EvaluationStep,
    EvaluationTask,
    EvaluationWorkflow,
    GeneticSearch,
    GridSearch,
    InMemoryStatistics,
    JSONCheckpointStore,
    LFUArtifactCache,
    LocalBackend,
    MetricArtifact,
    MetricObjective,
    OptimizationDirection,
    OptimizationSession,
    RandomSearch,
    SearchAlgorithm,
    SearchSpace,
    StageChoice,
)
from genio.search_space import SearchScenarioSpec, SlotSpec


class PlannedInterruption(RuntimeError):
    pass


EXECUTED_IDS: list[str] = []


@dataclass(frozen=True, slots=True)
class IndexMetricArtifact(MetricArtifact):
    value: float = 0.0

    def load(self):
        return ()

    def metrics(self):
        return {"score": self.value}


class IndexMetricTask(EvaluationTask):
    def run(self, context) -> list[Artifact]:
        EXECUTED_IDS.append(self.individual.id)
        return [
            IndexMetricArtifact(
                name="score",
                producer="score",
                individual_id=self.individual.id,
                value=float(self.individual.search_index or 0),
            )
        ]


class IndexMetricStep(EvaluationStep):
    id = "score"
    task_type = IndexMetricTask

    def checkpoint_signature(self):
        return EvaluationStep.checkpoint_signature(self)

    def create_task(self, individual, artifacts):
        return IndexMetricTask(individual=individual, step_id=self.id)


class UnsupportedAlgorithm(SearchAlgorithm):
    def ask(self, session):
        return ()

    def tell(self, evaluations):
        pass

    def should_stop(self):
        return True


def make_search_space(*, reverse_first_slot: bool = False) -> SearchSpace:
    first_alternatives = (
        StageChoice(slot=0, stage="a"),
        StageChoice(slot=0, stage="b"),
    )
    if reverse_first_slot:
        first_alternatives = tuple(reversed(first_alternatives))
    return SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="checkpoint_test",
            slots=(
                SlotSpec(index=0, alternatives=first_alternatives),
                SlotSpec(
                    index=1,
                    alternatives=(
                        StageChoice(slot=1, stage="c"),
                        StageChoice(slot=1, stage="d"),
                    ),
                ),
            ),
            design_spaces={"hls": {"npc": (1, 2)}},
        )
    )


def make_session(
    tmp_path: Path,
    *,
    algorithm: SearchAlgorithm,
    search_space: SearchSpace | None = None,
    run_id: str | None = "checkpoint-run",
    policy: CheckpointPolicy | None = None,
    statistics=None,
    artifact_cache=None,
) -> OptimizationSession:
    return OptimizationSession(
        search_space=search_space or make_search_space(),
        algorithm=algorithm,
        backend=LocalBackend(base_work_dir=tmp_path / "work"),
        evaluation_workflow=EvaluationWorkflow((IndexMetricStep(),)),
        statistics=statistics,
        id="checkpoint_test",
        run_id=run_id,
        checkpoint_policy=policy,
        artifact_cache=artifact_cache,
    )


def interrupt_after_running_checkpoint(
    session: OptimizationSession,
    *,
    completed_batches: int,
) -> None:
    store = session._checkpoint_store
    assert store is not None
    original_save = store.save

    def save_and_interrupt(payload, *, sequence):
        path = original_save(payload, sequence=sequence)
        if payload["status"] == "running" and sequence == completed_batches:
            raise PlannedInterruption("planned after committed checkpoint")
        return path

    store.save = save_and_interrupt


def result_signature(result):
    return [
        (
            evaluation.individual.id,
            evaluation.individual.genotype,
            evaluation.individual.search_index,
            evaluation.individual.metadata,
            evaluation.result.status.value,
            evaluation.result.metrics,
            evaluation.result.error,
            evaluation.result.metadata,
            evaluation.metadata,
        )
        for evaluation in result.evaluations
    ]


def test_json_checkpoint_store_roundtrip_latest_and_retention(tmp_path) -> None:
    policy = CheckpointPolicy(directory=tmp_path, keep_last=2)
    store = JSONCheckpointStore(policy)

    for sequence in range(1, 5):
        store.save({"status": "running", "value": sequence}, sequence=sequence)

    assert store.load()["value"] == 4
    snapshot_names = [
        path.name for path in sorted(tmp_path.glob("checkpoint-*.json"))
    ]
    assert len(snapshot_names) == 2
    assert snapshot_names[0].startswith("checkpoint-000003-running-")
    assert snapshot_names[1].startswith("checkpoint-000004-running-")
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert (tmp_path.stat().st_mode & 0o777) == 0o700
    assert all(
        (path.stat().st_mode & 0o777) == 0o600
        for path in tmp_path.glob("checkpoint-*.json")
    )


def test_json_checkpoint_store_rejects_corrupt_latest_snapshot(tmp_path) -> None:
    store = JSONCheckpointStore(CheckpointPolicy(directory=tmp_path))
    snapshot_path = store.save({"status": "running"}, sequence=1)
    snapshot_path.write_text("{}", encoding="utf-8")

    with pytest.raises(CheckpointFormatError, match="checksum"):
        store.load()


def test_json_checkpoint_store_preserves_previous_latest_if_publication_fails(
    tmp_path,
) -> None:
    store = JSONCheckpointStore(CheckpointPolicy(directory=tmp_path))
    first_path = store.save({"status": "running", "value": 1}, sequence=1)
    original_write = store._write_atomic

    def fail_latest(path, value):
        if path == store.latest_path:
            raise OSError("planned latest failure")
        original_write(path, value)

    store._write_atomic = fail_latest
    with pytest.raises(OSError, match="planned latest"):
        store.save({"status": "completed", "value": 2}, sequence=1)
    store._write_atomic = original_write

    assert first_path.exists()
    assert store.load()["value"] == 1
    assert store.load()["status"] == "running"


def test_json_checkpoint_store_verifies_direct_snapshot_content(tmp_path) -> None:
    store = JSONCheckpointStore(CheckpointPolicy(directory=tmp_path))
    snapshot_path = store.save({"status": "running", "value": 1}, sequence=1)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["value"] = 2
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(CheckpointFormatError, match="content checksum"):
        store.load(snapshot_path)


def test_json_checkpoint_store_rejects_latest_path_traversal(tmp_path) -> None:
    store = JSONCheckpointStore(CheckpointPolicy(directory=tmp_path))
    tmp_path.mkdir(parents=True, exist_ok=True)
    store.latest_path.write_text(
        json.dumps(
            {
                "format": store.LATEST_FORMAT,
                "schema_version": store.SCHEMA_VERSION,
                "checkpoint": "../outside.json",
                "sha256": "unused",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointFormatError, match="snapshot name"):
        store.load()


def test_json_checkpoint_store_supports_more_than_six_batch_digits(tmp_path) -> None:
    store = JSONCheckpointStore(CheckpointPolicy(directory=tmp_path))

    store.save({"status": "running", "value": 1}, sequence=1_000_000)

    assert store.load()["value"] == 1


def test_checkpoint_policy_validates_intervals(tmp_path) -> None:
    with pytest.raises(ValueError, match="every_batches"):
        CheckpointPolicy(directory=tmp_path, every_batches=0)
    with pytest.raises(ValueError, match="keep_last"):
        CheckpointPolicy(directory=tmp_path, keep_last=0)


def test_checkpointing_rejects_unsupported_algorithm_before_session_hooks(tmp_path) -> None:
    statistics = InMemoryStatistics()
    session = make_session(
        tmp_path,
        algorithm=UnsupportedAlgorithm(),
        policy=CheckpointPolicy(directory=tmp_path / "checkpoints"),
        statistics=statistics,
    )

    with pytest.raises(CheckpointNotSupportedError, match="UnsupportedAlgorithm"):
        session.run()

    assert statistics.snapshot() == {"evaluations": 0, "batches": 0}


def test_checkpointing_rejects_artifact_cache_until_it_is_persistable(tmp_path) -> None:
    session = make_session(
        tmp_path,
        algorithm=RandomSearch(max_evaluations=1),
        policy=CheckpointPolicy(directory=tmp_path / "checkpoints"),
        artifact_cache=LFUArtifactCache(default_capacity=1),
    )

    with pytest.raises(CheckpointNotSupportedError, match="artifact cache"):
        session.run()


def test_manual_checkpoint_is_rejected_before_start_and_during_ask(tmp_path) -> None:
    class SavingRandomSearch(RandomSearch):
        def ask(self, session):
            individuals = super().ask(session)
            session.save_checkpoint()
            return individuals

    checkpoint_dir = tmp_path / "checkpoints"
    session = make_session(
        tmp_path,
        algorithm=SavingRandomSearch(max_evaluations=1),
        policy=CheckpointPolicy(directory=checkpoint_dir),
    )
    with pytest.raises(CheckpointStateError, match="started"):
        session.save_checkpoint()
    with pytest.raises(CheckpointStateError, match="batch is in progress"):
        session.run()
    assert not (checkpoint_dir / "latest.json").exists()


def test_checkpoint_directory_has_single_session_owner(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    first = make_session(
        tmp_path / "first",
        algorithm=RandomSearch(max_evaluations=1),
        policy=CheckpointPolicy(directory=checkpoint_dir),
    )
    second = make_session(
        tmp_path / "second",
        algorithm=RandomSearch(max_evaluations=1),
        policy=CheckpointPolicy(directory=checkpoint_dir),
    )
    first._prepare_run()
    try:
        with pytest.raises(CheckpointFormatError, match="owned by another session"):
            second.run()
    finally:
        assert first._checkpoint_store is not None
        first._checkpoint_store.release_session_lease()


def test_random_session_resume_matches_uninterrupted_execution(tmp_path) -> None:
    EXECUTED_IDS.clear()
    reference = make_session(
        tmp_path / "reference",
        algorithm=RandomSearch(
            max_evaluations=6,
            batch_size=2,
            unique=False,
            random=Random(17),
        ),
    ).run()

    checkpoint_dir = tmp_path / "checkpoints"
    interrupted = make_session(
        tmp_path / "interrupted",
        algorithm=RandomSearch(
            max_evaluations=6,
            batch_size=2,
            unique=False,
            random=Random(17),
        ),
        policy=CheckpointPolicy(directory=checkpoint_dir, every_batches=1),
    )
    interrupt_after_running_checkpoint(interrupted, completed_batches=1)
    with pytest.raises(PlannedInterruption):
        interrupted.run()

    resumed = make_session(
        tmp_path / "resumed",
        algorithm=RandomSearch(
            max_evaluations=6,
            batch_size=2,
            unique=False,
            random=Random(999),
        ),
        run_id=None,
        policy=CheckpointPolicy(
            directory=checkpoint_dir,
            resume_from=checkpoint_dir / "latest.json",
        ),
    ).run()

    assert resumed.run_id == "checkpoint-run"
    assert result_signature(resumed) == result_signature(reference)
    assert resumed.statistics == {"evaluations": 6, "batches": 3}
    assert JSONCheckpointStore(
        CheckpointPolicy(directory=checkpoint_dir)
    ).load()["status"] == "completed"


def test_grid_session_resume_matches_uninterrupted_execution(tmp_path) -> None:
    reference = make_session(
        tmp_path / "reference",
        algorithm=GridSearch(max_evaluations=6, batch_size=2),
    ).run()
    checkpoint_dir = tmp_path / "checkpoints"
    interrupted = make_session(
        tmp_path / "interrupted",
        algorithm=GridSearch(max_evaluations=6, batch_size=2),
        policy=CheckpointPolicy(directory=checkpoint_dir),
    )
    interrupt_after_running_checkpoint(interrupted, completed_batches=1)
    with pytest.raises(PlannedInterruption):
        interrupted.run()

    resumed = make_session(
        tmp_path / "resumed",
        algorithm=GridSearch(max_evaluations=6, batch_size=2),
        run_id=None,
        policy=CheckpointPolicy(
            directory=checkpoint_dir,
            resume_from=checkpoint_dir / "latest.json",
        ),
    ).run()

    assert result_signature(resumed) == result_signature(reference)


def test_genetic_session_resume_restores_rng_lineage_and_best(tmp_path) -> None:
    initial_population = (
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
    )

    def algorithm(seed: int) -> GeneticSearch:
        return GeneticSearch(
            objectives=MetricObjective(
                "score.score",
                OptimizationDirection.MAXIMIZE,
            ),
            population_size=4,
            mutation_probability=0.5,
            max_generations=3,
            initial_population=initial_population,
            random=Random(seed),
        )

    reference_session = make_session(
        tmp_path / "reference",
        algorithm=algorithm(31),
    )
    reference = reference_session.run()

    checkpoint_dir = tmp_path / "genetic-checkpoints"
    interrupted = make_session(
        tmp_path / "interrupted",
        algorithm=algorithm(31),
        policy=CheckpointPolicy(directory=checkpoint_dir),
    )
    interrupt_after_running_checkpoint(interrupted, completed_batches=1)
    with pytest.raises(PlannedInterruption):
        interrupted.run()

    resumed_algorithm = algorithm(999)
    resumed = make_session(
        tmp_path / "resumed",
        algorithm=resumed_algorithm,
        run_id=None,
        policy=CheckpointPolicy(
            directory=checkpoint_dir,
            resume_from=checkpoint_dir / "latest.json",
        ),
    ).run()

    assert result_signature(resumed) == result_signature(reference)
    assert [item.id for item in resumed.best_individuals] == [
        item.id for item in reference.best_individuals
    ]
    assert resumed_algorithm.generation_fitnesses() == (
        reference_session.algorithm.generation_fitnesses()
    )


def test_genetic_search_rejects_checkpoint_with_pending_generation() -> None:
    search_space = make_search_space()
    algorithm = GeneticSearch(
        objectives=MetricObjective("score", OptimizationDirection.MAXIMIZE),
        population_size=4,
        max_generations=1,
    )
    algorithm.ask(type("Session", (), {"search_space": search_space})())

    with pytest.raises(CheckpointStateError, match="awaiting tell"):
        algorithm.checkpoint_state()


@pytest.mark.parametrize(
    "change",
    ("algorithm", "search_space", "run_id"),
)
def test_resume_rejects_incompatible_configuration_without_side_effects(
    tmp_path,
    change,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    interrupted = make_session(
        tmp_path / "interrupted",
        algorithm=RandomSearch(max_evaluations=4, batch_size=2, random=Random(4)),
        policy=CheckpointPolicy(directory=checkpoint_dir),
    )
    interrupt_after_running_checkpoint(interrupted, completed_batches=1)
    with pytest.raises(PlannedInterruption):
        interrupted.run()

    algorithm = RandomSearch(
        max_evaluations=6 if change == "algorithm" else 4,
        batch_size=2,
        random=Random(4),
    )
    search_space = make_search_space(reverse_first_slot=change == "search_space")
    statistics = InMemoryStatistics()
    resumed = make_session(
        tmp_path / "resumed",
        algorithm=algorithm,
        search_space=search_space,
        run_id="different-run" if change == "run_id" else None,
        policy=CheckpointPolicy(
            directory=checkpoint_dir,
            resume_from=checkpoint_dir / "latest.json",
        ),
        statistics=statistics,
    )

    with pytest.raises(CheckpointCompatibilityError):
        resumed.run()

    assert statistics.snapshot() == {"evaluations": 0, "batches": 0}


def test_completed_checkpoint_resume_executes_no_tasks(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    original = make_session(
        tmp_path / "original",
        algorithm=RandomSearch(max_evaluations=4, batch_size=2, random=Random(2)),
        policy=CheckpointPolicy(directory=checkpoint_dir),
    ).run()
    EXECUTED_IDS.clear()

    resumed = make_session(
        tmp_path / "resumed",
        algorithm=RandomSearch(max_evaluations=4, batch_size=2, random=Random(999)),
        run_id=None,
        policy=CheckpointPolicy(
            directory=checkpoint_dir,
            resume_from=checkpoint_dir / "latest.json",
        ),
    ).run()

    assert EXECUTED_IDS == []
    assert result_signature(resumed) == result_signature(original)
    assert resumed.statistics == original.statistics


def test_failed_completion_publication_can_be_retried(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    session = make_session(
        tmp_path,
        algorithm=RandomSearch(max_evaluations=2, batch_size=1, random=Random(2)),
        policy=CheckpointPolicy(directory=checkpoint_dir),
    )
    store = session._checkpoint_store
    assert store is not None
    original_save = store.save
    failed_once = False

    def fail_first_completion(payload, *, sequence):
        nonlocal failed_once
        if payload["status"] == "completed" and not failed_once:
            failed_once = True
            raise OSError("planned completion failure")
        return original_save(payload, sequence=sequence)

    store.save = fail_first_completion
    with pytest.raises(OSError, match="planned completion"):
        session.run()

    store.save = original_save
    result = session.run()

    assert len(result.evaluations) == 2
    assert store.load()["status"] == "completed"


def test_csv_statistics_resume_rebuilds_committed_rows(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "statistics"
    interrupted = make_session(
        tmp_path / "interrupted",
        algorithm=RandomSearch(max_evaluations=6, batch_size=2, random=Random(8)),
        policy=CheckpointPolicy(directory=checkpoint_dir),
        statistics=CSVStatisticsCollector(output_dir),
    )
    interrupt_after_running_checkpoint(interrupted, completed_batches=1)
    with pytest.raises(PlannedInterruption):
        interrupted.run()

    result = make_session(
        tmp_path / "resumed",
        algorithm=RandomSearch(max_evaluations=6, batch_size=2, random=Random(999)),
        run_id=None,
        policy=CheckpointPolicy(
            directory=checkpoint_dir,
            resume_from=checkpoint_dir / "latest.json",
        ),
        statistics=CSVStatisticsCollector(output_dir),
    ).run()

    with (output_dir / "individuals.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 6
    assert len({row["proposal_id"] for row in rows}) == 6
    assert result.statistics["evaluated_individuals"] == 6
    assert result.statistics["batches"] == 3


def test_completed_csv_checkpoint_resume_preserves_summary(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "statistics"
    original = make_session(
        tmp_path / "original",
        algorithm=RandomSearch(max_evaluations=2, batch_size=1, random=Random(3)),
        policy=CheckpointPolicy(directory=checkpoint_dir),
        statistics=CSVStatisticsCollector(output_dir),
    ).run()
    summary_path = output_dir / "run_summary.json"
    original_summary = summary_path.read_bytes()

    resumed = make_session(
        tmp_path / "resumed",
        algorithm=RandomSearch(max_evaluations=2, batch_size=1, random=Random(999)),
        run_id=None,
        policy=CheckpointPolicy(
            directory=checkpoint_dir,
            resume_from=checkpoint_dir / "latest.json",
        ),
        statistics=CSVStatisticsCollector(output_dir),
    ).run()

    assert summary_path.read_bytes() == original_summary
    assert result_signature(resumed) == result_signature(original)
