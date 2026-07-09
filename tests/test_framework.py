import sys
from pathlib import Path

from genio import (
    Artifact,
    Evaluation,
    EvaluationStep,
    EvaluationTask,
    ExecutionContext,
    EvaluationWorkflow,
    Individual,
    InMemoryStatistics,
    LocalBackend,
    MetricArtifact,
    OptimizationSession,
    Result,
    ResultStatus,
    SearchAlgorithm,
    SearchScenarioSpec,
    SearchSpace,
    SlotSpec,
    StatisticsCollector,
    StageChoice,
)


class MemoryArtifact(Artifact):
    def __init__(self, name: str, value, individual_id: str, producer: str = "test"):
        super().__init__(
            name=name,
            producer=producer,
            individual_id=individual_id,
            metadata={"value": value},
        )

    def load(self):
        return [self.metadata["value"]]


class MetricsMemoryArtifact(MetricArtifact):
    def __init__(self, name: str, values, individual_id: str, producer: str = "test"):
        super().__init__(
            name=name,
            producer=producer,
            individual_id=individual_id,
            metadata={"values": dict(values)},
        )

    def load(self):
        return [dict(self.metadata["values"])]

    def metrics(self):
        return self.metadata["values"]


class NamedTask(EvaluationTask):
    def __init__(self, individual: Individual, step_id: str, inputs=()) -> None:
        super().__init__(
            individual=individual,
            id=f"{individual.id}:{step_id}",
            step_id=step_id,
            metadata={"inputs": tuple(inputs)},
        )

    def run(self, context: ExecutionContext) -> list[Artifact]:
        return [
            MemoryArtifact(
                "output",
                {
                    "step_id": self.step_id,
                    "inputs": self.metadata["inputs"],
                    "base_work_dir_exists": context.base_work_dir.exists(),
                    "backend_id": context.backend_id,
                    "context_step_id": context.metadata["step_id"],
                },
                self.individual.id,
            )
        ]


class ScoreTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        score = self.individual.search_index or 0
        return [MetricsMemoryArtifact("score", {"score": score}, self.individual.id)]


class ScoreStep(EvaluationStep):
    id = "score"
    task_type = ScoreTask

    def create_task(self, individual: Individual, artifacts):
        return ScoreTask(individual=individual, step_id=self.id)


class NamedStep(EvaluationStep):
    task_type = NamedTask

    def __init__(self, id: str, depends_on=()) -> None:
        self.id = id
        self.depends_on = tuple(depends_on)

    def create_task(self, individual: Individual, artifacts):
        return NamedTask(individual, self.id, tuple(sorted(artifacts)))


class OneShotAlgorithm(SearchAlgorithm):
    def __init__(self) -> None:
        self._asked = False
        self._evaluations: list[Evaluation] = []

    def ask(self, session: OptimizationSession):
        self._asked = True
        return [session.search_space.from_index(1, id="individual_001")]

    def tell(self, evaluations):
        self._evaluations.extend(evaluations)

    def should_stop(self) -> bool:
        return bool(self._evaluations)

    def best_individuals(self):
        if not self._evaluations:
            return ()
        return [self._evaluations[0].individual]


class FirstIndividualAlgorithm(OneShotAlgorithm):
    def ask(self, session: OptimizationSession):
        self._asked = True
        return [session.search_space.from_index(0, id="individual_001")]


class TwoBatchAlgorithm(SearchAlgorithm):
    def __init__(self) -> None:
        self._next_index = 0
        self._evaluations: list[Evaluation] = []

    def ask(self, session: OptimizationSession):
        if self._next_index >= 2:
            return []
        individual = session.search_space.from_index(
            self._next_index,
            id=f"individual_{self._next_index}",
        )
        self._next_index += 1
        return [individual]

    def tell(self, evaluations):
        self._evaluations.extend(evaluations)

    def should_stop(self) -> bool:
        return len(self._evaluations) >= 2


class RecordingStatistics(StatisticsCollector):
    def __init__(self) -> None:
        self.events = []

    def on_batch_started(self, batch_index, individuals) -> None:
        self.events.append(
            (
                "batch_started",
                batch_index,
                tuple(individual.id for individual in individuals),
            )
        )

    def on_evaluation_completed(self, evaluation) -> None:
        self.events.append(
            (
                "evaluation_completed",
                evaluation.metadata["batch_index"],
                evaluation.individual.id,
            )
        )

    def on_batch_completed(self, batch_index, evaluations) -> None:
        self.events.append(
            (
                "batch_completed",
                batch_index,
                tuple(evaluation.individual.id for evaluation in evaluations),
            )
        )

    def snapshot(self):
        return {"events": list(self.events)}


def test_individual_keeps_concrete_stage_choices():
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
    search_space = SearchSpace.from_scenario(
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
    assert search_space.search_space_size == 6
    assert individual.scenario == "simple_threshold_pipeline"
    assert individual.genotype == (1, 2)
    assert individual.search_index == 5
    assert individual.stage_sequence() == ("bgr_to_gray", "otsu_threshold")
    assert search_space.index_to_genotype(5) == (1, 2)
    assert search_space.genotype_to_index((1, 2)) == 5
    assert search_space.to_index(individual) == 5


def test_search_space_rebuilds_individual_from_slots():
    search_space = SearchSpace.from_scenario(
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


def test_search_space_loads_real_test_file():
    root = Path(__file__).resolve().parents[1]
    search_space = SearchSpace(
        root / "search_space/tests/simple_threshold_pipeline.json",
        root / "search_space/stages/definitions",
    )

    assert search_space.scenario_id == "simple_threshold_pipeline"
    assert search_space.slot_lengths == (3, 19, 10)
    assert search_space.search_space_size == 570

    individual = search_space.from_genotype((1, 18, 0), id="loaded_001")

    assert individual.stage_sequence() == ("bgr_to_gray", "otsu_threshold", "nop")
    assert individual.search_index == search_space.genotype_to_index((1, 18, 0))


def test_optimization_session_coordinates_search_backend_and_statistics():
    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="one_slot_space",
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
    algorithm = FirstIndividualAlgorithm()
    statistics = InMemoryStatistics()

    result = OptimizationSession(
        id="threshold_optimization",
        search_space=search_space,
        algorithm=algorithm,
        backend=LocalBackend(),
        evaluation_workflow=EvaluationWorkflow((ScoreStep(),)),
        statistics=statistics,
    ).run()

    assert result.session_id == "threshold_optimization"
    assert result.statistics == {"evaluations": 1, "batches": 1}
    assert len(result.evaluations) == 1
    assert result.evaluations[0].individual.id == "individual_001"
    assert result.evaluations[0].metadata == {"batch_index": 0}
    assert result.evaluations[0].result.metrics == {"score.score": 0.0}
    assert result.best_individuals[0].id == "individual_001"


def test_optimization_session_notifies_batch_statistics_hooks():
    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="two_batch_space",
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
    statistics = RecordingStatistics()

    result = OptimizationSession(
        search_space=search_space,
        algorithm=TwoBatchAlgorithm(),
        backend=LocalBackend(),
        evaluation_workflow=EvaluationWorkflow((ScoreStep(),)),
        statistics=statistics,
    ).run()

    assert [evaluation.metadata["batch_index"] for evaluation in result.evaluations] == [0, 1]
    assert result.statistics == {
        "events": [
            ("batch_started", 0, ("individual_0",)),
            ("evaluation_completed", 0, "individual_0"),
            ("batch_completed", 0, ("individual_0",)),
            ("batch_started", 1, ("individual_1",)),
            ("evaluation_completed", 1, "individual_1"),
            ("batch_completed", 1, ("individual_1",)),
        ]
    }


def test_optimization_session_uses_evaluation_workflow_dependencies():
    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="workflow_space",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(StageChoice(slot=0, stage="threshold"),),
                ),
            ),
        )
    )
    algorithm = FirstIndividualAlgorithm()
    workflow = EvaluationWorkflow(
        (
            NamedStep("compile"),
            NamedStep("functional", depends_on=("compile",)),
        )
    )

    result = OptimizationSession(
        id="workflow_optimization",
        search_space=search_space,
        algorithm=algorithm,
        backend=LocalBackend(),
        evaluation_workflow=workflow,
    ).run()

    assert result.evaluations[0].result.metrics == {}


def test_local_backend_uses_configured_base_work_dir(tmp_path):
    class WorkDirTask(EvaluationTask):
        def run(self, context: ExecutionContext) -> list[Artifact]:
            task_dir = context.base_work_dir / self.individual.id / str(self.step_id)
            task_dir.mkdir(parents=True)
            return [MemoryArtifact("work_dir", task_dir, self.individual.id)]

    individual = Individual.from_slots(
        id="individual_001",
        scenario="work_dir_space",
        slots=[StageChoice(slot=0, stage="nop")],
    )
    backend = LocalBackend(base_work_dir=tmp_path)

    handle = backend.submit(WorkDirTask(individual=individual, step_id="prepare"))
    artifact = backend.collect(handle)[0]

    assert backend.status(handle).value == "done"
    assert artifact.load()[0] == tmp_path / "individual_001" / "prepare"
    assert artifact.load()[0].exists()


def test_local_backend_provides_runtime_context_metadata(tmp_path):
    class ContextTask(EvaluationTask):
        def run(self, context: ExecutionContext) -> list[Artifact]:
            return [
                MemoryArtifact(
                    "context",
                    {
                        "base_work_dir": context.base_work_dir,
                        "run_id": context.run_id,
                        "backend_id": context.backend_id,
                        "task_id": context.metadata["task_id"],
                        "individual_id": context.metadata["individual_id"],
                        "step_id": context.metadata["step_id"],
                        "custom": context.metadata["custom"],
                    },
                    self.individual.id,
                )
            ]

    individual = Individual.from_slots(
        id="individual_001",
        scenario="context_space",
        slots=[StageChoice(slot=0, stage="nop")],
    )
    backend = LocalBackend(
        base_work_dir=tmp_path,
        run_id="run_001",
        metadata={"custom": "value"},
    )

    task = ContextTask(individual=individual, step_id="context")
    handle = backend.submit(task)
    context_data = backend.collect(handle)[0].load()[0]

    assert handle.metadata == {"run_id": "run_001"}
    assert context_data == {
        "base_work_dir": tmp_path,
        "run_id": "run_001",
        "backend_id": "LocalBackend",
        "task_id": "individual_001:context",
        "individual_id": "individual_001",
        "step_id": "context",
        "custom": "value",
    }


def test_execution_context_filesystem_helpers(tmp_path):
    individual = Individual.from_slots(
        id="individual_001",
        scenario="helpers_space",
        slots=[StageChoice(slot=0, stage="nop")],
    )
    task = ScoreTask(individual=individual, step_id="helpers")
    context = ExecutionContext(base_work_dir=tmp_path)

    task_dir = context.task_dir(task)
    artifacts_dir = context.artifact_path(task)
    logs_dir = context.log_path(task)

    assert task_dir == tmp_path / "individual_001" / "helpers"
    assert artifacts_dir == task_dir / "artifacts"
    assert logs_dir == task_dir / "logs"

    context.ensure_dir(task_dir)
    text_path = context.write_text(task_dir / "config.txt", "hello")
    bytes_path = context.write_bytes(task_dir / "data.bin", b"abc")
    json_path = context.write_json(task_dir / "config.json", {"a": 1})
    log_path = context.write_log(task, "run.log", "log")

    assert context.read_text(text_path) == "hello"
    assert context.read_bytes(bytes_path) == b"abc"
    assert context.read_json(json_path) == {"a": 1}
    assert context.read_text(log_path) == "log"


def test_execution_context_copy_helpers(tmp_path):
    context = ExecutionContext(base_work_dir=tmp_path)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_file = source_dir / "template.txt"
    source_file.write_text("template", encoding="utf-8")

    copied_file = context.copy_file(source_file, "copied/template.txt")
    copied_tree = context.copy_tree(source_dir, "copied_tree")

    assert copied_file == tmp_path / "copied" / "template.txt"
    assert copied_file.read_text(encoding="utf-8") == "template"
    assert (copied_tree / "template.txt").read_text(encoding="utf-8") == "template"


def test_execution_context_run_command_helper(tmp_path):
    context = ExecutionContext(base_work_dir=tmp_path)

    result = context.run_command(
        [sys.executable, "-c", "import os; print(os.environ['GENIO_TEST'])"],
        env={"GENIO_TEST": "ok"},
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"
    assert result.stderr == ""
    assert result.command == (sys.executable, "-c", "import os; print(os.environ['GENIO_TEST'])")


def test_execution_context_run_command_can_skip_check(tmp_path):
    context = ExecutionContext(base_work_dir=tmp_path)

    result = context.run_command(
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        check=False,
    )

    assert result.returncode == 3


def test_evaluation_executor_rejects_step_task_type_mismatch():
    class MismatchedStep(EvaluationStep):
        id = "mismatch"
        task_type = NamedTask

        def create_task(self, individual: Individual, artifacts):
            return ScoreTask(individual=individual, step_id=self.id)

    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="mismatch_space",
            slots=(SlotSpec(index=0, alternatives=(StageChoice(slot=0, stage="nop"),)),),
        )
    )

    try:
        OptimizationSession(
            search_space=search_space,
            algorithm=FirstIndividualAlgorithm(),
            backend=LocalBackend(),
            evaluation_workflow=EvaluationWorkflow((MismatchedStep(),)),
        ).run()
    except Exception as exc:
        assert "declared task type NamedTask" in str(exc)
    else:
        raise AssertionError("Expected task type mismatch to fail")


def test_evaluation_executor_accumulates_metric_artifacts():
    class QualityTask(EvaluationTask):
        def run(self, context: ExecutionContext) -> list[Artifact]:
            return [
                MetricsMemoryArtifact(
                    "quality",
                    {"f1": 0.9, "latency": 12},
                    self.individual.id,
                )
            ]

    class QualityStep(EvaluationStep):
        id = "quality"
        task_type = QualityTask

        def create_task(self, individual: Individual, artifacts):
            return QualityTask(individual=individual, step_id=self.id)

    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="metric_space",
            slots=(SlotSpec(index=0, alternatives=(StageChoice(slot=0, stage="nop"),)),),
        )
    )

    result = OptimizationSession(
        search_space=search_space,
        algorithm=FirstIndividualAlgorithm(),
        backend=LocalBackend(),
        evaluation_workflow=EvaluationWorkflow((QualityStep(),)),
    ).run()

    assert result.evaluations[0].result.metrics == {
        "quality.f1": 0.9,
        "quality.latency": 12.0,
    }


def test_evaluation_executor_rejects_duplicate_artifact_keys():
    class DuplicateArtifactTask(EvaluationTask):
        def run(self, context: ExecutionContext) -> list[Artifact]:
            return [
                MemoryArtifact("output", 1, self.individual.id),
                MemoryArtifact("output", 2, self.individual.id),
            ]

    class DuplicateArtifactStep(EvaluationStep):
        id = "duplicate"
        task_type = DuplicateArtifactTask

        def create_task(self, individual: Individual, artifacts):
            return DuplicateArtifactTask(individual=individual, step_id=self.id)

    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="duplicate_artifact_space",
            slots=(SlotSpec(index=0, alternatives=(StageChoice(slot=0, stage="nop"),)),),
        )
    )

    try:
        OptimizationSession(
            search_space=search_space,
            algorithm=FirstIndividualAlgorithm(),
            backend=LocalBackend(),
            evaluation_workflow=EvaluationWorkflow((DuplicateArtifactStep(),)),
        ).run()
    except Exception as exc:
        assert "Duplicate artifact key 'duplicate.output'" in str(exc)
    else:
        raise AssertionError("Expected duplicate artifact key to fail")


def test_evaluation_executor_rejects_duplicate_metric_keys():
    class DuplicateMetricTask(EvaluationTask):
        def run(self, context: ExecutionContext) -> list[Artifact]:
            return [
                MetricsMemoryArtifact("quality_a", {"f1": 0.8}, self.individual.id),
                MetricsMemoryArtifact("quality_b", {"f1": 0.9}, self.individual.id),
            ]

    class DuplicateMetricStep(EvaluationStep):
        id = "quality"
        task_type = DuplicateMetricTask

        def create_task(self, individual: Individual, artifacts):
            return DuplicateMetricTask(individual=individual, step_id=self.id)

    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="duplicate_metric_space",
            slots=(SlotSpec(index=0, alternatives=(StageChoice(slot=0, stage="nop"),)),),
        )
    )

    try:
        OptimizationSession(
            search_space=search_space,
            algorithm=FirstIndividualAlgorithm(),
            backend=LocalBackend(),
            evaluation_workflow=EvaluationWorkflow((DuplicateMetricStep(),)),
        ).run()
    except Exception as exc:
        assert "Duplicate metric key 'quality.f1'" in str(exc)
    else:
        raise AssertionError("Expected duplicate metric key to fail")


def test_evaluation_executor_rejects_non_numeric_metrics():
    class InvalidMetricTask(EvaluationTask):
        def run(self, context: ExecutionContext) -> list[Artifact]:
            return [
                MetricsMemoryArtifact("quality", {"f1": "bad"}, self.individual.id),
            ]

    class InvalidMetricStep(EvaluationStep):
        id = "quality"
        task_type = InvalidMetricTask

        def create_task(self, individual: Individual, artifacts):
            return InvalidMetricTask(individual=individual, step_id=self.id)

    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="invalid_metric_space",
            slots=(SlotSpec(index=0, alternatives=(StageChoice(slot=0, stage="nop"),)),),
        )
    )

    try:
        OptimizationSession(
            search_space=search_space,
            algorithm=FirstIndividualAlgorithm(),
            backend=LocalBackend(),
            evaluation_workflow=EvaluationWorkflow((InvalidMetricStep(),)),
        ).run()
    except Exception as exc:
        assert "Metric 'quality.f1' must be numeric" in str(exc)
    else:
        raise AssertionError("Expected non-numeric metric to fail")
