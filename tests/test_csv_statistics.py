import csv
import json

from genio import (
    CSVStatisticsCollector,
    EvaluationStep,
    EvaluationTask,
    EvaluationWorkflow,
    ExecutionContext,
    ImageFunctionalMetricsArtifact,
    Individual,
    LFUArtifactCache,
    OptimizationSession,
    ParallelLocalBackend,
    ResultStatus,
    SearchAlgorithm,
    SearchScenarioSpec,
    SearchSpace,
    SlotSpec,
    StageChoice,
)


class StatisticsTask(EvaluationTask):
    def run(self, context: ExecutionContext):
        if self.individual.metadata.get("fail"):
            raise RuntimeError("intentional statistics failure")
        return [
            ImageFunctionalMetricsArtifact(
                name="metrics",
                producer="statistics_test",
                individual_id=self.individual.id,
                values={"score": float(self.individual.metadata["score"])},
            )
        ]


class StatisticsStep(EvaluationStep):
    id = "score"
    task_type = StatisticsTask

    def create_task(self, individual, artifacts):
        return StatisticsTask(individual=individual, step_id=self.id)


class TwoProposalAlgorithm(SearchAlgorithm):
    def __init__(self) -> None:
        self._asked = False
        self._evaluations = []

    def ask(self, session):
        if self._asked:
            return ()
        self._asked = True
        return tuple(
            Individual.from_slots(
                id=identifier,
                scenario="statistics_space",
                slots=[StageChoice(slot=0, stage="nop")],
                genotype=(0,),
                search_index=0,
                design={"hls": {"npc": "XF_NPPC1"}},
                metadata={
                    "score": score,
                    "fail": fail,
                    "algorithm": {
                        "generation": 3,
                        "population_index": position,
                        "proposal_origin": "initialization",
                        "parent_ids": [],
                    },
                },
            )
            for position, (identifier, score, fail) in enumerate(
                (("first", 0.9, False), ("second", 0.0, True))
            )
        )

    def tell(self, evaluations):
        self._evaluations.extend(evaluations)

    def should_stop(self):
        return bool(self._evaluations)

    def best_individuals(self):
        return (self._evaluations[0].individual,) if self._evaluations else ()


def make_search_space() -> SearchSpace:
    return SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="statistics_space",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(StageChoice(slot=0, stage="nop"),),
                ),
            ),
        )
    )


def test_csv_statistics_records_every_proposal_and_run_summary(tmp_path) -> None:
    statistics_dir = tmp_path / "statistics"
    collector = CSVStatisticsCollector(statistics_dir)
    with ParallelLocalBackend(
        max_workers=2,
        base_work_dir=tmp_path / "work",
    ) as backend:
        result = OptimizationSession(
            id="statistics_session",
            run_id="run_statistics",
            search_space=make_search_space(),
            algorithm=TwoProposalAlgorithm(),
            backend=backend,
            evaluation_workflow=EvaluationWorkflow((StatisticsStep(),)),
            statistics=collector,
            metadata={"experiment": "csv"},
        ).run()

    with (statistics_dir / "individuals.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert [row["proposal_id"] for row in rows] == [
        "run_statistics:000000",
        "run_statistics:000001",
    ]
    assert [row["proposal_sequence"] for row in rows] == ["0", "1"]
    assert [row["batch_index"] for row in rows] == ["0", "0"]
    assert [row["batch_position"] for row in rows] == ["0", "1"]
    assert [row["generation"] for row in rows] == ["3", "3"]
    assert [row["population_index"] for row in rows] == ["0", "1"]
    assert [row["duplicate_genotype_seen"] for row in rows] == ["False", "True"]
    assert [row["evaluation_status"] for row in rows] == ["success", "failed"]
    assert rows[0]["metric.score.score"] == "0.9"
    assert rows[1]["metric.score.score"] == ""
    assert rows[1]["error_type"] == "RuntimeError"
    assert rows[1]["error_message"] == "intentional statistics failure"
    assert rows[0]["design.hls.npc"] == "XF_NPPC1"
    assert rows[0]["slot.000.stage"] == "nop"
    assert json.loads(rows[0]["genotype_json"]) == [0]
    assert json.loads(rows[0]["algorithm.parent_ids"]) == []

    summary = json.loads((statistics_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == "run_statistics"
    assert summary["generated_individuals"] == 2
    assert summary["evaluated_individuals"] == 2
    assert summary["status_counts"] == {"failed": 1, "success": 1}
    assert summary["duplicate_genotypes"] == 1
    assert summary["best_individual_ids"] == ["first"]
    assert summary["metric_summary"]["score.score"] == {
        "count": 1,
        "min": 0.9,
        "max": 0.9,
        "mean": 0.9,
    }

    manifest = json.loads((statistics_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run_statistics"
    assert manifest["session_id"] == "statistics_session"
    assert manifest["session_metadata"] == {"experiment": "csv"}
    assert manifest["workflow_steps"] == [
        {
            "depends_on": [],
            "id": "score",
            "type": f"{StatisticsStep.__module__}.{StatisticsStep.__qualname__}",
        }
    ]

    assert result.run_id == "run_statistics"
    assert [evaluation.result.status for evaluation in result.evaluations] == [
        ResultStatus.SUCCESS,
        ResultStatus.FAILED,
    ]
    assert [evaluation.metadata for evaluation in result.evaluations] == [
        {
            "proposal_id": "run_statistics:000000",
            "proposal_sequence": 0,
            "batch_index": 0,
            "batch_position": 0,
        },
        {
            "proposal_id": "run_statistics:000001",
            "proposal_sequence": 1,
            "batch_index": 0,
            "batch_position": 1,
        },
    ]
    assert result.statistics == {
        "run_id": "run_statistics",
        "generated_individuals": 2,
        "evaluated_individuals": 2,
        "batches": 1,
        "individuals_csv": str(statistics_dir / "individuals.csv"),
        "run_manifest": str(statistics_dir / "run_manifest.json"),
        "run_summary": str(statistics_dir / "run_summary.json"),
        "cache": None,
    }


class CachedStatisticsTask(EvaluationTask):
    def cache_inputs(self):
        return {"pipeline": tuple(choice.stage for choice in self.individual.slots)}

    def run(self, context: ExecutionContext):
        self.metadata["calls"].append(self.individual.id)
        return [
            ImageFunctionalMetricsArtifact(
                name="metrics",
                producer="statistics_test",
                individual_id=self.individual.id,
                values={"score": 1.0},
            )
        ]


class CachedStatisticsStep(EvaluationStep):
    id = "cached"
    task_type = CachedStatisticsTask

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def create_task(self, individual, artifacts):
        return CachedStatisticsTask(
            individual=individual,
            step_id=self.id,
            metadata={"calls": self.calls},
        )


def test_csv_statistics_records_cache_hits_and_summary(tmp_path) -> None:
    statistics_dir = tmp_path / "statistics"
    collector = CSVStatisticsCollector(statistics_dir)
    calls: list[str] = []
    cache = LFUArtifactCache({"cached": 2})

    with ParallelLocalBackend(
        max_workers=2,
        base_work_dir=tmp_path / "work",
    ) as backend:
        OptimizationSession(
            id="cache_statistics_session",
            run_id="run_cache_statistics",
            search_space=make_search_space(),
            algorithm=TwoProposalAlgorithm(),
            backend=backend,
            evaluation_workflow=EvaluationWorkflow((CachedStatisticsStep(calls),)),
            statistics=collector,
            artifact_cache=cache,
        ).run()

    with (statistics_dir / "individuals.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert calls == ["first"]
    assert [row["cache_hit"] for row in rows] == ["False", "True"]
    assert [row["cache.cached.status"] for row in rows] == ["miss", "coalesced"]
    assert rows[1]["cache.cached.cache_source_individual_id"] == "first"
    summary = json.loads((statistics_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["cache"]["totals"] == {
        "bypasses": 0,
        "coalesced": 1,
        "entries": 1,
        "evictions": 0,
        "executions_avoided": 1,
        "hit_rate": 0.5,
        "hits": 0,
        "misses": 1,
        "stores": 1,
    }
