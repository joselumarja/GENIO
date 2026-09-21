from __future__ import annotations

import json
import csv
import shutil

import pytest

from genio import (
    Evaluation,
    CheckpointPolicy,
    EvaluationWorkflow,
    GridSearch,
    Individual,
    LocalBackend,
    OptimizationSession,
    PopulationAnalysisCollector,
    PopulationPlotConfig,
    PopulationStatisticsCollector,
    Proposal,
    Result,
    SearchAlgorithm,
    SearchScenarioSpec,
    SearchSpace,
    SlotSpec,
    StageChoice,
)
from genio.objective import (
    EvaluatedBatch,
    EvaluatedIndividual,
    ObjectiveEvaluationStatus,
    ObjectiveValues,
)
from genio.statistics.population import PopulationAnalysisError


def individual(identifier: str, gene: int) -> Individual:
    return Individual.from_slots(
        id=identifier,
        scenario="population_statistics",
        slots=(StageChoice(slot=0, stage=f"stage-{gene}", parameters={"value": gene}),),
        genotype=(gene,),
        search_index=gene,
        metadata={"algorithm": {"generation": 0, "population_index": gene}},
    )


def proposal(candidate: Individual, position: int) -> Proposal:
    return Proposal(
        proposal_id=f"run:{position:06d}",
        proposal_sequence=position,
        batch_index=0,
        batch_position=position,
        individual=candidate,
    )


def evaluated(candidate_proposal: Proposal, score: float) -> EvaluatedIndividual:
    evaluation = Evaluation(
        individual=candidate_proposal.individual,
        result=Result.success(candidate_proposal.individual.id, {"metric.score": score}),
        metadata=candidate_proposal.evaluation_metadata(),
    )
    return EvaluatedIndividual(
        evaluation=evaluation,
        objective_values=ObjectiveValues(
            names=("score",),
            raw=(score,),
            minimize=(-score,),
            maximize=(score,),
            aggregate_score=score,
        ),
        status=ObjectiveEvaluationStatus.VALID,
    )


def test_population_collector_correlates_objectives_and_closes_snapshot(tmp_path) -> None:
    collector = PopulationAnalysisCollector(tmp_path / "analysis")
    candidates = (individual("first", 0), individual("second", 1))
    proposals = tuple(
        proposal(candidate, position)
        for position, candidate in enumerate(candidates)
    )
    items = (evaluated(proposals[0], 0.2), evaluated(proposals[1], 0.8))

    collector.on_session_started(object())  # type: ignore[arg-type]
    collector.on_batch_started(0, candidates)
    collector.on_proposals_generated(proposals)
    collector.on_evaluated_batch(
        EvaluatedBatch(tuple(reversed(items)), ("score",), batch_index=0)
    )
    collector.on_batch_completed(0, tuple(item.evaluation for item in items))
    collector.on_session_completed(
        type("ResultView", (), {"best_individuals": (candidates[1],)})()
    )  # type: ignore[arg-type]

    assert len(collector.snapshots) == 1
    assert [record.individual_id for record in collector.records] == ["first", "second"]
    assert [record.aggregate_score for record in collector.records] == [0.2, 0.8]
    assert collector.best_individual_ids == ("second",)
    assert collector.snapshot() == {
        "output_dir": str(tmp_path / "analysis"),
        "batches": 1,
        "records": 2,
        "evaluated_records": 2,
        "valid_objective_records": 2,
        "best_individual_ids": ["second"],
        "analysis_summary": str(tmp_path / "analysis/analysis_summary.json"),
        "analysis_manifest": str(tmp_path / "analysis/analysis_manifest.json"),
        "generated_plots": 11,
        "plot_warnings": 0,
    }
    summary = json.loads(collector.summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(collector.manifest_path.read_text(encoding="utf-8"))
    assert summary["records"] == 2
    assert summary["best_individual_ids"] == ["second"]
    assert len(manifest["generated"]) == 11
    assert manifest["skipped"]["final"] == {
        "objective_scatter": "Exactly two raw objectives are required."
    }


def test_population_collector_rejects_unknown_or_incomplete_batches(tmp_path) -> None:
    collector = PopulationAnalysisCollector(tmp_path)
    candidate = individual("first", 0)
    candidate_proposal = proposal(candidate, 0)
    item = evaluated(candidate_proposal, 0.5)

    with pytest.raises(PopulationAnalysisError, match="was not proposed"):
        collector.on_evaluated_batch(
            EvaluatedBatch((item,), ("score",), batch_index=0)
        )

    collector.on_batch_started(0, (candidate,))
    collector.on_proposals_generated((candidate_proposal,))
    with pytest.raises(PopulationAnalysisError, match="incomplete"):
        collector.on_batch_completed(0, (item.evaluation,))


class OnePopulationAlgorithm(SearchAlgorithm):
    def __init__(self) -> None:
        self._asked = False
        self._evaluations: tuple[Evaluation, ...] = ()

    def ask(self):
        if self._asked:
            return ()
        self._asked = True
        return (
            self.context.search_space.from_index(0, id="session-individual"),
        )

    def tell(self, batch):
        self._evaluations = batch.evaluations

    def should_stop(self):
        return bool(self._evaluations)

    def best_individuals(self):
        return tuple(evaluation.individual for evaluation in self._evaluations)


def test_population_collector_integrates_with_optimization_session(tmp_path) -> None:
    collector = PopulationAnalysisCollector(tmp_path / "analysis")
    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="population_statistics",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(StageChoice(slot=0, stage="stage-0"),),
                ),
            ),
        )
    )

    result = OptimizationSession(
        search_space=search_space,
        algorithm=OnePopulationAlgorithm(),
        backend=LocalBackend(base_work_dir=tmp_path / "work"),
        evaluation_workflow=EvaluationWorkflow(()),
        statistics=collector,
    ).run()

    assert len(result.evaluations) == 1
    assert collector.snapshot()["records"] == 1
    assert collector.records[0].objective_status == "not_configured"
    assert collector.best_individual_ids == ("session-individual",)


def test_population_collector_generates_periodic_plots_without_final_render(
    tmp_path,
) -> None:
    collector = PopulationAnalysisCollector(
        tmp_path / "analysis",
        plots=PopulationPlotConfig(every_batches=1, final_plots=False),
    )
    candidate = individual("periodic", 0)
    candidate_proposal = proposal(candidate, 0)
    item = evaluated(candidate_proposal, 0.5)

    collector.on_session_started(object())  # type: ignore[arg-type]
    collector.on_batch_started(0, (candidate,))
    collector.on_proposals_generated((candidate_proposal,))
    collector.on_evaluated_batch(EvaluatedBatch((item,), ("score",), batch_index=0))
    collector.on_batch_completed(0, (item.evaluation,))
    collector.on_session_completed(
        type("ResultView", (), {"best_individuals": (candidate,)})()
    )  # type: ignore[arg-type]

    manifest = json.loads(collector.manifest_path.read_text(encoding="utf-8"))
    assert manifest["generated"]
    assert all(path.startswith("plots/batch_000000/") for path in manifest["generated"])
    assert not (collector.output_dir / "plots/final").exists()


def test_population_statistics_facade_generates_csv_and_analysis(tmp_path) -> None:
    output_dir = tmp_path / "statistics"
    collector = PopulationStatisticsCollector(output_dir)
    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="population_statistics",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(StageChoice(slot=0, stage="stage-0"),),
                ),
            ),
        )
    )

    result = OptimizationSession(
        search_space=search_space,
        algorithm=OnePopulationAlgorithm(),
        backend=LocalBackend(base_work_dir=tmp_path / "work"),
        evaluation_workflow=EvaluationWorkflow(()),
        statistics=collector,
    ).run()

    assert collector.collectors == (
        collector.csv_collector,
        collector.analysis_collector,
    )
    assert collector.supports_checkpointing is True
    assert (output_dir / "individuals.csv").is_file()
    assert (output_dir / "run_manifest.json").is_file()
    assert (output_dir / "run_summary.json").is_file()
    assert (output_dir / "analysis/analysis_summary.json").is_file()
    assert (output_dir / "analysis/analysis_manifest.json").is_file()
    assert result.statistics["csv"]["evaluated_individuals"] == 1
    assert result.statistics["analysis"]["records"] == 1
    assert result.statistics["analysis"]["generated_plots"] > 0


def test_population_statistics_facade_restores_and_regenerates_outputs(
    tmp_path,
) -> None:
    output_dir = tmp_path / "statistics"
    checkpoint_dir = tmp_path / "checkpoints"
    work_dir = tmp_path / "work"

    def search_space() -> SearchSpace:
        return SearchSpace.from_scenario(
            SearchScenarioSpec(
                id="population_statistics",
                slots=(
                    SlotSpec(
                        index=0,
                        alternatives=(StageChoice(slot=0, stage="stage-0"),),
                    ),
                ),
            )
        )

    initial_collector = PopulationStatisticsCollector(output_dir)
    initial_result = OptimizationSession(
        search_space=search_space(),
        algorithm=GridSearch(max_evaluations=1, batch_size=1),
        backend=LocalBackend(base_work_dir=work_dir),
        evaluation_workflow=EvaluationWorkflow(()),
        statistics=initial_collector,
        checkpoint_policy=CheckpointPolicy(directory=checkpoint_dir),
    ).run()
    assert len(initial_result.evaluations) == 1
    assert initial_collector.analysis_collector.snapshots

    shutil.rmtree(output_dir / "analysis/plots/final")
    (output_dir / "analysis/analysis_summary.json").unlink()
    (output_dir / "analysis/analysis_manifest.json").unlink()

    restored_collector = PopulationStatisticsCollector(output_dir)
    restored_result = OptimizationSession(
        search_space=search_space(),
        algorithm=GridSearch(max_evaluations=1, batch_size=1),
        backend=LocalBackend(base_work_dir=work_dir),
        evaluation_workflow=EvaluationWorkflow(()),
        statistics=restored_collector,
        checkpoint_policy=CheckpointPolicy(
            directory=checkpoint_dir,
            resume_from=checkpoint_dir / "latest.json",
        ),
    ).run()

    assert len(restored_result.evaluations) == 1
    assert len(restored_collector.analysis_collector.snapshots) == 1
    assert (output_dir / "analysis/analysis_summary.json").is_file()
    assert (output_dir / "analysis/analysis_manifest.json").is_file()
    assert (output_dir / "analysis/plots/final").is_dir()
    with (output_dir / "individuals.csv").open(encoding="utf-8", newline="") as file:
        assert len(list(csv.DictReader(file))) == 1
