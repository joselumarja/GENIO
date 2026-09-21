from __future__ import annotations

import pytest

from genio import Evaluation, Individual, Proposal, Result, StageChoice
from genio.objective import (
    EvaluatedIndividual,
    ObjectiveEvaluationStatus,
    ObjectiveValues,
)
from genio.statistics import PopulationRecord, PopulationSnapshot


def proposal_fixture(
    *,
    proposal_id: str = "run:000000",
    batch_index: int = 0,
    batch_position: int = 0,
    individual_id: str = "individual",
) -> Proposal:
    individual = Individual.from_slots(
        id=individual_id,
        scenario="population",
        slots=(
            StageChoice(
                slot=0,
                stage="threshold",
                parameters={"threshold": 127},
            ),
            StageChoice(
                slot=1,
                stage="blur",
                parameters={"kernel": 3},
            ),
        ),
        genotype=(1, 2),
        search_index=5,
        design={"hls": {"npc": "XF_NPPC1"}},
        metadata={
            "algorithm": {
                "generation": 2,
                "parent_ids": ["parent-a", "parent-b"],
            }
        },
    )
    return Proposal(
        proposal_id=proposal_id,
        proposal_sequence=batch_position,
        batch_index=batch_index,
        batch_position=batch_position,
        individual=individual,
    )


def completed_item(proposal: Proposal) -> EvaluatedIndividual:
    evaluation = Evaluation(
        individual=proposal.individual,
        result=Result.success(proposal.individual.id, {"quality.f1": 0.9}),
        metadata=proposal.evaluation_metadata(),
    )
    return EvaluatedIndividual(
        evaluation=evaluation,
        objective_values=ObjectiveValues(
            names=("quality", "latency"),
            raw=(0.9, 12.0),
            minimize=(-0.9, 12.0),
            maximize=(0.9, -12.0),
            aggregate_score=0.8,
        ),
        status=ObjectiveEvaluationStatus.VALID,
    )


def test_population_record_captures_proposal_configuration_immutably() -> None:
    proposal = proposal_fixture()
    record = PopulationRecord.from_proposal(proposal)

    assert record.proposal_id == "run:000000"
    assert record.batch_index == 0
    assert record.genotype == (1, 2)
    assert record.stages == ("threshold", "blur")
    assert record.stage_parameters[0] == {"threshold": 127}
    assert record.design == {"hls": {"npc": "XF_NPPC1"}}
    assert record.algorithm_metadata["parent_ids"] == ("parent-a", "parent-b")
    assert record.evaluated is False

    proposal.individual.design["hls"]["npc"] = "changed"
    assert record.design["hls"]["npc"] == "XF_NPPC1"
    with pytest.raises(TypeError):
        record.design["new"] = 1  # type: ignore[index]


def test_population_record_completes_from_matching_evaluated_individual() -> None:
    proposal = proposal_fixture()
    pending = PopulationRecord.from_proposal(proposal)

    completed = pending.with_evaluation(completed_item(proposal))

    assert pending.evaluated is False
    assert completed.evaluated is True
    assert completed.objectives_valid is True
    assert completed.evaluation_status == "success"
    assert completed.objective_values == {"quality": 0.9, "latency": 12.0}
    assert completed.aggregate_score == 0.8


def test_population_record_rejects_mismatched_evaluation_occurrence() -> None:
    pending = PopulationRecord.from_proposal(proposal_fixture())

    with pytest.raises(ValueError, match="proposal_id"):
        pending.with_evaluation(
            completed_item(proposal_fixture(proposal_id="run:000001"))
        )


def test_population_snapshot_validates_batch_and_order() -> None:
    first = PopulationRecord.from_proposal(proposal_fixture())
    second = PopulationRecord.from_proposal(
        proposal_fixture(
            proposal_id="run:000001",
            batch_position=1,
            individual_id="second",
        )
    )
    snapshot = PopulationSnapshot(0, (first, second))

    assert tuple(snapshot) == (first, second)
    assert snapshot[:] == (first, second)
    assert snapshot.genotypes == ((1, 2), (1, 2))
    assert snapshot.evaluated_records == ()

    with pytest.raises(ValueError, match="contiguous"):
        PopulationSnapshot(0, (second, first))


def test_population_snapshot_exposes_evaluated_and_valid_views() -> None:
    proposal = proposal_fixture()
    completed = PopulationRecord.from_proposal(proposal).with_evaluation(
        completed_item(proposal)
    )
    snapshot = PopulationSnapshot(0, (completed,))

    assert snapshot.evaluated_records == (completed,)
    assert snapshot.valid_objective_records == (completed,)
