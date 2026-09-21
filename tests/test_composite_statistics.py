from __future__ import annotations

from typing import Any

import pytest

from genio import (
    CompositeStatisticsCollector,
    Evaluation,
    Individual,
    Proposal,
    Result,
    SearchResult,
    StatisticsCollector,
)
from genio.checkpoint import CheckpointNotSupportedError
from genio.objective import EvaluatedBatch


class RecordingCollector(StatisticsCollector):
    supports_checkpointing = True

    def __init__(self, name: str, events: list[tuple[str, str]]) -> None:
        self.name = name
        self.events = events
        self.value = 0

    def _record(self, event: str) -> None:
        self.events.append((self.name, event))

    def on_session_started(self, session) -> None:
        self._record("session_started")

    def on_batch_started(self, batch_index, individuals) -> None:
        self._record("batch_started")

    def on_proposals_generated(self, proposals) -> None:
        self._record("proposals_generated")

    def on_evaluation_completed(self, evaluation) -> None:
        self._record("evaluation_completed")

    def on_evaluated_batch(self, batch) -> None:
        self._record("evaluated_batch")

    def on_batch_completed(self, batch_index, evaluations) -> None:
        self._record("batch_completed")

    def on_session_completed(self, result) -> None:
        self._record("session_completed")

    def snapshot(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value}

    def checkpoint_signature(self) -> dict[str, Any]:
        return {"type": "recording", "name": self.name}

    def checkpoint_state(self) -> dict[str, Any]:
        return {"value": self.value}

    def restore_checkpoint_state(
        self,
        state,
        *,
        session,
        evaluations,
        evaluated_batches,
        completed,
    ) -> None:
        self.value = int(state["value"])
        self._record("restored")


def evaluated_fixture():
    individual = Individual.from_slots(
        id="individual",
        scenario="statistics",
        slots=(),
    )
    evaluation = Evaluation(
        individual=individual,
        result=Result.success(individual.id),
        metadata={
            "proposal_id": "run:000000",
            "proposal_sequence": 0,
            "batch_index": 0,
            "batch_position": 0,
        },
    )
    proposal = Proposal(
        proposal_id="run:000000",
        proposal_sequence=0,
        batch_index=0,
        batch_position=0,
        individual=individual,
    )
    batch = EvaluatedBatch.from_evaluations((evaluation,), batch_index=0)
    result = SearchResult(session_id="session", evaluations=(evaluation,))
    return individual, proposal, evaluation, batch, result


def test_composite_statistics_validates_collectors() -> None:
    with pytest.raises(TypeError, match="sequence"):
        CompositeStatisticsCollector(iter(()))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        CompositeStatisticsCollector(())
    with pytest.raises(TypeError, match="only StatisticsCollector"):
        CompositeStatisticsCollector((object(),))  # type: ignore[arg-type]

    collector = RecordingCollector("one", [])
    with pytest.raises(ValueError, match="same instance"):
        CompositeStatisticsCollector((collector, collector))


def test_composite_statistics_forwards_every_hook_in_order() -> None:
    events: list[tuple[str, str]] = []
    composite = CompositeStatisticsCollector(
        (
            RecordingCollector("first", events),
            RecordingCollector("second", events),
        )
    )
    individual, proposal, evaluation, batch, result = evaluated_fixture()

    composite.on_session_started(object())  # type: ignore[arg-type]
    composite.on_batch_started(0, (individual,))
    composite.on_proposals_generated((proposal,))
    composite.on_evaluation_completed(evaluation)
    composite.on_evaluated_batch(batch)
    composite.on_batch_completed(0, (evaluation,))
    composite.on_session_completed(result)

    assert events == [
        (name, event)
        for event in (
            "session_started",
            "batch_started",
            "proposals_generated",
            "evaluation_completed",
            "evaluated_batch",
            "batch_completed",
            "session_completed",
        )
        for name in ("first", "second")
    ]


def test_composite_statistics_snapshots_and_signatures_are_collision_free() -> None:
    events: list[tuple[str, str]] = []
    first = RecordingCollector("first", events)
    second = RecordingCollector("second", events)
    first.value = 1
    second.value = 2
    composite = CompositeStatisticsCollector((first, second))

    assert [
        entry["snapshot"] for entry in composite.snapshot()["collectors"]
    ] == [
        {"name": "first", "value": 1},
        {"name": "second", "value": 2},
    ]
    assert composite.checkpoint_signature()["collectors"] == [
        {"type": "recording", "name": "first"},
        {"type": "recording", "name": "second"},
    ]


def test_composite_statistics_roundtrips_child_checkpoint_state() -> None:
    events: list[tuple[str, str]] = []
    first = RecordingCollector("first", events)
    second = RecordingCollector("second", events)
    first.value = 3
    second.value = 5
    composite = CompositeStatisticsCollector((first, second))
    state = composite.checkpoint_state()
    first.value = 0
    second.value = 0

    composite.restore_checkpoint_state(
        state,
        session=object(),  # type: ignore[arg-type]
        evaluations=(),
        evaluated_batches=(),
        completed=False,
    )

    assert (first.value, second.value) == (3, 5)
    assert events == [("first", "restored"), ("second", "restored")]


def test_composite_statistics_requires_checkpoint_support_from_every_child() -> None:
    composite = CompositeStatisticsCollector(
        (RecordingCollector("supported", []), StatisticsCollector())
    )

    assert composite.supports_checkpointing is False
    with pytest.raises(CheckpointNotSupportedError, match="StatisticsCollector"):
        composite.checkpoint_state()
