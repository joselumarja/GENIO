"""Immutable records used by population-oriented statistics collectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, overload

from genio.core.proposal import Proposal
from genio.core.result import ResultStatus
from genio.objective.runtime import EvaluatedIndividual, ObjectiveEvaluationStatus


@dataclass(frozen=True, slots=True)
class PopulationRecord:
    """Capture one proposed individual and its eventual objective result.

    Records are keyed by proposal occurrence rather than genotype or individual
    identity, so repeated evaluations remain distinguishable.
    """

    proposal_id: str
    individual_id: str
    batch_index: int
    batch_position: int
    genotype: tuple[int, ...] | None
    search_index: int | None
    stages: tuple[str, ...]
    stage_parameters: tuple[Mapping[str, Any], ...]
    design: Mapping[str, Any] = field(default_factory=dict)
    algorithm_metadata: Mapping[str, Any] = field(default_factory=dict)
    evaluation_status: str = "not_evaluated"
    objective_status: str = "not_evaluated"
    objective_values: Mapping[str, float] = field(default_factory=dict)
    aggregate_score: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.proposal_id, "proposal_id")
        _validate_non_empty_string(self.individual_id, "individual_id")
        _validate_non_negative_integer(self.batch_index, "batch_index")
        _validate_non_negative_integer(self.batch_position, "batch_position")
        if self.search_index is not None:
            _validate_non_negative_integer(self.search_index, "search_index")
        if self.genotype is not None:
            genotype = tuple(self.genotype)
            if any(isinstance(gene, bool) or not isinstance(gene, Integral) for gene in genotype):
                raise TypeError("genotype must contain only integers.")
            object.__setattr__(self, "genotype", tuple(int(gene) for gene in genotype))
        stages = tuple(self.stages)
        if any(not isinstance(stage, str) or not stage for stage in stages):
            raise TypeError("stages must contain non-empty strings.")
        parameters = tuple(self.stage_parameters)
        if len(parameters) != len(stages):
            raise ValueError("stage_parameters must contain one mapping per stage.")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(
            self,
            "stage_parameters",
            tuple(_freeze_mapping(value, "stage_parameters") for value in parameters),
        )
        object.__setattr__(self, "design", _freeze_mapping(self.design, "design"))
        object.__setattr__(
            self,
            "algorithm_metadata",
            _freeze_mapping(self.algorithm_metadata, "algorithm_metadata"),
        )
        _validate_non_empty_string(self.evaluation_status, "evaluation_status")
        _validate_non_empty_string(self.objective_status, "objective_status")
        valid_evaluation_statuses = {
            "not_evaluated",
            *(status.value for status in ResultStatus),
        }
        if self.evaluation_status not in valid_evaluation_statuses:
            raise ValueError("Unknown evaluation_status.")
        valid_objective_statuses = {
            "not_evaluated",
            *(status.value for status in ObjectiveEvaluationStatus),
        }
        if self.objective_status not in valid_objective_statuses:
            raise ValueError("Unknown objective_status.")
        if (self.evaluation_status == "not_evaluated") != (
            self.objective_status == "not_evaluated"
        ):
            raise ValueError(
                "Evaluation and objective statuses must be pending together."
            )
        if (
            self.objective_status == ObjectiveEvaluationStatus.VALID.value
            and self.evaluation_status != ResultStatus.SUCCESS.value
        ):
            raise ValueError("Valid objectives require a successful evaluation.")
        if (
            self.objective_status
            == ObjectiveEvaluationStatus.EVALUATION_FAILED.value
            and self.evaluation_status != ResultStatus.FAILED.value
        ):
            raise ValueError(
                "EVALUATION_FAILED objective status requires a failed evaluation."
            )
        objective_values = _freeze_objective_values(self.objective_values)
        object.__setattr__(self, "objective_values", objective_values)
        if self.aggregate_score is not None:
            object.__setattr__(
                self,
                "aggregate_score",
                _finite_float(self.aggregate_score, "aggregate_score"),
            )
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be text or None.")
        if self.objective_status == ObjectiveEvaluationStatus.VALID.value:
            if not objective_values:
                raise ValueError("A valid objective result must contain objective values.")
        elif objective_values or self.aggregate_score is not None:
            raise ValueError(
                "Objective values and aggregate score require objective_status='valid'."
            )

    @classmethod
    def from_proposal(cls, proposal: Proposal) -> PopulationRecord:
        """Create an unevaluated record from session proposal provenance."""

        if not isinstance(proposal, Proposal):
            raise TypeError("proposal must be a Proposal.")
        if proposal.batch_index is None:
            raise ValueError("Population proposals must have a batch_index.")
        algorithm_metadata = proposal.individual.metadata.get("algorithm", {})
        if not isinstance(algorithm_metadata, Mapping):
            raise TypeError("individual metadata 'algorithm' must be a mapping.")
        return cls(
            proposal_id=proposal.proposal_id,
            individual_id=proposal.individual.id,
            batch_index=proposal.batch_index,
            batch_position=proposal.batch_position,
            genotype=proposal.individual.genotype,
            search_index=proposal.individual.search_index,
            stages=tuple(choice.stage for choice in proposal.individual.slots),
            stage_parameters=tuple(
                choice.parameters for choice in proposal.individual.slots
            ),
            design=proposal.individual.design,
            algorithm_metadata=algorithm_metadata,
        )

    def with_evaluation(self, evaluated: EvaluatedIndividual) -> PopulationRecord:
        """Return a completed copy correlated by proposal and individual ID."""

        if not isinstance(evaluated, EvaluatedIndividual):
            raise TypeError("evaluated must be an EvaluatedIndividual.")
        proposal_id = evaluated.evaluation.metadata.get("proposal_id")
        if proposal_id != self.proposal_id:
            raise ValueError("Evaluated proposal_id does not match the population record.")
        if evaluated.individual.id != self.individual_id:
            raise ValueError("Evaluated individual does not match the population record.")
        values = evaluated.objective_values
        objective_values = (
            dict(zip(values.names, values.raw, strict=True))
            if values is not None
            else {}
        )
        return replace(
            self,
            evaluation_status=evaluated.evaluation.result.status.value,
            objective_status=evaluated.status.value,
            objective_values=objective_values,
            aggregate_score=(values.aggregate_score if values is not None else None),
            error=evaluated.error or evaluated.evaluation.result.error,
        )

    @property
    def evaluated(self) -> bool:
        """Return whether an evaluation result has been attached."""

        return self.evaluation_status != "not_evaluated"

    @property
    def objectives_valid(self) -> bool:
        """Return whether objective extraction completed successfully."""

        return self.objective_status == ObjectiveEvaluationStatus.VALID.value


@dataclass(frozen=True, slots=True)
class PopulationSnapshot(Sequence[PopulationRecord]):
    """Represent one complete population batch in proposal order."""

    batch_index: int
    records: tuple[PopulationRecord, ...]

    def __post_init__(self) -> None:
        _validate_non_negative_integer(self.batch_index, "batch_index")
        records = tuple(self.records)
        if any(not isinstance(record, PopulationRecord) for record in records):
            raise TypeError("records must contain only PopulationRecord instances.")
        if any(record.batch_index != self.batch_index for record in records):
            raise ValueError("All population records must belong to the snapshot batch.")
        proposal_ids = tuple(record.proposal_id for record in records)
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ValueError("Population snapshot proposal IDs must be unique.")
        positions = tuple(record.batch_position for record in records)
        if positions != tuple(range(len(records))):
            raise ValueError(
                "Population snapshot positions must be contiguous and ordered."
            )
        object.__setattr__(self, "records", records)

    def __len__(self) -> int:
        return len(self.records)

    @overload
    def __getitem__(self, index: int) -> PopulationRecord: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[PopulationRecord, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> PopulationRecord | tuple[PopulationRecord, ...]:
        return self.records[index]

    @property
    def evaluated_records(self) -> tuple[PopulationRecord, ...]:
        """Return records that have received an evaluation result."""

        return tuple(record for record in self.records if record.evaluated)

    @property
    def valid_objective_records(self) -> tuple[PopulationRecord, ...]:
        """Return records with valid objective values."""

        return tuple(record for record in self.records if record.objectives_valid)

    @property
    def genotypes(self) -> tuple[tuple[int, ...], ...]:
        """Return all available genotypes in population order."""

        return tuple(
            record.genotype for record in self.records if record.genotype is not None
        )


def _validate_non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")


def _validate_non_negative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number.")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    return converted


def _freeze_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings.")
        frozen[key] = _freeze_value(item)
    return MappingProxyType(frozen)


def _freeze_objective_values(value: object) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("objective_values must be a mapping.")
    if any(not isinstance(key, str) or not key for key in value):
        raise TypeError("objective_values keys must be non-empty strings.")
    return MappingProxyType(
        {
            key: _finite_float(item, f"objective_values[{key!r}]")
            for key, item in value.items()
        }
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value, "nested metadata")
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return deepcopy(value)


__all__ = ["PopulationRecord", "PopulationSnapshot"]
