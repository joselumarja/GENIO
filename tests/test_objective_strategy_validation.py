from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from genio.checkpoint.codec import decode_evaluated_batch, encode_evaluated_batch
from genio.core import Evaluation, Individual, Result
from genio.objective import (
    EvaluatedBatch,
    EvaluatedIndividual,
    IdentityScalarizer,
    MetricObjective,
    MinMaxNormalizationState,
    MinMaxNormalizer,
    NormalizationScope,
    NormalizationState,
    Normalizer,
    ObjectiveSchema,
    ObjectiveSet,
    ObjectiveEvaluationStatus,
    ObjectiveValues,
    OptimizationDirection,
    Scalarizer,
    WeightedMeanScalarizer,
)


def objective(*, bounds: tuple[float, float] | None = None) -> MetricObjective:
    return MetricObjective(
        "step.metric",
        OptimizationDirection.MAXIMIZE,
        normalization_bounds=bounds,
    )


class RejectingNormalizer(Normalizer):
    def validate(self, schema: ObjectiveSchema) -> None:
        super().validate(schema)
        raise ValueError("normalizer rejected schema")

    def fit(
        self,
        values: Sequence[Sequence[float]],
        *,
        schema: ObjectiveSchema,
        previous: NormalizationState | None = None,
    ) -> NormalizationState:
        raise AssertionError("fit must not run during configuration")

    def transform(
        self,
        values: Sequence[Sequence[float]],
        state: NormalizationState,
    ) -> tuple[tuple[float, ...], ...]:
        raise AssertionError("transform must not run during configuration")

    def restore_state(self, state: Mapping[str, object]) -> NormalizationState:
        raise AssertionError("restore must not run during configuration")


class RejectingScalarizer(Scalarizer):
    def validate(self, schema: ObjectiveSchema) -> None:
        super().validate(schema)
        raise ValueError("scalarizer rejected schema")

    def scalarize(
        self,
        objective_names: Sequence[str],
        values: Sequence[float],
    ) -> float:
        raise AssertionError("scalarize must not run during configuration")

    def checkpoint_signature(self) -> Mapping[str, object]:
        return {"type": "rejecting"}


class ToggleScalarizer(Scalarizer):
    def __init__(self) -> None:
        self.fail = False

    def scalarize(
        self,
        objective_names: Sequence[str],
        values: Sequence[float],
    ) -> float:
        if self.fail:
            return float("inf")
        return float(values[0])

    def checkpoint_signature(self) -> Mapping[str, object]:
        return {"type": "toggle"}


class MalformedNormalizer(Normalizer):
    def fit(
        self,
        values: Sequence[Sequence[float]],
        *,
        schema: ObjectiveSchema,
        previous: NormalizationState | None = None,
    ) -> NormalizationState:
        return MinMaxNormalizationState(
            objective_names=schema.names,
            minimums=(0.0,),
            maximums=(1.0,),
            scope=NormalizationScope.BATCH,
            version=1,
        )

    def transform(
        self,
        values: Sequence[Sequence[float]],
        state: NormalizationState,
    ) -> tuple[tuple[float, ...], ...]:
        return ()

    def restore_state(self, state: Mapping[str, object]) -> NormalizationState:
        return MinMaxNormalizationState.restore(state)


def successful_evaluation(identifier: str, value: float) -> Evaluation:
    individual = Individual.from_slots(
        scenario="objective_validation",
        slots=(),
        id=identifier,
    )
    return Evaluation(
        individual=individual,
        result=Result.success(identifier, metrics={"step.metric": value}),
    )


def objective_values(value: float = 1.0) -> ObjectiveValues:
    return ObjectiveValues(
        names=("step.metric",),
        raw=(value,),
        minimize=(-value,),
        maximize=(value,),
    )


def test_objective_set_delegates_normalizer_validation() -> None:
    with pytest.raises(ValueError, match="normalizer rejected"):
        ObjectiveSet((objective(),), normalizer=RejectingNormalizer())


def test_objective_set_delegates_scalarizer_validation() -> None:
    with pytest.raises(ValueError, match="scalarizer rejected"):
        ObjectiveSet((objective(),), scalarizer=RejectingScalarizer())


def test_fixed_min_max_requires_bounds_during_objective_set_configuration() -> None:
    with pytest.raises(ValueError, match="requires normalization_bounds"):
        ObjectiveSet(
            (objective(),),
            normalizer=MinMaxNormalizer(NormalizationScope.FIXED),
        )

    configured = ObjectiveSet(
        (objective(bounds=(0.0, 1.0)),),
        normalizer=MinMaxNormalizer(NormalizationScope.FIXED),
    )
    assert configured.schema.normalization_bounds == ((0.0, 1.0),)


def test_weighted_mean_validates_names_through_scalarizer_contract() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        ObjectiveSet(
            (objective(),),
            scalarizer=WeightedMeanScalarizer({"other": 1.0}),
        )


def test_runtime_commits_normalization_only_after_scalarization_succeeds() -> None:
    scalarizer = ToggleScalarizer()
    runtime = ObjectiveSet(
        (objective(),),
        normalizer=MinMaxNormalizer(NormalizationScope.CUMULATIVE),
        scalarizer=scalarizer,
    ).bind()
    runtime.evaluate_batch((successful_evaluation("first", 1.0),))
    previous = runtime.normalization_state

    scalarizer.fail = True
    with pytest.raises(ValueError, match="Scalarizer result must be finite"):
        runtime.evaluate_batch((successful_evaluation("second", 2.0),))

    assert runtime.normalization_state is previous


def test_runtime_rejects_malformed_normalizer_output_without_committing() -> None:
    runtime = ObjectiveSet(
        (objective(),),
        normalizer=MalformedNormalizer(),
    ).bind()

    with pytest.raises(ValueError, match="exact finite"):
        runtime.evaluate_batch((successful_evaluation("invalid", 1.0),))

    assert runtime.normalization_state is None


def test_objective_values_validate_dimensions_and_finiteness() -> None:
    with pytest.raises(ValueError, match="expected 1"):
        ObjectiveValues(
            names=("step.metric",),
            raw=(),
            minimize=(-1.0,),
            maximize=(1.0,),
        )
    with pytest.raises(ValueError, match="must be finite"):
        objective_values(float("nan"))


def test_evaluated_batch_is_a_sequence_of_items_with_explicit_evaluations() -> None:
    first_evaluation = successful_evaluation("first", 1.0)
    second_evaluation = successful_evaluation("second", 2.0)
    first = EvaluatedIndividual(
        evaluation=first_evaluation,
        objective_values=objective_values(1.0),
        status=ObjectiveEvaluationStatus.VALID,
    )
    second = EvaluatedIndividual(
        evaluation=second_evaluation,
        objective_values=objective_values(2.0),
        status=ObjectiveEvaluationStatus.VALID,
    )
    batch = EvaluatedBatch((first, second), ("step.metric",))

    assert tuple(batch) == (first, second)
    assert batch[0] is first
    assert batch[:] == (first, second)
    assert batch.evaluations == (first_evaluation, second_evaluation)


def test_unconfigured_batch_preserves_execution_failures() -> None:
    success = successful_evaluation("success", 1.0)
    failed_individual = Individual.from_slots(
        scenario="objective_validation",
        slots=(),
        id="failed",
    )
    failure = Evaluation(
        individual=failed_individual,
        result=Result.failed("failed", "evaluation failed"),
    )

    batch = EvaluatedBatch.from_evaluations((success, failure))

    assert batch[0].status is ObjectiveEvaluationStatus.NOT_CONFIGURED
    assert batch[1].status is ObjectiveEvaluationStatus.EVALUATION_FAILED
    assert batch.valid_items == ()


def test_evaluated_batch_rejects_duplicate_individual_ids() -> None:
    evaluation = successful_evaluation("duplicate", 1.0)
    item = EvaluatedIndividual(
        evaluation=evaluation,
        objective_values=objective_values(),
        status=ObjectiveEvaluationStatus.VALID,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        EvaluatedBatch((item, item), ("step.metric",))


def test_hardened_evaluated_batch_roundtrips_through_checkpoint_codec() -> None:
    evaluation = successful_evaluation("roundtrip", 1.0)
    batch = ObjectiveSet(
        (objective(),),
        scalarizer=IdentityScalarizer(),
    ).bind().evaluate_batch((evaluation,), batch_index=0)

    restored = decode_evaluated_batch(
        encode_evaluated_batch(batch),
        evaluations=(evaluation,),
    )

    assert restored == batch


def test_minmax_batch_codec_delegates_state_restore_to_normalizer() -> None:
    evaluation = successful_evaluation("normalized-roundtrip", 0.5)
    normalizer = MinMaxNormalizer(NormalizationScope.BATCH)
    batch = ObjectiveSet(
        (objective(bounds=(0.0, 1.0)),),
        normalizer=normalizer,
    ).bind().evaluate_batch((evaluation,), batch_index=0)

    restored = decode_evaluated_batch(
        encode_evaluated_batch(batch),
        evaluations=(evaluation,),
        normalizer=normalizer,
    )

    assert restored == batch
