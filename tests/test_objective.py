import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass, fields
from math import nan

import pytest

from genio import Evaluation, Individual, Result, StageChoice
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
    Objective,
    ObjectiveDescriptor,
    ObjectiveError,
    ObjectiveEvaluationStatus,
    ObjectiveSchema,
    ObjectiveSet,
    ObjectiveValues,
    OptimizationDirection,
    Scalarizer,
    WeightedMeanScalarizer,
)


def make_evaluation(
    metrics: dict[str, float],
    *,
    individual_id: str = "individual_001",
) -> Evaluation:
    individual = Individual.from_slots(
        id=individual_id,
        scenario="objective_space",
        slots=[StageChoice(slot=0, stage="nop")],
    )
    return Evaluation(
        individual=individual,
        result=Result.success(individual.id, metrics=metrics),
    )


def make_failed_evaluation(
    *,
    individual_id: str = "failed_001",
    error: str = "workflow failed",
) -> Evaluation:
    individual = Individual.from_slots(
        id=individual_id,
        scenario="objective_space",
        slots=[StageChoice(slot=0, stage="nop")],
    )
    return Evaluation(
        individual=individual,
        result=Result.failed(individual.id, error),
    )


def make_values(
    *,
    names: Sequence[str] = ("score",),
    raw: Sequence[float] = (2.0,),
    normalized: bool = False,
    aggregate_score: float | None = None,
) -> ObjectiveValues:
    raw_values = tuple(raw)
    return ObjectiveValues(
        names=tuple(names),
        raw=raw_values,
        minimize=tuple(-value for value in raw_values),
        maximize=raw_values,
        normalized_minimize=(0.0,) * len(raw_values) if normalized else None,
        normalized_maximize=(0.0,) * len(raw_values) if normalized else None,
        aggregate_score=aggregate_score,
    )


class CountingObjective(Objective):
    def __init__(self, name: str, metric: str, *, fail: bool = False) -> None:
        self._name = name
        self.metric = metric
        self.fail = fail
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def direction(self) -> OptimizationDirection:
        return OptimizationDirection.MAXIMIZE

    def value(self, evaluation: Evaluation) -> float:
        self.calls += 1
        if self.fail:
            raise ObjectiveError(f"invalid {self.name}")
        return evaluation.result.metrics[self.metric]


@dataclass(frozen=True)
class CustomNormalizationState(NormalizationState):
    objective_names: tuple[str, ...]
    version: int

    def checkpoint_state(self) -> Mapping[str, object]:
        return {
            "objective_names": list(self.objective_names),
            "version": self.version,
        }


class CustomNormalizer(Normalizer):
    def __init__(self) -> None:
        self.validate_calls = 0
        self.restore_calls = 0
        self.bad_transform = False

    def validate(self, schema: ObjectiveSchema) -> None:
        self.validate_calls += 1

    def fit(
        self,
        values: Sequence[Sequence[float]],
        *,
        schema: ObjectiveSchema,
        previous: NormalizationState | None = None,
    ) -> NormalizationState:
        version = previous.version + 1 if isinstance(previous, CustomNormalizationState) else 1
        return CustomNormalizationState(schema.names, version)

    def transform(
        self,
        values: Sequence[Sequence[float]],
        state: NormalizationState,
    ) -> tuple[tuple[float, ...], ...]:
        matrix = tuple(tuple(float(value) for value in row) for row in values)
        return matrix[:-1] if self.bad_transform else matrix

    def restore_state(self, data: Mapping[str, object]) -> NormalizationState:
        self.restore_calls += 1
        return CustomNormalizationState(
            tuple(data["objective_names"]),  # type: ignore[arg-type]
            int(data["version"]),
        )

    def checkpoint_signature(self) -> Mapping[str, object]:
        return {"type": "custom"}


class CustomScalarizer(Scalarizer):
    def __init__(self, result: object = 1.0) -> None:
        self.result = result
        self.validate_calls = 0

    def validate(self, schema: ObjectiveSchema) -> None:
        self.validate_calls += 1

    def scalarize(
        self,
        objective_names: Sequence[str],
        values: Sequence[float],
    ) -> float:
        return self.result  # type: ignore[return-value]

    def checkpoint_signature(self) -> Mapping[str, object]:
        return {"type": "custom"}


class UnsignedNormalizer(CustomNormalizer):
    def checkpoint_signature(self) -> Mapping[str, object]:
        return Normalizer.checkpoint_signature(self)


class UnsignedScalarizer(CustomScalarizer):
    def checkpoint_signature(self) -> Mapping[str, object]:
        return Scalarizer.checkpoint_signature(self)


def test_metric_objective_has_exact_final_signature():
    parameters = inspect.signature(MetricObjective).parameters

    assert tuple(parameters) == (
        "metric",
        "direction",
        "name",
        "normalization_bounds",
    )
    assert parameters["name"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["normalization_bounds"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        MetricObjective("score", "maximize", id="old")
    with pytest.raises(TypeError):
        MetricObjective("score", optimization_direction="maximize")
    with pytest.raises(TypeError):
        MetricObjective("score", "maximize", bounds=(0, 1))


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("maximize", OptimizationDirection.MAXIMIZE),
        ("minimize", OptimizationDirection.MINIMIZE),
        (OptimizationDirection.MAXIMIZE, OptimizationDirection.MAXIMIZE),
    ],
)
def test_metric_objective_normalizes_direction_and_name(direction, expected):
    objective = MetricObjective(
        "quality.score",
        direction,
        name="score",
        normalization_bounds=(0, 10),
    )

    assert objective.metric == "quality.score"
    assert objective.name == "score"
    assert objective.direction is expected
    assert objective.normalization_bounds == (0.0, 10.0)
    assert not hasattr(objective, "id")
    assert not hasattr(objective, "optimization_direction")
    assert not hasattr(objective, "bounds")


def test_metric_objective_defaults_name_and_bounds():
    objective = MetricObjective("score", "maximize")

    assert objective.name == "score"
    assert objective.normalization_bounds is None


@pytest.mark.parametrize("metric", ["", "   ", None, 3])
def test_metric_objective_rejects_invalid_metric(metric):
    with pytest.raises(ObjectiveError, match="metric must be a non-empty string"):
        MetricObjective(metric, "maximize")


@pytest.mark.parametrize("name", ["", "   ", 3])
def test_metric_objective_rejects_invalid_name(name):
    with pytest.raises(ObjectiveError, match="name must be a non-empty string"):
        MetricObjective("score", "maximize", name=name)


def test_metric_objective_rejects_invalid_direction():
    with pytest.raises(ObjectiveError, match="Unknown optimization direction"):
        MetricObjective("score", "higher")


@pytest.mark.parametrize(
    "bounds",
    [
        (0,),
        (0, 1, 2),
        (True, 1),
        (0, False),
        ("0", 1),
        (0, object()),
        (0, float("inf")),
        (float("-inf"), 1),
        (0, nan),
        (1, 1),
        (2, 1),
    ],
)
def test_metric_objective_rejects_invalid_normalization_bounds(bounds):
    with pytest.raises(ObjectiveError, match="normalization"):
        MetricObjective("score", "maximize", normalization_bounds=bounds)


def test_metric_objective_extracts_finite_values_and_exposes_direction():
    evaluation = make_evaluation({"latency": 12})
    objective = MetricObjective("latency", "minimize")

    assert objective.value(evaluation) == 12.0
    assert objective.direction is OptimizationDirection.MINIMIZE
    assert not hasattr(objective, "score")


@pytest.mark.parametrize("value", [True, "bad", nan, float("inf")])
def test_metric_objective_rejects_invalid_values(value):
    objective = MetricObjective("score", "maximize")

    with pytest.raises(ObjectiveError, match="numeric|finite"):
        objective.value(make_evaluation({"score": value}))


def test_metric_objective_rejects_missing_metric():
    with pytest.raises(ObjectiveError, match="not available"):
        MetricObjective("missing", "maximize").value(make_evaluation({}))


def test_objective_exposes_no_normalization_bounds_by_default():
    objective = CountingObjective("score", "score")

    assert objective.normalization_bounds is None


def test_descriptor_and_schema_expose_normalization_bounds_only():
    descriptor = ObjectiveDescriptor(
        "score", OptimizationDirection.MAXIMIZE, (0, 1)
    )
    schema = ObjectiveSchema((descriptor,))

    assert descriptor.normalization_bounds == (0.0, 1.0)
    assert schema.normalization_bounds == ((0.0, 1.0),)
    assert not hasattr(descriptor, "bounds")
    assert not hasattr(schema, "bounds")


def test_objective_set_accepts_sequence_stores_tuple_and_defaults_to_no_strategies():
    source = [MetricObjective("score", "maximize")]
    objective_set = ObjectiveSet(source)

    source.clear()
    assert isinstance(objective_set.objectives, tuple)
    assert len(objective_set.objectives) == 1
    assert objective_set.normalizer is None
    assert objective_set.scalarizer is None
    with pytest.raises(FrozenInstanceError):
        objective_set.scalarizer = IdentityScalarizer()


def test_objective_set_rejects_non_sequence_empty_and_duplicate_names():
    with pytest.raises(TypeError, match="sequence"):
        ObjectiveSet(iter((MetricObjective("score", "maximize"),)))
    with pytest.raises(ObjectiveError, match="at least one"):
        ObjectiveSet(())
    with pytest.raises(ObjectiveError, match="Duplicate"):
        ObjectiveSet(
            (
                MetricObjective("first", "maximize", name="score"),
                MetricObjective("second", "minimize", name="score"),
            )
        )


def test_objective_set_removed_legacy_extraction_and_dominance_apis():
    objective_set = ObjectiveSet((MetricObjective("score", "maximize"),))
    import genio.objective as objective_api

    assert not hasattr(objective_set, "values")
    assert not hasattr(objective_set, "scores")
    assert not hasattr(objective_api, "IdentityNormalizer")
    assert not hasattr(objective_api, "dominates")
    assert "dominates" not in objective_api.__all__


def test_objective_set_validates_strategy_contracts_at_construction_and_bind():
    normalizer = CustomNormalizer()
    scalarizer = CustomScalarizer()
    objective_set = ObjectiveSet(
        [MetricObjective("score", "maximize")],
        normalizer=normalizer,
        scalarizer=scalarizer,
    )

    assert normalizer.validate_calls == 1
    assert scalarizer.validate_calls == 1
    objective_set.bind()
    assert normalizer.validate_calls == 2
    assert scalarizer.validate_calls == 2


def test_base_checkpoint_signatures_reject_implicit_extension_identity():
    objective_set = ObjectiveSet(
        (CountingObjective("score", "score"),),
        normalizer=UnsignedNormalizer(),
        scalarizer=UnsignedScalarizer(),
    )

    with pytest.raises(NotImplementedError, match="CountingObjective"):
        objective_set.objectives[0].checkpoint_signature()
    with pytest.raises(NotImplementedError, match="UnsignedNormalizer"):
        objective_set.normalizer.checkpoint_signature()
    with pytest.raises(NotImplementedError, match="UnsignedScalarizer"):
        objective_set.scalarizer.checkpoint_signature()


def test_builtin_checkpoint_signatures_are_complete_and_json_compatible():
    objective_set = ObjectiveSet(
        (
            MetricObjective(
                "quality.score",
                "maximize",
                name="score",
                normalization_bounds=(0, 1),
            ),
        ),
        normalizer=MinMaxNormalizer("fixed"),
        scalarizer=WeightedMeanScalarizer({"score": 2}),
    )

    signature = json.loads(json.dumps(objective_set.checkpoint_signature()))
    assert signature["objectives"][0] == {
        "type": "genio.objective.base.MetricObjective",
        "metric": "quality.score",
        "direction": "maximize",
        "name": "score",
        "normalization_bounds": [0.0, 1.0],
    }
    assert signature["normalizer"]["scope"] == "fixed"
    assert signature["scalarizer"]["weights"] == {"score": 1.0}
    assert "type" in IdentityScalarizer().checkpoint_signature()


def test_normalization_state_is_abstract_and_minmax_state_is_strict():
    with pytest.raises(TypeError):
        NormalizationState()
    state = MinMaxNormalizationState(
        objective_names=("score",),
        minimums=(0,),
        maximums=(1,),
        scope=NormalizationScope.BATCH,
        version=1,
    )

    assert state.minimums == (0.0,)
    with pytest.raises(FrozenInstanceError):
        state.version = 2
    with pytest.raises(ValueError, match="ranges must match"):
        MinMaxNormalizationState(
            objective_names=("score",),
            minimums=(),
            maximums=(),
            scope=NormalizationScope.BATCH,
            version=1,
        )


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("batch", NormalizationScope.BATCH),
        ("cumulative", NormalizationScope.CUMULATIVE),
        (NormalizationScope.FIXED, NormalizationScope.FIXED),
    ],
)
def test_minmax_normalizer_accepts_enum_or_string_scope(scope, expected):
    assert MinMaxNormalizer(scope).scope is expected


def test_minmax_normalizer_rejects_unknown_scope():
    with pytest.raises(ValueError, match="Unknown normalization scope"):
        MinMaxNormalizer("global")


def test_fixed_normalization_validates_bounds_eagerly():
    with pytest.raises(ValueError, match="normalization_bounds for every"):
        ObjectiveSet(
            (MetricObjective("score", "maximize"),),
            normalizer=MinMaxNormalizer("fixed"),
        )


def test_fixed_normalization_uses_configured_bounds():
    runtime = ObjectiveSet(
        (
            MetricObjective(
                "score", "maximize", normalization_bounds=(0, 20)
            ),
        ),
        normalizer=MinMaxNormalizer("fixed"),
    ).bind()

    batch = runtime.evaluate_batch([make_evaluation({"score": 5})])

    assert isinstance(batch.normalization_state, MinMaxNormalizationState)
    assert batch.normalization_state.minimums == (0.0,)
    assert batch.normalization_state.maximums == (20.0,)
    assert batch.normalized_maximization_matrix() == ((0.25,),)


def test_batch_normalization_prefers_bounds_and_fits_unbounded_columns():
    runtime = ObjectiveSet(
        (
            MetricObjective(
                "quality", "maximize", normalization_bounds=(0, 10)
            ),
            MetricObjective("latency", "minimize"),
        ),
        normalizer=MinMaxNormalizer("batch"),
    ).bind()

    batch = runtime.evaluate_batch(
        [
            make_evaluation({"quality": 2, "latency": 100}, individual_id="a"),
            make_evaluation({"quality": 8, "latency": 200}, individual_id="b"),
        ]
    )

    assert batch.normalization_state.minimums == (0.0, 100.0)
    assert batch.normalization_state.maximums == (10.0, 200.0)
    assert batch.normalized_maximization_matrix() == ((0.2, 1.0), (0.8, 0.0))


def test_cumulative_normalization_combines_previous_ranges_and_versions_state():
    runtime = ObjectiveSet(
        (MetricObjective("score", "maximize"),),
        normalizer=MinMaxNormalizer("cumulative"),
    ).bind()
    first = runtime.evaluate_batch(
        [
            make_evaluation({"score": 0}, individual_id="a"),
            make_evaluation({"score": 10}, individual_id="b"),
        ]
    )
    second = runtime.evaluate_batch(
        [
            make_evaluation({"score": -10}, individual_id="c"),
            make_evaluation({"score": 5}, individual_id="d"),
        ]
    )

    assert first.normalization_state.version == 1
    assert second.normalization_state.version == 2
    assert second.normalization_state.minimums == (-10.0,)
    assert second.normalization_state.maximums == (10.0,)
    assert second.normalized_maximization_matrix() == ((0.0,), (0.75,))


def test_constant_minmax_column_is_neutral_in_both_orientations():
    runtime = ObjectiveSet(
        (MetricObjective("score", "maximize"),),
        normalizer=MinMaxNormalizer(),
    ).bind()
    batch = runtime.evaluate_batch(
        [
            make_evaluation({"score": 4}, individual_id="a"),
            make_evaluation({"score": 4}, individual_id="b"),
        ]
    )

    assert batch.normalized_minimization_matrix() == ((0.0,), (0.0,))
    assert batch.normalized_maximization_matrix() == ((0.0,), (0.0,))


def test_minmax_restore_state_validates_its_own_scope():
    state_data = {
        "objective_names": ["score"],
        "minimums": [0],
        "maximums": [1],
        "scope": "batch",
        "version": 1,
    }

    restored = MinMaxNormalizer("batch").restore_state(state_data)
    assert restored == MinMaxNormalizationState(
        ("score",), (0.0,), (1.0,), NormalizationScope.BATCH, 1
    )
    with pytest.raises(ValueError, match="scope"):
        MinMaxNormalizer("cumulative").restore_state(state_data)


def test_weighted_mean_uses_equal_mean_without_weights():
    scalarizer = WeightedMeanScalarizer()

    assert scalarizer.scalarize(("quality", "latency"), (0.25, 0.75)) == 0.5


def test_weighted_mean_validates_and_normalizes_explicit_weights():
    scalarizer = WeightedMeanScalarizer({"quality": 3, "latency": 1, "area": 0})
    schema = ObjectiveSchema(
        (
            ObjectiveDescriptor("area", OptimizationDirection.MINIMIZE),
            ObjectiveDescriptor("latency", OptimizationDirection.MINIMIZE),
            ObjectiveDescriptor("quality", OptimizationDirection.MAXIMIZE),
        )
    )

    scalarizer.validate(schema)
    assert dict(scalarizer.weights) == {
        "quality": 0.75,
        "latency": 0.25,
        "area": 0.0,
    }
    assert scalarizer.scalarize(schema.names, (99, 0, 1)) == 0.75
    with pytest.raises(TypeError):
        scalarizer.weights["quality"] = 1.0


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"score": 0},
        {"score": -1},
        {"score": True},
        {"score": "1"},
        {"score": float("inf")},
        {"score": nan},
    ],
)
def test_weighted_mean_rejects_invalid_weights(weights):
    with pytest.raises((TypeError, ValueError)):
        WeightedMeanScalarizer(weights)


def test_weighted_mean_requires_exact_schema_names_at_objective_set_creation():
    with pytest.raises(ValueError, match="exactly match"):
        ObjectiveSet(
            (MetricObjective("quality", "maximize"),),
            scalarizer=WeightedMeanScalarizer({"other": 1}),
        )


def test_identity_scalarizer_requires_exactly_one_objective():
    assert (
        ObjectiveSet(
            (MetricObjective("score", "maximize"),),
            scalarizer=IdentityScalarizer(),
        ).scalarizer.scalarize(("score",), (4,))
        == 4.0
    )
    with pytest.raises(ValueError, match="exactly one"):
        ObjectiveSet(
            (
                MetricObjective("first", "maximize"),
                MetricObjective("second", "maximize"),
            ),
            scalarizer=IdentityScalarizer(),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"names": ()},
        {"names": ("score", "score"), "raw": (1, 2)},
        {"raw": ()},
        {"raw": (nan,)},
        {"normalized_minimize": (0,), "normalized_maximize": None},
        {"normalized_minimize": (0, 1), "normalized_maximize": (0, 1)},
        {"aggregate_score": float("inf")},
    ],
)
def test_objective_values_reject_invalid_names_dimensions_and_finiteness(changes):
    arguments = {
        "names": ("score",),
        "raw": (1,),
        "minimize": (-1,),
        "maximize": (1,),
        "normalized_minimize": None,
        "normalized_maximize": None,
        "aggregate_score": None,
    }
    arguments.update(changes)

    with pytest.raises((TypeError, ValueError)):
        ObjectiveValues(**arguments)


def test_objective_values_copy_sequences_and_convert_to_floats():
    values = ObjectiveValues(
        names=["score"],
        raw=[1],
        minimize=[-1],
        maximize=[1],
        normalized_minimize=[0],
        normalized_maximize=[1],
        aggregate_score=1,
    )

    assert values.names == ("score",)
    assert values.raw == (1.0,)
    assert values.aggregate_score == 1.0


def test_evaluated_individual_has_single_authoritative_individual():
    evaluation = make_evaluation({"score": 2})
    item = EvaluatedIndividual(
        evaluation=evaluation,
        objective_values=make_values(),
        status=ObjectiveEvaluationStatus.VALID,
    )

    assert item.individual is evaluation.individual
    assert "individual" not in {field.name for field in fields(EvaluatedIndividual)}


def test_evaluated_individual_rejects_mismatched_evaluation_identity():
    evaluation = make_evaluation({"score": 2})
    mismatched = Evaluation(
        individual=evaluation.individual,
        result=Result.success("other", {"score": 2}),
    )

    with pytest.raises(ValueError, match="identity"):
        EvaluatedIndividual(
            evaluation=mismatched,
            objective_values=make_values(),
            status=ObjectiveEvaluationStatus.VALID,
        )


@pytest.mark.parametrize(
    ("evaluation", "values", "status", "error"),
    [
        (
            make_failed_evaluation(),
            make_values(),
            ObjectiveEvaluationStatus.VALID,
            None,
        ),
        (
            make_evaluation({}),
            None,
            ObjectiveEvaluationStatus.EVALUATION_FAILED,
            None,
        ),
        (
            make_evaluation({}),
            None,
            ObjectiveEvaluationStatus.INVALID_OBJECTIVES,
            None,
        ),
        (
            make_evaluation({}),
            make_values(),
            ObjectiveEvaluationStatus.NOT_CONFIGURED,
            None,
        ),
    ],
)
def test_evaluated_individual_rejects_incoherent_status(
    evaluation, values, status, error
):
    with pytest.raises(ValueError):
        EvaluatedIndividual(
            evaluation=evaluation,
            objective_values=values,
            status=status,
            error=error,
        )


def test_evaluated_batch_is_a_sequence_of_items_and_exposes_sources():
    first_evaluation = make_evaluation({"score": 1}, individual_id="a")
    second_evaluation = make_evaluation({"score": 2}, individual_id="b")
    first = EvaluatedIndividual(
        first_evaluation, make_values(raw=(1,)), ObjectiveEvaluationStatus.VALID
    )
    second = EvaluatedIndividual(
        second_evaluation, make_values(raw=(2,)), ObjectiveEvaluationStatus.VALID
    )
    batch = EvaluatedBatch([first, second], ["score"])

    assert isinstance(batch, Sequence)
    assert tuple(batch) == (first, second)
    assert batch[0] is first
    assert batch[:] == (first, second)
    assert batch.evaluations == (first_evaluation, second_evaluation)
    assert batch.individuals == (
        first_evaluation.individual,
        second_evaluation.individual,
    )


def test_evaluated_batch_from_evaluations_marks_unconfigured_and_failed_items():
    success = make_evaluation({}, individual_id="success")
    failure = make_failed_evaluation(individual_id="failure")

    batch = EvaluatedBatch.from_evaluations((success, failure), batch_index=3)

    assert batch.objective_names == ()
    assert batch[0].status is ObjectiveEvaluationStatus.NOT_CONFIGURED
    assert batch[1].status is ObjectiveEvaluationStatus.EVALUATION_FAILED
    assert batch.valid_items == ()
    assert batch.valid_indices == ()
    assert batch.raw_matrix() == ()
    assert batch.aggregate_scores() == ()


def test_evaluated_batch_rejects_item_name_and_not_configured_schema_mismatches():
    evaluation = make_evaluation({"score": 1})
    valid = EvaluatedIndividual(
        evaluation, make_values(), ObjectiveEvaluationStatus.VALID
    )
    unconfigured = EvaluatedIndividual(
        evaluation, None, ObjectiveEvaluationStatus.NOT_CONFIGURED
    )

    with pytest.raises(ValueError, match="names"):
        EvaluatedBatch((valid,), ("other",))
    with pytest.raises(ValueError, match="NOT_CONFIGURED"):
        EvaluatedBatch((unconfigured,), ("score",))


def test_evaluated_batch_rejects_normalization_state_name_mismatch():
    evaluation = make_evaluation({"score": 1})
    valid = EvaluatedIndividual(
        evaluation, make_values(), ObjectiveEvaluationStatus.VALID
    )
    state = CustomNormalizationState(("other",), 1)

    with pytest.raises(ValueError, match="state objective names"):
        EvaluatedBatch((valid,), ("score",), normalization_state=state)


def test_matrices_and_scores_return_only_valid_items_with_source_indices():
    runtime = ObjectiveSet(
        (MetricObjective("score", "maximize"),),
        scalarizer=IdentityScalarizer(),
    ).bind()
    batch = runtime.evaluate_batch(
        [
            make_failed_evaluation(individual_id="failed"),
            make_evaluation({"score": 3}, individual_id="valid"),
        ]
    )

    assert batch.valid_indices == (1,)
    assert batch.raw_matrix() == ((3.0,),)
    assert batch.minimization_matrix() == ((-3.0,),)
    assert batch.maximization_matrix() == ((3.0,),)
    assert batch.aggregate_scores() == (3.0,)
    assert batch.raw_matrix_with_placeholder(invalid_value=-1) == ((-1.0,), (3.0,))
    assert batch.aggregate_scores_with_placeholder(invalid_value=-2) == (-2.0, 3.0)
    with pytest.raises(TypeError):
        batch.raw_matrix(invalid_value=0)
    with pytest.raises(TypeError):
        batch.aggregate_scores(invalid_value=0)


def test_runtime_defaults_to_raw_vectors_without_aggregate_or_normalized_values():
    batch = ObjectiveSet((MetricObjective("latency", "minimize"),)).bind().evaluate_batch(
        [make_evaluation({"latency": 3})]
    )

    assert batch.raw_matrix() == ((3.0,),)
    assert batch.maximization_matrix() == ((-3.0,),)
    with pytest.raises(ObjectiveError, match="normalizer"):
        batch.normalized_maximization_matrix()
    with pytest.raises(ObjectiveError, match="scalarizer"):
        batch.aggregate_scores()


def test_runtime_scalarizes_normalized_maximization_values_when_configured():
    runtime = ObjectiveSet(
        (
            MetricObjective("quality", "maximize"),
            MetricObjective("latency", "minimize"),
        ),
        normalizer=MinMaxNormalizer(),
        scalarizer=WeightedMeanScalarizer({"quality": 3, "latency": 1}),
    ).bind()
    batch = runtime.evaluate_batch(
        [
            make_evaluation({"quality": 2, "latency": 1}, individual_id="a"),
            make_evaluation({"quality": 8, "latency": 9}, individual_id="b"),
        ]
    )

    assert batch.aggregate_scores() == (0.25, 0.75)


def test_runtime_eagerly_extracts_every_objective_once_and_skips_failed_results():
    invalid = CountingObjective("invalid", "invalid", fail=True)
    valid = CountingObjective("valid", "valid")
    runtime = ObjectiveSet((invalid, valid)).bind()

    batch = runtime.evaluate_batch(
        [
            make_failed_evaluation(individual_id="failed"),
            make_evaluation({"invalid": 1, "valid": 2}, individual_id="evaluated"),
        ]
    )

    assert invalid.calls == 1
    assert valid.calls == 1
    assert batch[0].status is ObjectiveEvaluationStatus.EVALUATION_FAILED
    assert batch[1].status is ObjectiveEvaluationStatus.INVALID_OBJECTIVES
    assert "invalid invalid" in batch[1].error


def test_runtime_validates_normalizer_matrix_postconditions_transactionally():
    normalizer = CustomNormalizer()
    runtime = ObjectiveSet(
        (MetricObjective("score", "maximize"),), normalizer=normalizer
    ).bind()
    runtime.evaluate_batch([make_evaluation({"score": 1})])
    previous = runtime.normalization_state
    normalizer.bad_transform = True

    with pytest.raises(ValueError, match="exact finite"):
        runtime.evaluate_batch([make_evaluation({"score": 2})])
    assert runtime.normalization_state is previous
    assert previous.version == 1
    assert normalizer.restore_calls == 0


@pytest.mark.parametrize("invalid_result", [True, "bad", nan, float("inf")])
def test_runtime_validates_scalarizer_postcondition_without_publishing_state(
    invalid_result,
):
    normalizer = CustomNormalizer()
    scalarizer = CustomScalarizer(invalid_result)
    runtime = ObjectiveSet(
        (MetricObjective("score", "maximize"),),
        normalizer=normalizer,
        scalarizer=scalarizer,
    ).bind()

    with pytest.raises((TypeError, ValueError), match="Scalarizer result"):
        runtime.evaluate_batch([make_evaluation({"score": 1})])
    assert runtime.normalization_state is None


def test_all_failed_batch_does_not_report_inherited_batch_normalization_state():
    runtime = ObjectiveSet(
        (MetricObjective("score", "maximize"),),
        normalizer=MinMaxNormalizer("batch"),
    ).bind()
    first = runtime.evaluate_batch([make_evaluation({"score": 1})])
    previous = runtime.normalization_state
    failed = runtime.evaluate_batch([make_failed_evaluation()])

    assert first.normalization_state is previous
    assert failed.normalization_state is None
    assert runtime.normalization_state is previous


def test_runtime_checkpoint_restore_delegates_to_normalizer_and_is_transactional():
    normalizer = CustomNormalizer()
    runtime = ObjectiveSet(
        (MetricObjective("score", "maximize"),), normalizer=normalizer
    ).bind()
    state = {
        "normalization": {"objective_names": ["score"], "version": 4},
    }

    runtime.restore(json.loads(json.dumps(state)))

    assert normalizer.restore_calls == 1
    assert runtime.normalization_state == CustomNormalizationState(("score",), 4)
    with pytest.raises(ValueError, match="objective names"):
        runtime.restore(
            {
                "normalization": {"objective_names": ["other"], "version": 5},
            }
        )
    assert runtime.normalization_state == CustomNormalizationState(("score",), 4)


def test_minmax_runtime_checkpoint_round_trip_is_json_compatible():
    objective_set = ObjectiveSet(
        (MetricObjective("score", "maximize"),),
        normalizer=MinMaxNormalizer("cumulative"),
        scalarizer=IdentityScalarizer(),
    )
    runtime = objective_set.bind()
    runtime.evaluate_batch(
        [
            make_evaluation({"score": 1}, individual_id="a"),
            make_evaluation({"score": 3}, individual_id="b"),
        ]
    )
    serialized = json.loads(json.dumps(runtime.checkpoint_state()))
    restored = objective_set.bind()

    restored.restore(serialized)
    next_batch = restored.evaluate_batch([make_evaluation({"score": 5})])

    assert next_batch.normalization_state.version == 2
    assert next_batch.normalization_state.minimums == (1.0,)
    assert next_batch.normalization_state.maximums == (5.0,)


@pytest.mark.parametrize(
    "state",
    [
        {"normalization": None, "unexpected": True},
        {},
        {"normalization": []},
        [],
    ],
)
def test_runtime_rejects_malformed_checkpoint_without_mutation(state):
    runtime = ObjectiveSet((MetricObjective("score", "maximize"),)).bind()

    with pytest.raises((TypeError, ValueError)):
        runtime.restore(state)
    assert runtime.normalization_state is None
