"""Session-scoped extraction and transformation of objective values."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, overload

from genio.core.evaluation import Evaluation
from genio.core.individual import Individual
from genio.core.result import ResultStatus
from genio.objective.base import ObjectiveError, ObjectiveSchema, OptimizationDirection
from genio.objective.normalization import NormalizationState

if TYPE_CHECKING:
    from genio.objective.base import Objective, ObjectiveSet


class ObjectiveEvaluationStatus(str, Enum):
    """Validity states produced while interpreting an evaluation."""

    VALID = "valid"
    EVALUATION_FAILED = "evaluation_failed"
    INVALID_OBJECTIVES = "invalid_objectives"
    NOT_CONFIGURED = "not_configured"


@dataclass(frozen=True, slots=True)
class ObjectiveValues:
    """All reusable representations of one objective vector."""

    names: tuple[str, ...]
    raw: tuple[float, ...]
    minimize: tuple[float, ...]
    maximize: tuple[float, ...]
    normalized_minimize: tuple[float, ...] | None = None
    normalized_maximize: tuple[float, ...] | None = None
    aggregate_score: float | None = None

    def __post_init__(self) -> None:
        names = _validated_names(self.names)
        raw = _validated_values(self.raw, len(names), "raw")
        minimize = _validated_values(self.minimize, len(names), "minimize")
        maximize = _validated_values(self.maximize, len(names), "maximize")
        if (self.normalized_minimize is None) != (self.normalized_maximize is None):
            raise ValueError(
                "normalized_minimize and normalized_maximize must both be set or None."
            )
        normalized_minimize = (
            _validated_values(
                self.normalized_minimize,
                len(names),
                "normalized_minimize",
            )
            if self.normalized_minimize is not None
            else None
        )
        normalized_maximize = (
            _validated_values(
                self.normalized_maximize,
                len(names),
                "normalized_maximize",
            )
            if self.normalized_maximize is not None
            else None
        )
        aggregate_score = (
            _finite_float(self.aggregate_score, "aggregate_score")
            if self.aggregate_score is not None
            else None
        )
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "raw", raw)
        object.__setattr__(self, "minimize", minimize)
        object.__setattr__(self, "maximize", maximize)
        object.__setattr__(self, "normalized_minimize", normalized_minimize)
        object.__setattr__(self, "normalized_maximize", normalized_maximize)
        object.__setattr__(self, "aggregate_score", aggregate_score)


@dataclass(frozen=True, slots=True)
class EvaluatedIndividual:
    """Pair a candidate with its evaluation and objective representations."""

    evaluation: Evaluation
    objective_values: ObjectiveValues | None
    status: ObjectiveEvaluationStatus
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, Evaluation):
            raise TypeError("evaluation must be an Evaluation.")
        if self.evaluation.individual.id != self.evaluation.result.individual_id:
            raise ValueError("Evaluation and result identity must match.")
        if not isinstance(self.status, ObjectiveEvaluationStatus):
            raise TypeError("status must be an ObjectiveEvaluationStatus.")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be text or None.")

        result_succeeded = self.evaluation.result.status is ResultStatus.SUCCESS
        if self.status is ObjectiveEvaluationStatus.VALID:
            if not result_succeeded or self.objective_values is None or self.error is not None:
                raise ValueError("VALID objective data requires a successful evaluation.")
        elif self.status is ObjectiveEvaluationStatus.EVALUATION_FAILED:
            if result_succeeded or self.objective_values is not None:
                raise ValueError(
                    "EVALUATION_FAILED requires a failed evaluation without objectives."
                )
        elif self.status is ObjectiveEvaluationStatus.INVALID_OBJECTIVES:
            if (
                not result_succeeded
                or self.objective_values is not None
                or not self.error
            ):
                raise ValueError(
                    "INVALID_OBJECTIVES requires a successful evaluation and an error."
                )
        elif self.status is ObjectiveEvaluationStatus.NOT_CONFIGURED:
            if not result_succeeded or self.objective_values is not None or self.error is not None:
                raise ValueError(
                    "NOT_CONFIGURED requires a successful evaluation without objectives."
                )

    @property
    def individual(self) -> Individual:
        """Return the single authoritative individual from the evaluation."""

        return self.evaluation.individual

    @property
    def valid(self) -> bool:
        """Return whether all objectives were extracted successfully."""

        return self.status is ObjectiveEvaluationStatus.VALID


@dataclass(frozen=True, slots=True)
class EvaluatedBatch(Sequence[EvaluatedIndividual]):
    """Ordered objective-aware results delivered to a search algorithm."""

    items: tuple[EvaluatedIndividual, ...]
    objective_names: tuple[str, ...]
    batch_index: int | None = None
    normalization_state: NormalizationState | None = None

    def __post_init__(self) -> None:
        items = tuple(self.items)
        names = _validated_names(self.objective_names, allow_empty=True)
        if any(not isinstance(item, EvaluatedIndividual) for item in items):
            raise TypeError("items must contain EvaluatedIndividual instances.")
        identifiers = tuple(item.individual.id for item in items)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Duplicate individual IDs are not allowed in a batch.")
        if self.batch_index is not None and (
            isinstance(self.batch_index, bool)
            or not isinstance(self.batch_index, Integral)
            or self.batch_index < 0
        ):
            raise ValueError("batch_index must be a non-negative integer or None.")
        for item in items:
            if item.status is ObjectiveEvaluationStatus.NOT_CONFIGURED:
                if names:
                    raise ValueError(
                        "NOT_CONFIGURED items require an empty objective schema."
                    )
                continue
            if item.objective_values is not None and item.objective_values.names != names:
                raise ValueError("EvaluatedBatch item objective names do not match.")
        if self.normalization_state is not None:
            if self.normalization_state.objective_names != names:
                raise ValueError(
                    "Normalization state objective names do not match the batch."
                )
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "objective_names", names)
        if self.batch_index is not None:
            object.__setattr__(self, "batch_index", int(self.batch_index))

    @classmethod
    def from_evaluations(
        cls,
        evaluations: Sequence[Evaluation],
        *,
        batch_index: int | None = None,
    ) -> EvaluatedBatch:
        """Build a batch for algorithms used without objective configuration."""

        return cls(
            items=tuple(
                EvaluatedIndividual(
                    evaluation=evaluation,
                    objective_values=None,
                    status=(
                        ObjectiveEvaluationStatus.NOT_CONFIGURED
                        if evaluation.result.status is ResultStatus.SUCCESS
                        else ObjectiveEvaluationStatus.EVALUATION_FAILED
                    ),
                    error=(
                        None
                        if evaluation.result.status is ResultStatus.SUCCESS
                        else evaluation.result.error
                    ),
                )
                for evaluation in evaluations
            ),
            objective_names=(),
            batch_index=batch_index,
        )

    def __len__(self) -> int:
        """Return the number of evaluated individuals."""

        return len(self.items)

    def __iter__(self) -> Iterator[EvaluatedIndividual]:
        """Iterate over objective-aware items in evaluation order."""

        return iter(self.items)

    @overload
    def __getitem__(self, index: int) -> EvaluatedIndividual: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[EvaluatedIndividual, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> EvaluatedIndividual | tuple[EvaluatedIndividual, ...]:
        """Return one item or an immutable slice of items."""

        return self.items[index]

    @property
    def evaluations(self) -> tuple[Evaluation, ...]:
        """Return source evaluations in input order."""

        return tuple(item.evaluation for item in self.items)

    @property
    def individuals(self) -> tuple[Individual, ...]:
        """Return source individuals in input order."""

        return tuple(item.individual for item in self.items)

    @property
    def valid_items(self) -> tuple[EvaluatedIndividual, ...]:
        """Return only successfully interpreted items."""

        return tuple(item for item in self.items if item.valid)

    @property
    def valid_indices(self) -> tuple[int, ...]:
        """Return source positions of successfully interpreted items."""

        return tuple(index for index, item in enumerate(self.items) if item.valid)

    @property
    def valid_mask(self) -> tuple[bool, ...]:
        """Mark successfully interpreted evaluations."""

        return tuple(item.valid for item in self.items)

    @property
    def invalid_mask(self) -> tuple[bool, ...]:
        """Mark all failed or invalid-objective evaluations."""

        return tuple(not item.valid for item in self.items)

    @property
    def evaluation_failed_mask(self) -> tuple[bool, ...]:
        """Mark evaluations whose workflow result failed."""

        return tuple(
            item.status is ObjectiveEvaluationStatus.EVALUATION_FAILED
            for item in self.items
        )

    @property
    def invalid_objectives_mask(self) -> tuple[bool, ...]:
        """Mark successful results with invalid objective extraction."""

        return tuple(
            item.status is ObjectiveEvaluationStatus.INVALID_OBJECTIVES
            for item in self.items
        )

    def raw_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Return raw rows for valid items only."""

        return self._matrix("raw")

    def minimization_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Return minimization rows for valid items only."""

        return self._matrix("minimize")

    def maximization_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Return maximization rows for valid items only."""

        return self._matrix("maximize")

    def normalized_minimization_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Return normalized minimization rows for valid items only."""

        return self._matrix("normalized_minimize")

    def normalized_maximization_matrix(self) -> tuple[tuple[float, ...], ...]:
        """Return normalized maximization rows for valid items only."""

        return self._matrix("normalized_maximize")

    def aggregate_scores(self) -> tuple[float, ...]:
        """Return aggregate scores for valid items only."""

        scores: list[float] = []
        for item in self.valid_items:
            assert item.objective_values is not None
            score = item.objective_values.aggregate_score
            if score is None:
                raise ObjectiveError("ObjectiveSet has no configured scalarizer.")
            scores.append(score)
        return tuple(scores)

    def raw_matrix_with_placeholder(
        self, *, invalid_value: float
    ) -> tuple[tuple[float, ...], ...]:
        return self._matrix_with_placeholder("raw", invalid_value)

    def minimization_matrix_with_placeholder(
        self, *, invalid_value: float
    ) -> tuple[tuple[float, ...], ...]:
        return self._matrix_with_placeholder("minimize", invalid_value)

    def maximization_matrix_with_placeholder(
        self, *, invalid_value: float
    ) -> tuple[tuple[float, ...], ...]:
        return self._matrix_with_placeholder("maximize", invalid_value)

    def normalized_minimization_matrix_with_placeholder(
        self, *, invalid_value: float
    ) -> tuple[tuple[float, ...], ...]:
        return self._matrix_with_placeholder("normalized_minimize", invalid_value)

    def normalized_maximization_matrix_with_placeholder(
        self, *, invalid_value: float
    ) -> tuple[tuple[float, ...], ...]:
        return self._matrix_with_placeholder("normalized_maximize", invalid_value)

    def aggregate_scores_with_placeholder(
        self, *, invalid_value: float
    ) -> tuple[float, ...]:
        placeholder = _finite_float(invalid_value, "invalid_value")
        scores: list[float] = []
        for item in self.items:
            if not item.valid:
                scores.append(placeholder)
                continue
            assert item.objective_values is not None
            score = item.objective_values.aggregate_score
            if score is None:
                raise ObjectiveError("ObjectiveSet has no configured scalarizer.")
            scores.append(score)
        return tuple(scores)

    def _matrix(
        self,
        attribute: str,
    ) -> tuple[tuple[float, ...], ...]:
        rows: list[tuple[float, ...]] = []
        for item in self.valid_items:
            assert item.objective_values is not None
            values = getattr(item.objective_values, attribute)
            if values is None:
                raise ObjectiveError(
                    f"Objective representation {attribute!r} requires a normalizer."
                )
            rows.append(values)
        return tuple(rows)

    def _matrix_with_placeholder(
        self,
        attribute: str,
        invalid_value: float,
    ) -> tuple[tuple[float, ...], ...]:
        placeholder = _finite_float(invalid_value, "invalid_value")
        invalid_row = (placeholder,) * len(self.objective_names)
        valid_rows = iter(self._matrix(attribute))
        return tuple(
            tuple(next(valid_rows)) if item.valid else invalid_row
            for item in self.items
        )


class ObjectiveRuntime:
    """Evaluate objectives once and own normalization state for one session."""

    def __init__(self, objective_set: "ObjectiveSet") -> None:
        from genio.objective.base import ObjectiveSet

        if not isinstance(objective_set, ObjectiveSet):
            raise TypeError("objective_set must be an ObjectiveSet.")
        self.objective_set = objective_set
        self._normalization_state: NormalizationState | None = None

    @property
    def schema(self) -> ObjectiveSchema:
        """Return the immutable schema of the bound configuration."""

        return self.objective_set.schema

    @property
    def normalization_state(self) -> NormalizationState | None:
        """Return the most recently fitted immutable normalization state."""

        return self._normalization_state

    def evaluate_batch(
        self,
        evaluations: Sequence[Evaluation],
        *,
        batch_index: int | None = None,
    ) -> EvaluatedBatch:
        """Extract, orient, normalize, and scalarize an evaluation batch."""

        evaluations = tuple(evaluations)
        names = self.schema.names
        raw_values: list[tuple[float, ...] | None] = []
        statuses: list[ObjectiveEvaluationStatus] = []
        errors: list[str | None] = []

        for evaluation in evaluations:
            if evaluation.individual.id != evaluation.result.individual_id:
                raise ValueError(
                    f"Evaluation individual {evaluation.individual.id!r} does not "
                    f"match result {evaluation.result.individual_id!r}."
                )
            if evaluation.result.status is not ResultStatus.SUCCESS:
                raw_values.append(None)
                statuses.append(ObjectiveEvaluationStatus.EVALUATION_FAILED)
                errors.append(evaluation.result.error)
                continue
            values: list[float] = []
            objective_errors: list[str] = []
            for objective in self.objective_set.objectives:
                try:
                    values.append(self._objective_value(objective, evaluation))
                except ObjectiveError as exc:
                    objective_errors.append(str(exc))
            if objective_errors:
                raw_values.append(None)
                statuses.append(ObjectiveEvaluationStatus.INVALID_OBJECTIVES)
                errors.append("; ".join(objective_errors))
                continue
            raw_values.append(tuple(values))
            statuses.append(ObjectiveEvaluationStatus.VALID)
            errors.append(None)

        valid_values = tuple(value for value in raw_values if value is not None)
        next_state = self._normalization_state
        batch_state: NormalizationState | None = None
        normalized_values: tuple[tuple[float, ...], ...] | None = None
        if valid_values and self.objective_set.normalizer is not None:
            candidate_state = self.objective_set.normalizer.fit(
                valid_values,
                schema=self.schema,
                previous=self._normalization_state,
            )
            if not isinstance(candidate_state, NormalizationState):
                raise TypeError(
                    "Normalizer.fit() must return a NormalizationState instance."
                )
            if candidate_state.objective_names != names:
                raise ValueError(
                    "Normalizer state objective names do not match the objective schema."
                )
            self._validate_normalization_state(candidate_state)
            transformed = self.objective_set.normalizer.transform(
                valid_values,
                candidate_state,
            )
            normalized_values = self._validated_matrix(
                transformed,
                row_count=len(valid_values),
                column_count=len(names),
                label="Normalizer.transform()",
            )
            next_state = candidate_state
            batch_state = candidate_state

        normalized_iterator = iter(normalized_values or ())
        items: list[EvaluatedIndividual] = []
        for evaluation, raw, status, error in zip(
            evaluations, raw_values, statuses, errors, strict=True
        ):
            if raw is None:
                items.append(
                    EvaluatedIndividual(
                        evaluation=evaluation,
                        objective_values=None,
                        status=status,
                        error=error,
                    )
                )
                continue

            minimize, maximize = self._orient(raw)
            normalized_minimize: tuple[float, ...] | None = None
            normalized_maximize: tuple[float, ...] | None = None
            if normalized_values is not None:
                normalized = next(normalized_iterator)
                normalized_minimize, normalized_maximize = self._orient_normalized(
                    normalized,
                    batch_state,
                )
            scalar_input = (
                normalized_maximize
                if normalized_maximize is not None
                else maximize
            )
            aggregate_score = (
                self._validated_scalar(
                    self.objective_set.scalarizer.scalarize(names, scalar_input)
                )
                if self.objective_set.scalarizer is not None
                else None
            )
            items.append(
                EvaluatedIndividual(
                    evaluation=evaluation,
                    objective_values=ObjectiveValues(
                        names=names,
                        raw=raw,
                        minimize=minimize,
                        maximize=maximize,
                        normalized_minimize=normalized_minimize,
                        normalized_maximize=normalized_maximize,
                        aggregate_score=aggregate_score,
                    ),
                    status=status,
                )
            )

        batch = EvaluatedBatch(
            items=tuple(items),
            objective_names=names,
            batch_index=batch_index,
            normalization_state=batch_state,
        )
        self._normalization_state = next_state
        return batch

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return JSON-compatible immutable runtime configuration."""

        return dict(self.objective_set.checkpoint_signature())

    def checkpoint_state(self) -> Mapping[str, Any]:
        """Return JSON-compatible normalization state owned by this runtime."""

        return {
            "normalization": (
                dict(self._normalization_state.checkpoint_state())
                if self._normalization_state is not None
                else None
            ),
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        """Validate and restore a checkpoint without partially mutating runtime."""

        if not isinstance(state, Mapping):
            raise TypeError("objective runtime checkpoint state must be a mapping.")
        if set(state) != {"normalization"}:
            raise ValueError(
                "objective runtime checkpoint state must contain exactly "
                "'normalization'."
            )

        raw_normalization = state["normalization"]
        restored: NormalizationState | None = None
        if raw_normalization is not None:
            if self.objective_set.normalizer is None:
                raise ValueError(
                    "Cannot restore normalization state without a configured normalizer."
                )
            restored = self.objective_set.normalizer.restore_state(raw_normalization)
            if restored.objective_names != self.schema.names:
                raise ValueError(
                    "Checkpoint objective names do not match the configured schema."
                )
            self._validate_normalization_state(restored)
        self._normalization_state = restored

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Compatibility alias for checkpoint-aware framework components."""

        self.restore(state)

    def _orient(
        self,
        values: Sequence[float],
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        minimize: list[float] = []
        maximize: list[float] = []
        for value, direction in zip(values, self.schema.directions, strict=True):
            if direction is OptimizationDirection.MINIMIZE:
                minimize.append(float(value))
                maximize.append(-float(value))
            else:
                minimize.append(-float(value))
                maximize.append(float(value))
        return tuple(minimize), tuple(maximize)

    def _orient_normalized(
        self,
        values: Sequence[float],
        state: NormalizationState | None,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        assert state is not None
        minimize: list[float] = []
        maximize: list[float] = []
        for value, direction, is_constant in zip(
            values,
            self.schema.directions,
            state.constant_mask,
            strict=True,
        ):
            if is_constant:
                minimize.append(0.0)
                maximize.append(0.0)
            elif direction is OptimizationDirection.MINIMIZE:
                minimize.append(float(value))
                maximize.append(1.0 - float(value))
            else:
                minimize.append(1.0 - float(value))
                maximize.append(float(value))
        return tuple(minimize), tuple(maximize)

    @staticmethod
    def _validate_normalization_state(state: NormalizationState) -> None:
        names = _validated_names(state.objective_names)
        if (
            isinstance(state.version, bool)
            or not isinstance(state.version, Integral)
            or state.version < 0
        ):
            raise ValueError("Normalization state version must be non-negative.")
        constant_mask = tuple(state.constant_mask)
        if len(constant_mask) != len(names) or any(
            not isinstance(value, bool) for value in constant_mask
        ):
            raise ValueError(
                "Normalization state constant mask must match objective names."
            )

    @staticmethod
    def _objective_value(objective: "Objective", evaluation: Evaluation) -> float:
        value = objective.value(evaluation)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ObjectiveError(
                f"Objective {objective.name!r} must produce a numeric value."
            )
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise ObjectiveError(
                f"Objective {objective.name!r} produced a non-finite value."
            ) from exc
        if not isfinite(numeric):
            raise ObjectiveError(
                f"Objective {objective.name!r} produced a non-finite value."
            )
        return numeric

    @staticmethod
    def _validated_matrix(
        values: object,
        *,
        row_count: int,
        column_count: int,
        label: str,
    ) -> tuple[tuple[float, ...], ...]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{label} must return a sequence of rows.")
        rows = tuple(values)
        if len(rows) != row_count:
            raise ValueError(
                f"{label} must return the exact finite {row_count}x{column_count} "
                f"matrix; got {len(rows)} rows."
            )
        validated: list[tuple[float, ...]] = []
        for row_index, row in enumerate(rows):
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                raise TypeError(f"{label} row {row_index} must be a sequence.")
            raw_row = tuple(row)
            if len(raw_row) != column_count:
                raise ValueError(
                    f"{label} must return the exact finite {row_count}x{column_count} "
                    f"matrix; row {row_index} has {len(raw_row)} columns."
                )
            validated.append(
                tuple(
                    ObjectiveRuntime._validated_scalar(
                        value,
                        label=f"{label}[{row_index}][{column}]",
                    )
                    for column, value in enumerate(raw_row)
                )
            )
        return tuple(validated)

    @staticmethod
    def _validated_scalar(value: object, *, label: str = "Scalarizer result") -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{label} must be a finite real number.")
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{label} must be finite.") from exc
        if not isfinite(numeric):
            raise ValueError(f"{label} must be finite.")
        return numeric


def _validated_names(
    names: object,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise TypeError("Objective names must be a sequence of strings.")
    normalized = tuple(names)
    if not allow_empty and not normalized:
        raise ValueError("Objective names must not be empty.")
    if any(not isinstance(name, str) or not name.strip() for name in normalized):
        raise TypeError("Objective names must be non-empty strings.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Objective names must be unique.")
    return normalized


def _validated_values(
    values: object,
    length: int,
    label: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of finite real numbers.")
    normalized = tuple(
        _finite_float(value, f"{label}[{index}]")
        for index, value in enumerate(values)
    )
    if len(normalized) != length:
        raise ValueError(
            f"{label} has {len(normalized)} values; expected {length}."
        )
    return normalized


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite real number.")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not isfinite(numeric):
        raise ValueError(f"{label} must be finite.")
    return numeric


__all__ = [
    "EvaluatedBatch",
    "EvaluatedIndividual",
    "ObjectiveEvaluationStatus",
    "ObjectiveRuntime",
    "ObjectiveValues",
]
