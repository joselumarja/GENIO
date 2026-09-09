"""Objective-value normalization independent from objective extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from genio.objective.base import ObjectiveSchema


class NormalizationScope(str, Enum):
    """Determine which observations define normalization ranges."""

    BATCH = "batch"
    CUMULATIVE = "cumulative"
    FIXED = "fixed"


class NormalizationState(ABC):
    """Base contract for immutable, checkpointable normalization state."""

    objective_names: tuple[str, ...]
    version: int

    @property
    def constant_mask(self) -> tuple[bool, ...]:
        """Mark objective columns whose normalization range is constant."""

        return (False,) * len(self.objective_names)

    @abstractmethod
    def checkpoint_state(self) -> Mapping[str, Any]:
        """Return JSON-compatible state owned by the concrete normalizer."""

        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class MinMaxNormalizationState(NormalizationState):
    """Store immutable, versioned ranges fitted by ``MinMaxNormalizer``."""

    objective_names: tuple[str, ...]
    minimums: tuple[float, ...]
    maximums: tuple[float, ...]
    scope: NormalizationScope
    version: int

    def __post_init__(self) -> None:
        names = _validate_objective_names(self.objective_names)
        if not isinstance(self.scope, NormalizationScope):
            raise TypeError("state scope must be a NormalizationScope.")
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, Integral)
            or self.version < 0
        ):
            raise ValueError("state version must be a non-negative integer.")
        minimums = tuple(
            _finite_float(value, f"state.minimums[{column}]")
            for column, value in enumerate(self.minimums)
        )
        maximums = tuple(
            _finite_float(value, f"state.maximums[{column}]")
            for column, value in enumerate(self.maximums)
        )
        object.__setattr__(self, "objective_names", names)
        object.__setattr__(self, "minimums", minimums)
        object.__setattr__(self, "maximums", maximums)
        object.__setattr__(self, "version", int(self.version))
        _validate_state(self)

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return the state structure used for compatibility checks."""

        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "objective_names": list(self.objective_names),
            "scope": self.scope.value,
        }

    def checkpoint_state(self) -> Mapping[str, Any]:
        """Return this fitted range as JSON-compatible data."""

        return {
            "objective_names": list(self.objective_names),
            "minimums": list(self.minimums),
            "maximums": list(self.maximums),
            "scope": self.scope.value,
            "version": self.version,
        }

    @property
    def constant_mask(self) -> tuple[bool, ...]:
        """Mark columns whose fitted minimum and maximum are equal."""

        return tuple(
            minimum == maximum
            for minimum, maximum in zip(
                self.minimums, self.maximums, strict=True
            )
        )

    @classmethod
    def restore(cls, state: Mapping[str, Any]) -> "MinMaxNormalizationState":
        """Validate and restore a normalization state from checkpoint data."""

        if not isinstance(state, Mapping):
            raise TypeError("normalization state must be a mapping.")
        expected = {"objective_names", "minimums", "maximums", "scope", "version"}
        if set(state) != expected:
            raise ValueError(
                "normalization state must contain exactly "
                f"{sorted(expected)!r}."
            )
        try:
            scope = NormalizationScope(state["scope"])
        except (TypeError, ValueError) as exc:
            raise ValueError("normalization state has an invalid scope.") from exc
        return cls(
            objective_names=_as_tuple(
                state["objective_names"],
                "normalization state objective_names must be a sequence.",
            ),
            minimums=_as_tuple(
                state["minimums"],
                "normalization state minimums must be a sequence.",
            ),
            maximums=_as_tuple(
                state["maximums"],
                "normalization state maximums must be a sequence.",
            ),
            scope=scope,
            version=state["version"],
        )


class Normalizer(ABC):
    """Fit and apply a transformation to an objective-value matrix."""

    __slots__ = ()

    def validate(self, schema: ObjectiveSchema) -> None:
        """Validate generic compatibility with an objective schema.

        Subclasses only need to override this method when they impose additional
        schema requirements.
        """

        _validate_schema(schema)

    @abstractmethod
    def fit(
        self,
        values: Sequence[Sequence[float]],
        *,
        schema: ObjectiveSchema,
        previous: NormalizationState | None = None,
    ) -> NormalizationState:
        """Return the state with which ``values`` should be transformed."""

        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        values: Sequence[Sequence[float]],
        state: NormalizationState,
    ) -> tuple[tuple[float, ...], ...]:
        """Transform ``values`` using a previously fitted state."""

        raise NotImplementedError

    @abstractmethod
    def restore_state(self, state: Mapping[str, Any]) -> NormalizationState:
        """Validate and restore this normalizer's concrete checkpoint state."""

        raise NotImplementedError

    def checkpoint_signature(self) -> Mapping[str, object]:
        """Require configurable normalizers to declare checkpoint identity."""

        raise NotImplementedError(
            f"{type(self).__qualname__} must implement checkpoint_signature()."
        )


@dataclass(frozen=True, slots=True)
class MinMaxNormalizer(Normalizer):
    """Scale each objective using configured or observed min/max ranges."""

    scope: NormalizationScope = NormalizationScope.BATCH

    def __post_init__(self) -> None:
        if not isinstance(self.scope, NormalizationScope):
            try:
                scope = NormalizationScope(self.scope)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unknown normalization scope: {self.scope!r}.") from exc
            object.__setattr__(self, "scope", scope)

    def validate(self, schema: ObjectiveSchema) -> None:
        """Reject fixed normalization when an objective has no bounds."""

        Normalizer.validate(self, schema)
        if self.scope is NormalizationScope.FIXED and any(
            bound is None for bound in schema.normalization_bounds
        ):
            raise ValueError(
                "FIXED normalization requires normalization_bounds for every objective."
            )

    def fit(
        self,
        values: Sequence[Sequence[float]],
        *,
        schema: ObjectiveSchema,
        previous: NormalizationState | None = None,
    ) -> MinMaxNormalizationState:
        """Fit ranges according to the configured normalization scope."""

        self.validate(schema)
        names = schema.names
        matrix = _validate_matrix(values, len(names))
        validated_bounds = _validate_bounds(
            schema.normalization_bounds,
            len(names),
        )
        if self.scope is NormalizationScope.FIXED and any(
            bound is None for bound in validated_bounds
        ):
            raise ValueError(
                "FIXED normalization requires normalization_bounds for every objective."
            )

        version = _next_version(previous, names, expected_scope=self.scope)
        if not matrix and any(bound is None for bound in validated_bounds):
            raise ValueError(
                "At least one value row is required for objectives without bounds."
            )

        minimums: list[float] = []
        maximums: list[float] = []
        for column, bound in enumerate(validated_bounds):
            if bound is not None:
                minimum, maximum = bound
            else:
                column_values = tuple(row[column] for row in matrix)
                minimum = min(column_values)
                maximum = max(column_values)
                if self.scope is NormalizationScope.CUMULATIVE and previous is not None:
                    assert isinstance(previous, MinMaxNormalizationState)
                    minimum = min(minimum, previous.minimums[column])
                    maximum = max(maximum, previous.maximums[column])
            minimums.append(minimum)
            maximums.append(maximum)

        return MinMaxNormalizationState(
            objective_names=names,
            minimums=tuple(minimums),
            maximums=tuple(maximums),
            scope=self.scope,
            version=version,
        )

    def transform(
        self,
        values: Sequence[Sequence[float]],
        state: NormalizationState,
    ) -> tuple[tuple[float, ...], ...]:
        """Apply min-max scaling without clipping values outside fitted ranges."""

        if not isinstance(state, MinMaxNormalizationState):
            raise TypeError("MinMaxNormalizer requires MinMaxNormalizationState.")
        _validate_state(state)
        matrix = _validate_matrix(values, len(state.objective_names))
        transformed: list[tuple[float, ...]] = []
        for row in matrix:
            normalized_row: list[float] = []
            for value, minimum, maximum in zip(
                row, state.minimums, state.maximums, strict=True
            ):
                if minimum == maximum:
                    normalized_row.append(0.0)
                    continue
                numerator = value - minimum
                denominator = maximum - minimum
                if isfinite(numerator) and isfinite(denominator):
                    normalized_row.append(numerator / denominator)
                    continue

                # Avoid intermediate overflow for valid extreme finite ranges.
                scale = max(abs(value), abs(minimum), abs(maximum))
                normalized_row.append(
                    (value / scale - minimum / scale)
                    / (maximum / scale - minimum / scale)
                )
            transformed.append(tuple(normalized_row))
        return tuple(transformed)

    def restore_state(self, state: Mapping[str, Any]) -> MinMaxNormalizationState:
        """Restore min-max ranges and require the configured scope."""

        restored = MinMaxNormalizationState.restore(state)
        if restored.scope is not self.scope:
            raise ValueError(
                "Checkpoint normalization scope does not match the normalizer."
            )
        return restored

    def checkpoint_signature(self) -> Mapping[str, object]:
        """Return the normalizer type and range scope."""

        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "scope": self.scope.value,
        }


def _as_tuple(value: object, message: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(message)
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(message) from exc


def _validate_schema(schema: ObjectiveSchema) -> None:
    from genio.objective.base import ObjectiveSchema

    if not isinstance(schema, ObjectiveSchema):
        raise TypeError("schema must be an ObjectiveSchema.")


def _validate_objective_names(objective_names: Sequence[str]) -> tuple[str, ...]:
    names = _as_tuple(
        objective_names,
        "objective_names must be a sequence of strings.",
    )
    if not names:
        raise ValueError("objective_names must contain at least one name.")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Objective names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("Objective names must be unique.")
    return names


def _validate_matrix(
    values: Sequence[Sequence[float]],
    column_count: int,
) -> tuple[tuple[float, ...], ...]:
    rows = _as_tuple(values, "values must be a sequence of rows.")
    matrix: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        raw_row = _as_tuple(row, f"values row {row_index} must be a numeric sequence.")
        if len(raw_row) != column_count:
            raise ValueError(
                f"values row {row_index} has {len(raw_row)} columns; "
                f"expected {column_count}."
            )
        matrix.append(
            tuple(
                _finite_float(value, f"values[{row_index}][{column}]")
                for column, value in enumerate(raw_row)
            )
        )
    return tuple(matrix)


def _validate_bounds(
    bounds: Sequence[tuple[float, float] | None],
    column_count: int,
) -> tuple[tuple[float, float] | None, ...]:
    raw_bounds = _as_tuple(bounds, "bounds must be a sequence of pairs or None.")
    if len(raw_bounds) != column_count:
        raise ValueError(
            f"bounds has {len(raw_bounds)} columns; expected {column_count}."
        )
    validated: list[tuple[float, float] | None] = []
    for column, bound in enumerate(raw_bounds):
        if bound is None:
            validated.append(None)
            continue
        raw_bound = _as_tuple(bound, f"bounds[{column}] must be a pair or None.")
        if len(raw_bound) != 2:
            raise ValueError(f"bounds[{column}] must contain exactly two values.")
        minimum = _finite_float(raw_bound[0], f"bounds[{column}][0]")
        maximum = _finite_float(raw_bound[1], f"bounds[{column}][1]")
        if minimum > maximum:
            raise ValueError(
                f"bounds[{column}] minimum cannot be greater than its maximum."
            )
        validated.append((minimum, maximum))
    return tuple(validated)


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number, got {value!r}.")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite, got {value!r}.") from exc
    if not isfinite(converted):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    return converted


def _validate_state(state: MinMaxNormalizationState) -> None:
    if not isinstance(state, MinMaxNormalizationState):
        raise TypeError("state must be a MinMaxNormalizationState.")
    names = _validate_objective_names(state.objective_names)
    if not isinstance(state.scope, NormalizationScope):
        raise TypeError("state scope must be a NormalizationScope.")
    if (
        isinstance(state.version, bool)
        or not isinstance(state.version, Integral)
        or state.version < 0
    ):
        raise ValueError("state version must be a non-negative integer.")
    if len(state.minimums) != len(names) or len(state.maximums) != len(names):
        raise ValueError("state ranges must match its objective columns.")
    for column, (raw_minimum, raw_maximum) in enumerate(
        zip(state.minimums, state.maximums, strict=True)
    ):
        minimum = _finite_float(raw_minimum, f"state.minimums[{column}]")
        maximum = _finite_float(raw_maximum, f"state.maximums[{column}]")
        if minimum > maximum:
            raise ValueError(
                f"state range {column} minimum cannot be greater than its maximum."
            )


def _next_version(
    previous: NormalizationState | None,
    objective_names: tuple[str, ...],
    *,
    expected_scope: NormalizationScope,
) -> int:
    if previous is None:
        return 1
    if not isinstance(previous, MinMaxNormalizationState):
        raise TypeError(
            "previous state must be a MinMaxNormalizationState."
        )
    _validate_state(previous)
    if previous.objective_names != objective_names:
        raise ValueError("previous state objective names do not match objective_names.")
    if previous.scope is not expected_scope:
        raise ValueError("previous state scope does not match the normalizer scope.")
    return int(previous.version) + 1


__all__ = [
    "MinMaxNormalizationState",
    "MinMaxNormalizer",
    "NormalizationScope",
    "NormalizationState",
    "Normalizer",
]
