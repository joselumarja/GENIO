"""Scalarization strategies for ordered objective values."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genio.objective.base import ObjectiveSchema


class Scalarizer(ABC):
    """Convert one or more ordered objective values into a scalar score."""

    __slots__ = ()

    def validate(self, schema: ObjectiveSchema) -> None:
        """Validate generic compatibility with an objective schema.

        Subclasses only need to override this method when they impose additional
        schema requirements.
        """

        _validate_schema(schema)

    @abstractmethod
    def scalarize(
        self,
        objective_names: Sequence[str],
        values: Sequence[float],
    ) -> float:
        """Return one finite scalar for the supplied objective values."""

        raise NotImplementedError

    def checkpoint_signature(self) -> Mapping[str, object]:
        """Reject implicit signatures for configurable extensions."""

        raise NotImplementedError(
            f"{type(self).__qualname__} must implement checkpoint_signature()."
        )


@dataclass(frozen=True, slots=True)
class WeightedMeanScalarizer(Scalarizer):
    """Compute a weighted mean, using equal weights when none are configured."""

    weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.weights is None:
            return
        if not isinstance(self.weights, Mapping):
            raise TypeError("weights must be a mapping from objective names to weights.")

        copied_weights = dict(self.weights)
        if not copied_weights:
            raise ValueError("weights must contain at least one objective.")
        if any(not isinstance(name, str) or not name.strip() for name in copied_weights):
            raise TypeError("Weight names must be non-empty strings.")

        numeric_weights: dict[str, float] = {}
        for name, weight in copied_weights.items():
            if isinstance(weight, bool) or not isinstance(weight, Real):
                raise TypeError(f"Weight for objective {name!r} must be a real number.")
            try:
                numeric_weight = float(weight)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"Weight for objective {name!r} must be finite.") from exc
            if not isfinite(numeric_weight):
                raise ValueError(f"Weight for objective {name!r} must be finite.")
            if numeric_weight < 0.0:
                raise ValueError(f"Weight for objective {name!r} cannot be negative.")
            numeric_weights[name] = numeric_weight

        maximum = max(numeric_weights.values())
        if maximum == 0.0:
            raise ValueError("At least one weight must be greater than zero.")
        scaled_weights = {
            name: weight / maximum for name, weight in numeric_weights.items()
        }
        total = fsum(scaled_weights.values())
        normalized = {name: weight / total for name, weight in scaled_weights.items()}
        object.__setattr__(self, "weights", MappingProxyType(normalized))

    def validate(self, schema: ObjectiveSchema) -> None:
        """Require explicit weights to exactly match objective names."""

        _validate_schema(schema)
        if self.weights is None:
            return
        supplied = set(schema.names)
        configured = set(self.weights)
        if supplied != configured:
            missing = sorted(supplied - configured)
            unexpected = sorted(configured - supplied)
            raise ValueError(
                "Weights must exactly match objective names; "
                f"missing={missing!r}, unexpected={unexpected!r}."
            )

    def scalarize(
        self,
        objective_names: Sequence[str],
        values: Sequence[float],
    ) -> float:
        """Return the finite weighted or equal arithmetic mean."""

        names, numeric_values = _validate_inputs(objective_names, values)
        if not names:
            raise ValueError("WeightedMeanScalarizer requires at least one objective.")

        if self.weights is None:
            normalized_weights = (1.0 / len(names),) * len(names)
        else:
            supplied = set(names)
            configured = set(self.weights)
            if supplied != configured:
                missing = sorted(supplied - configured)
                unexpected = sorted(configured - supplied)
                raise ValueError(
                    "Weights must exactly match objective names; "
                    f"missing={missing!r}, unexpected={unexpected!r}."
                )
            normalized_weights = tuple(self.weights[name] for name in names)

        scale = max(abs(value) for value in numeric_values)
        if scale == 0.0:
            return 0.0
        result = (
            fsum(
                weight * (value / scale)
                for weight, value in zip(
                    normalized_weights, numeric_values, strict=True
                )
            )
            * scale
        )
        if not isfinite(result):
            raise ValueError("Scalarization result must be finite.")
        return result

    def checkpoint_signature(self) -> Mapping[str, object]:
        """Return the complete scalarizer configuration."""

        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "weights": dict(self.weights) if self.weights is not None else None,
        }


@dataclass(frozen=True, slots=True)
class IdentityScalarizer(Scalarizer):
    """Return the value of exactly one objective unchanged."""

    def validate(self, schema: ObjectiveSchema) -> None:
        """Require a schema containing exactly one objective."""

        _validate_schema(schema)
        if len(schema) != 1:
            raise ValueError("IdentityScalarizer requires exactly one objective.")

    def scalarize(
        self,
        objective_names: Sequence[str],
        values: Sequence[float],
    ) -> float:
        """Return the sole finite objective value."""

        names, numeric_values = _validate_inputs(objective_names, values)
        if len(names) != 1:
            raise ValueError("IdentityScalarizer requires exactly one objective.")
        return numeric_values[0]

    def checkpoint_signature(self) -> Mapping[str, object]:
        """Return the complete identity scalarizer configuration."""

        return {"type": f"{type(self).__module__}.{type(self).__qualname__}"}


def _validate_schema(schema: ObjectiveSchema) -> None:
    from genio.objective.base import ObjectiveSchema

    if not isinstance(schema, ObjectiveSchema):
        raise TypeError("schema must be an ObjectiveSchema.")


def _validated_names(objective_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(objective_names, (str, bytes)) or not isinstance(
        objective_names, Sequence
    ):
        raise TypeError("objective_names must be a sequence of strings.")
    names = tuple(objective_names)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise TypeError("Objective names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("Objective names must be unique.")
    return names


def _validated_values(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("values must be a sequence of finite real numbers.")
    validated: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Objective values must be real numbers, got {value!r}.")
        try:
            numeric_value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                f"Objective values must be finite, got {value!r}."
            ) from exc
        if not isfinite(numeric_value):
            raise ValueError(f"Objective values must be finite, got {value!r}.")
        validated.append(numeric_value)
    return tuple(validated)


def _validate_inputs(
    objective_names: Sequence[str],
    values: Sequence[float],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    names = _validated_names(objective_names)
    numeric_values = _validated_values(values)
    if len(names) != len(numeric_values):
        raise ValueError(
            "objective_names and values must contain the same number of items."
        )
    return names, numeric_values


__all__ = ["IdentityScalarizer", "Scalarizer", "WeightedMeanScalarizer"]
