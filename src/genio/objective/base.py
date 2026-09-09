"""Objective definitions and immutable transformation configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Any

from genio.core.evaluation import Evaluation
from genio.objective.normalization import Normalizer
from genio.objective.scalarization import Scalarizer

if TYPE_CHECKING:
    from genio.objective.runtime import ObjectiveRuntime


class ObjectiveError(ValueError):
    """Raised when an objective cannot evaluate an evaluation."""


class OptimizationDirection(str, Enum):
    """Directions supported when optimizing an objective."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True, slots=True)
class ObjectiveDescriptor:
    """Immutable objective metadata exposed to search algorithms."""

    name: str
    direction: OptimizationDirection
    normalization_bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ObjectiveError("Objective name must be a non-empty string.")
        if not isinstance(self.direction, OptimizationDirection):
            raise ObjectiveError("Objective direction must be an OptimizationDirection.")
        if self.normalization_bounds is not None:
            object.__setattr__(
                self,
                "normalization_bounds",
                _validated_bounds(self.normalization_bounds),
            )


@dataclass(frozen=True, slots=True)
class ObjectiveSchema:
    """Ordered objective metadata required to configure an algorithm."""

    objectives: tuple[ObjectiveDescriptor, ...]

    def __post_init__(self) -> None:
        descriptors = tuple(self.objectives)
        object.__setattr__(self, "objectives", descriptors)
        if not descriptors:
            raise ObjectiveError("ObjectiveSchema requires at least one objective.")
        if any(not isinstance(item, ObjectiveDescriptor) for item in descriptors):
            raise TypeError("ObjectiveSchema entries must be ObjectiveDescriptor instances.")
        duplicate_names = sorted(
            name for name in set(self.names) if self.names.count(name) > 1
        )
        if duplicate_names:
            raise ObjectiveError(f"Duplicate objective names: {duplicate_names!r}.")

    def __len__(self) -> int:
        return len(self.objectives)

    @property
    def names(self) -> tuple[str, ...]:
        """Return objective names in matrix-column order."""

        return tuple(objective.name for objective in self.objectives)

    @property
    def directions(self) -> tuple[OptimizationDirection, ...]:
        """Return objective directions in matrix-column order."""

        return tuple(objective.direction for objective in self.objectives)

    @property
    def normalization_bounds(self) -> tuple[tuple[float, float] | None, ...]:
        """Return optional objective bounds in matrix-column order."""

        return tuple(objective.normalization_bounds for objective in self.objectives)


class Objective(ABC):
    """Interpret one numeric optimization objective from an evaluation.

    Custom objectives define a stable name, an optimization direction, and a
    value extractor. Search algorithms use :meth:`score` when they need a common
    convention in which larger values are always better.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the objective."""

        raise NotImplementedError

    @property
    @abstractmethod
    def direction(self) -> OptimizationDirection:
        """Return the direction in which the objective is optimized."""

        raise NotImplementedError

    @abstractmethod
    def value(self, evaluation: Evaluation) -> float:
        """Extract the objective value from an evaluation."""

        raise NotImplementedError

    @property
    def normalization_bounds(self) -> tuple[float, float] | None:
        """Return optional fixed bounds used only for normalization."""

        return None

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Require configurable objectives to declare checkpoint identity."""

        raise NotImplementedError(
            f"{type(self).__qualname__} must implement checkpoint_signature()."
        )


@dataclass(frozen=True, slots=True, init=False)
class MetricObjective(Objective):
    """Read an objective from a qualified key in ``Result.metrics``.

    Args:
        metric: Exact metric key, normally ``step_id.metric_name``.
        direction: ``maximize``/``minimize`` or its enum member.
        name: Optional objective name. It defaults to ``metric``.
        normalization_bounds: Optional finite ``(lower, upper)`` bounds.
    """

    metric: str
    _direction: OptimizationDirection
    _name: str
    _normalization_bounds: tuple[float, float] | None

    def __init__(
        self,
        metric: str,
        direction: OptimizationDirection | str,
        *,
        name: str | None = None,
        normalization_bounds: tuple[float, float] | None = None,
    ) -> None:
        if not isinstance(metric, str) or not metric.strip():
            raise ObjectiveError("MetricObjective metric must be a non-empty string.")
        if not isinstance(direction, OptimizationDirection):
            try:
                direction = OptimizationDirection(direction)
            except (TypeError, ValueError) as exc:
                raise ObjectiveError(
                    f"Unknown optimization direction: {direction!r}."
                ) from exc
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ObjectiveError("MetricObjective name must be a non-empty string.")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "_direction", direction)
        object.__setattr__(self, "_name", name or metric)
        object.__setattr__(
            self,
            "_normalization_bounds",
            _validated_bounds(normalization_bounds)
            if normalization_bounds is not None
            else None,
        )

    @property
    def name(self) -> str:
        """Return the configured identifier or metric name."""

        return self._name

    @property
    def direction(self) -> OptimizationDirection:
        """Return the normalized optimization direction."""

        return self._direction

    @property
    def normalization_bounds(self) -> tuple[float, float] | None:
        """Return optional fixed bounds used by normalization strategies."""

        return self._normalization_bounds

    def value(self, evaluation: Evaluation) -> float:
        """Return a finite metric value, rejecting booleans and non-reals."""

        try:
            value = evaluation.result.metrics[self.metric]
        except KeyError as exc:
            raise ObjectiveError(
                f"Metric {self.metric!r} is not available in evaluation result."
            ) from exc
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ObjectiveError(
                f"Metric {self.metric!r} must be numeric, got {value!r}."
            )
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise ObjectiveError(
                f"Metric {self.metric!r} must be finite, got {value!r}."
            ) from exc
        if not isfinite(numeric):
            raise ObjectiveError(
                f"Metric {self.metric!r} must be finite, got {value!r}."
            )
        return numeric

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return metric lookup, direction, and optional bounds configuration."""

        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "name": self.name,
            "direction": self.direction.value,
            "metric": self.metric,
            "normalization_bounds": (
                list(self.normalization_bounds)
                if self.normalization_bounds is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ObjectiveSet:
    """Store immutable objective, normalization, and scalarization configuration."""

    objectives: Sequence[Objective]
    normalizer: Normalizer | None = None
    scalarizer: Scalarizer | None = None

    def __post_init__(self) -> None:
        if isinstance(self.objectives, (str, bytes)) or not isinstance(
            self.objectives, Sequence
        ):
            raise TypeError("objectives must be a sequence of Objective instances.")
        objectives = tuple(self.objectives)
        object.__setattr__(self, "objectives", objectives)
        if not objectives:
            raise ObjectiveError("ObjectiveSet requires at least one objective.")
        if any(not isinstance(objective, Objective) for objective in objectives):
            raise TypeError("ObjectiveSet entries must be Objective instances.")
        names: list[str] = []
        for objective in objectives:
            if not isinstance(objective.name, str) or not objective.name.strip():
                raise ObjectiveError("Objective names must be non-empty strings.")
            if not isinstance(objective.direction, OptimizationDirection):
                raise ObjectiveError(
                    "Objective directions must be OptimizationDirection values."
                )
            names.append(objective.name)
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise ObjectiveError(f"Duplicate objective names: {duplicate_names!r}.")
        if self.normalizer is not None and not isinstance(self.normalizer, Normalizer):
            raise TypeError("normalizer must be a Normalizer or None.")
        if self.scalarizer is not None and not isinstance(self.scalarizer, Scalarizer):
            raise TypeError("scalarizer must be a Scalarizer or None.")
        schema = self.schema
        if self.normalizer is not None:
            self.normalizer.validate(schema)
        if self.scalarizer is not None:
            self.scalarizer.validate(schema)

    @property
    def schema(self) -> ObjectiveSchema:
        """Return immutable objective metadata without executable extractors."""

        return ObjectiveSchema(
            tuple(
                ObjectiveDescriptor(
                    name=objective.name,
                    direction=objective.direction,
                    normalization_bounds=objective.normalization_bounds,
                )
                for objective in self.objectives
            )
        )

    def bind(self) -> "ObjectiveRuntime":
        """Create isolated transformation state for one optimization session."""

        from genio.objective.runtime import ObjectiveRuntime

        if self.normalizer is not None:
            self.normalizer.validate(self.schema)
        if self.scalarizer is not None:
            self.scalarizer.validate(self.schema)
        return ObjectiveRuntime(self)

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return JSON-compatible objective and transformation configuration."""

        return {
            "objectives": [
                dict(objective.checkpoint_signature()) for objective in self.objectives
            ],
            "normalizer": (
                dict(self.normalizer.checkpoint_signature())
                if self.normalizer is not None
                else None
            ),
            "scalarizer": (
                dict(self.scalarizer.checkpoint_signature())
                if self.scalarizer is not None
                else None
            ),
        }

def _validated_bounds(bounds: object) -> tuple[float, float]:
    if isinstance(bounds, (str, bytes)):
        raise ObjectiveError(
            "Objective normalization_bounds must contain exactly two values."
        )
    try:
        values: tuple[Any, ...] = tuple(bounds)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ObjectiveError(
            "Objective normalization_bounds must contain exactly two values."
        ) from exc
    if len(values) != 2:
        raise ObjectiveError(
            "Objective normalization_bounds must contain exactly two values."
        )
    lower, upper = values
    numeric: list[float] = []
    for value in (lower, upper):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ObjectiveError(
                "Objective normalization_bounds must be finite numeric values."
            )
        try:
            converted = float(value)
        except (OverflowError, ValueError) as exc:
            raise ObjectiveError(
                "Objective normalization_bounds must be finite numeric values."
            ) from exc
        if not isfinite(converted):
            raise ObjectiveError(
                "Objective normalization_bounds must be finite numeric values."
            )
        numeric.append(converted)
    if numeric[0] >= numeric[1]:
        raise ObjectiveError(
            "Objective normalization lower bound must be smaller than upper bound."
        )
    return numeric[0], numeric[1]


__all__ = [
    "MetricObjective",
    "Objective",
    "ObjectiveDescriptor",
    "ObjectiveError",
    "ObjectiveSchema",
    "ObjectiveSet",
    "OptimizationDirection",
]
