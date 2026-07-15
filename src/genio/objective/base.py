from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from numbers import Real

from genio.core.evaluation import Evaluation


class ObjectiveError(Exception):
    """Raised when an objective cannot evaluate an evaluation."""


class OptimizationDirection(str, Enum):
    """Directions supported when optimizing an objective."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class Objective(ABC):
    """Interprets one numeric objective from an evaluation."""

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

    def score(self, evaluation: Evaluation) -> float:
        """Return a score normalized so that larger values are better."""
        value = self.value(evaluation)
        if self.direction is OptimizationDirection.MAXIMIZE:
            return value
        return -value


@dataclass(frozen=True, slots=True)
class MetricObjective(Objective):
    """Objective backed by one metric in Result.metrics."""

    metric: str
    optimization_direction: OptimizationDirection
    id: str | None = None

    @property
    def name(self) -> str:
        """Return the configured identifier or metric name."""
        return self.id or self.metric

    @property
    def direction(self) -> OptimizationDirection:
        """Return the configured optimization direction."""
        return self.optimization_direction

    def value(self, evaluation: Evaluation) -> float:
        """Return the configured metric as a floating-point value."""
        try:
            value = evaluation.result.metrics[self.metric]
        except KeyError as exc:
            msg = f"Metric {self.metric!r} is not available in evaluation result."
            raise ObjectiveError(msg) from exc
        if isinstance(value, bool) or not isinstance(value, Real):
            msg = f"Metric {self.metric!r} must be numeric, got {value!r}."
            raise ObjectiveError(msg)
        return float(value)


@dataclass(frozen=True, slots=True)
class ObjectiveSet:
    """Ordered collection of objectives for multi-objective algorithms."""

    objectives: tuple[Objective, ...]

    def __post_init__(self) -> None:
        if not self.objectives:
            raise ObjectiveError("ObjectiveSet requires at least one objective.")
        names = [objective.name for objective in self.objectives]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            msg = f"Duplicate objective names: {duplicate_names!r}."
            raise ObjectiveError(msg)

    def values(self, evaluation: Evaluation) -> dict[str, float]:
        """Return objective values keyed by objective name."""
        return {
            objective.name: objective.value(evaluation)
            for objective in self.objectives
        }

    def scores(self, evaluation: Evaluation) -> dict[str, float]:
        """Return normalized objective scores keyed by objective name."""
        return {
            objective.name: objective.score(evaluation)
            for objective in self.objectives
        }


def dominates(
    left: Evaluation,
    right: Evaluation,
    objectives: ObjectiveSet,
) -> bool:
    """Return whether left Pareto-dominates right for the given objectives."""

    left_is_strictly_better = False
    for objective in objectives.objectives:
        left_value = objective.value(left)
        right_value = objective.value(right)
        if objective.direction is OptimizationDirection.MAXIMIZE:
            if left_value < right_value:
                return False
            if left_value > right_value:
                left_is_strictly_better = True
        else:
            if left_value > right_value:
                return False
            if left_value < right_value:
                left_is_strictly_better = True
    return left_is_strictly_better
