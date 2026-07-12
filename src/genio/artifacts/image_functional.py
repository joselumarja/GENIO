from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from genio.artifacts.base import MetricArtifact


@dataclass(frozen=True, slots=True)
class ImageFunctionalMetricsArtifact(MetricArtifact):
    """Aggregated metrics produced by an image functional evaluation."""

    values: Mapping[str, float] = field(default_factory=dict)
    per_sample_values: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def load(self) -> tuple[Mapping[str, float]]:
        return (self.values,)

    def metrics(self) -> Mapping[str, float]:
        return self.values


__all__ = ["ImageFunctionalMetricsArtifact"]
