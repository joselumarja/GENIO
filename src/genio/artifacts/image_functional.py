from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field

from genio.artifacts.base import MetricArtifact


@dataclass(frozen=True, slots=True)
class ImageFunctionalMetricsArtifact(MetricArtifact):
    """Aggregated metrics produced by an image functional evaluation."""

    values: Mapping[str, float] = field(default_factory=dict)
    per_sample_values: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def load(self) -> tuple[Mapping[str, float]]:
        """Return the aggregate metric mapping as the artifact payload."""
        return (self.values,)

    def metrics(self) -> Mapping[str, float]:
        """Return the aggregate image evaluation metrics."""
        return self.values

    def for_cache(self, target_dir) -> "ImageFunctionalMetricsArtifact":
        """Return an independent copy because this artifact has no disk payload."""

        del target_dir
        return deepcopy(self)


__all__ = ["ImageFunctionalMetricsArtifact"]
