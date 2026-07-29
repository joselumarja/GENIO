from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from genio.artifacts.base import MetricArtifact


@dataclass(frozen=True, slots=True)
class XHeepSimulationArtifact(MetricArtifact):
    """Metrics and output files produced by an X-HEEP Verilator execution."""

    log_paths: tuple[Path, ...] = ()
    values: Mapping[str, float] = field(default_factory=dict)

    def load(self) -> Sequence[Any]:
        """Return simulation logs and parsed metrics."""

        return (self.log_paths, self.values)

    def metrics(self) -> Mapping[str, float]:
        """Return metrics prefixed by their simulation origin."""

        return {f"xheep_verilator.{key}": value for key, value in self.values.items()}


__all__ = ["XHeepSimulationArtifact"]
