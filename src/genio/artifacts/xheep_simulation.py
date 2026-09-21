from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import shutil
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

    def for_cache(self, target_dir: str | Path) -> "XHeepSimulationArtifact":
        """Copy simulation logs into cache-owned storage and rebind paths."""

        root = Path(target_dir) / "logs"
        if self.log_paths:
            root.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        names: set[str] = set()
        for source in self.log_paths:
            source = Path(source)
            if not source.is_file():
                raise FileNotFoundError(f"Simulation log does not exist: {source}")
            if source.name in names:
                raise ValueError(
                    f"Simulation log names must be unique: {source.name!r}"
                )
            names.add(source.name)
            target = root / source.name
            shutil.copy2(source, target)
            copied.append(target)
        metadata = {
            key: deepcopy(value)
            for key, value in self.metadata.items()
            if key not in {"path", "checkout_dir"}
        }
        descriptor = Path(target_dir) / "artifact.json"
        descriptor.parent.mkdir(parents=True, exist_ok=True)
        descriptor.write_text(
            json.dumps(
                {
                    "log_paths": [str(path) for path in copied],
                    "metrics": dict(self.values),
                    "metadata": metadata,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        metadata["path"] = str(descriptor)
        return replace(
            deepcopy(self),
            log_paths=tuple(copied),
            metadata=metadata,
        )


__all__ = ["XHeepSimulationArtifact"]
