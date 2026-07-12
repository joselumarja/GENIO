from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genio.composer.base import ExecutionPackage


@dataclass(frozen=True, slots=True)
class HLSExecutionPackage(ExecutionPackage):
    """Portable HLS execution package suitable for Vitis HLS synthesis."""

    entrypoint: str = "hls_config.cfg"
    files: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def materialize(self, target_dir: str | Path) -> Path:
        package_dir = Path(target_dir)
        package_dir.mkdir(parents=True, exist_ok=True)

        for relative_path, content in self.files.items():
            destination = package_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")

        if self.metadata:
            metadata_path = package_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(dict(self.metadata), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        return package_dir


__all__ = ["HLSExecutionPackage"]
