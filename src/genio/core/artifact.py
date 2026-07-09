from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Sequence


class ArtifactError(Exception):
    """Base error for artifact handling failures."""


@dataclass(frozen=True, slots=True)
class Artifact(ABC):
    """Base interface for artifacts produced and consumed by evaluators."""

    name: str
    producer: str
    individual_id: str
    _: KW_ONLY
    objective: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def load(self) -> Sequence[Any]:
        """Load and return the objects referenced by this artifact.

        Subclasses decide how references are resolved. For example, an artifact
        can point to local or remote files and use fsspec internally to load
        them as in-memory objects, parsed reports, images, or file-like handles.
        """


@dataclass(frozen=True, slots=True)
class MetricArtifact(Artifact, ABC):
    """Artifact that exposes numeric metrics for result composition."""

    @abstractmethod
    def metrics(self) -> Mapping[str, float]:
        """Return normalized numeric metrics exposed by this artifact."""
