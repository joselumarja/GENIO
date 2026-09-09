"""Artifact contracts used to exchange data between evaluation steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import Any, Sequence


class ArtifactError(Exception):
    """Base error for artifact handling failures."""


@dataclass(frozen=True, slots=True)
class Artifact(ABC):
    """Base interface for data produced and consumed by evaluation steps.

    Attributes:
        name: Name local to the producing step. The executor exposes the artifact
            under the qualified key ``step_id.name``.
        producer: Identifier of the task or component that created the artifact.
        individual_id: Candidate to which the artifact is currently bound.
        objective: Optional application-defined objective association.
        metadata: Provenance and backend-specific information.

    Note:
        Concrete artifacts are frozen dataclasses so that cache entries can clone
        and rebind them with :func:`dataclasses.replace`.
    """

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

    def for_individual(
        self,
        individual_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Artifact":
        """Clone this artifact for another individual without moving its payload.

        Cache hits use this method to give a logically equivalent artifact to a
        different individual. Referenced files and other payload locations remain
        unchanged; only identity and metadata are rebound.

        Args:
            individual_id: Identifier that should own the cloned artifact.
            metadata: Values merged over a deep copy of the original metadata.

        Returns:
            A deep-cloned artifact bound to ``individual_id``.

        Raises:
            ArtifactError: If the concrete dataclass cannot be replaced using the
                base artifact fields.
        """

        cloned = deepcopy(self)
        try:
            return replace(
                cloned,
                individual_id=individual_id,
                metadata={
                    **deepcopy(cloned.metadata),
                    **deepcopy(dict(metadata or {})),
                },
            )
        except TypeError as exc:
            raise ArtifactError(
                f"Artifact type {type(self).__name__} cannot be rebound for caching."
            ) from exc


@dataclass(frozen=True, slots=True)
class MetricArtifact(Artifact, ABC):
    """Artifact that exposes numeric metrics for result composition.

    Metric names are local to the producing step. The executor prefixes each name
    with ``step_id.`` and rejects booleans or non-real values before constructing
    a :class:`genio.Result`.
    """

    @abstractmethod
    def metrics(self) -> Mapping[str, float]:
        """Return numeric metrics exposed by this artifact.

        Returns:
            A mapping from step-local metric names to real numeric values.
        """


__all__ = ["Artifact", "ArtifactError", "MetricArtifact"]
