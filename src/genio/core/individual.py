"""Immutable domain models used to describe search-space candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class StageChoice:
    """Describe the concrete stage selected for one pipeline slot.

    Attributes:
        slot: Zero-based position occupied by the stage in the pipeline.
        stage: Identifier used to resolve the stage definition and implementation.
        parameters: Concrete values selected from the stage parameter space.
        wrapper_inputs: Backend-specific values consumed while wrapping the stage.

    Note:
        The dataclass is frozen, but the parameter mappings are not deeply
        immutable. Treat them as read-only after construction.
    """

    slot: int
    stage: str
    parameters: dict[str, Any] = field(default_factory=dict)
    wrapper_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Individual:
    """Represent one concrete candidate from a search scenario.

    Attributes:
        id: Identity of this materialized candidate. Equivalent genotypes may
            have different identifiers when proposed more than once.
        scenario: Identifier of the search scenario that owns the candidate.
        values: Concrete selections by domain. ``pipeline``, when present, is an
            ordered tuple of :class:`StageChoice`; other domains map parameter
            names to their selected values.
        genotype: Mixed-radix genes used to construct the candidate, when known.
        search_index: Stable mixed-radix index within the scenario, when known.
        metadata: Provenance supplied by algorithms or application code.

    Note:
        Instances are frozen, but nested mappings remain mutable and should be
        treated as read-only snapshots.
    """

    id: str
    scenario: str
    values: dict[str, Any] = field(default_factory=dict)
    genotype: tuple[int, ...] | None = None
    search_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
