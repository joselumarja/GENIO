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
        slots: Stage choices in pipeline order.
        genotype: Mixed-radix genes used to construct the candidate, when known.
        search_index: Stable mixed-radix index within the scenario, when known.
        design: Values from non-pipeline design domains such as hardware options.
        metadata: Provenance supplied by algorithms or application code.

    Note:
        Instances are frozen, but nested mappings remain mutable and should be
        treated as read-only snapshots.
    """

    id: str
    scenario: str
    slots: tuple[StageChoice, ...]
    genotype: tuple[int, ...] | None = None
    search_index: int | None = None
    design: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_slots(
        cls,
        id: str,
        scenario: str,
        slots: list[StageChoice] | tuple[StageChoice, ...],
        genotype: tuple[int, ...] | None = None,
        search_index: int | None = None,
        design: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Individual":
        """Create an individual from an ordered collection of stage choices.

        This convenience constructor normalizes ``slots`` to a tuple and copies
        missing ``design`` and ``metadata`` values to empty mappings. It does not
        validate the candidate against a :class:`genio.SearchSpace`; use the
        search-space factory methods when validation and index calculation are
        required.

        Returns:
            A materialized individual containing the supplied configuration.
        """
        return cls(
            id=id,
            scenario=scenario,
            slots=tuple(slots),
            genotype=genotype,
            search_index=search_index,
            design=design or {},
            metadata=metadata or {},
        )

    def stage_sequence(self) -> tuple[str, ...]:
        """Return the selected stage names in slot order."""
        return tuple(choice.stage for choice in self.slots)

    def parameters_by_slot(self) -> dict[int, dict[str, Any]]:
        """Return the selected parameters keyed by slot index."""
        return {choice.slot: choice.parameters for choice in self.slots}
