from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class StageChoice:
    """Concrete stage selected for one pipeline slot."""

    slot: int
    stage: str
    parameters: dict[str, Any] = field(default_factory=dict)
    wrapper_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Individual:
    """Concrete pipeline configuration sampled from a search scenario."""

    id: str
    scenario: str
    slots: tuple[StageChoice, ...]
    genotype: tuple[int, ...] | None = None
    search_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_slots(
        cls,
        id: str,
        scenario: str,
        slots: list[StageChoice] | tuple[StageChoice, ...],
        genotype: tuple[int, ...] | None = None,
        search_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Individual":
        return cls(
            id=id,
            scenario=scenario,
            slots=tuple(slots),
            genotype=genotype,
            search_index=search_index,
            metadata=metadata or {},
        )

    def stage_sequence(self) -> tuple[str, ...]:
        return tuple(choice.stage for choice in self.slots)

    def parameters_by_slot(self) -> dict[int, dict[str, Any]]:
        return {choice.slot: choice.parameters for choice in self.slots}
