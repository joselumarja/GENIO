from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from genio.core import StageChoice


@dataclass(frozen=True, slots=True)
class SlotSpec:
    """Valid concrete alternatives available for one pipeline slot."""

    index: int
    alternatives: tuple[StageChoice, ...]

    @property
    def stage_groups(self) -> tuple[tuple[int, ...], ...]:
        """Alternative indexes grouped by stage, preserving first-seen order."""
        groups: dict[str, list[int]] = {}
        for alternative_index, alternative in enumerate(self.alternatives):
            groups.setdefault(alternative.stage, []).append(alternative_index)
        return tuple(tuple(indexes) for indexes in groups.values())


@dataclass(frozen=True, slots=True)
class SearchScenarioSpec:
    """Finite search-space description for one scenario."""

    id: str
    slots: tuple[SlotSpec, ...]
    design_spaces: dict[str, dict[str, tuple[Any, ...]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
