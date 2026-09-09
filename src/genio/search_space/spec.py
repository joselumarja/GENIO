"""Expanded, finite specifications used by :mod:`genio.search_space`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from genio.core import StageChoice


@dataclass(frozen=True, slots=True)
class SlotSpec:
    """Store all valid concrete alternatives for one pipeline slot.

    Attributes:
        index: Contiguous zero-based slot position.
        alternatives: Expanded stage choices addressable by a genotype gene.
    """

    index: int
    alternatives: tuple[StageChoice, ...]

    @property
    def stage_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return alternative indexes grouped by stage identifier.

        The outer order follows the first appearance of each stage. These groups
        let balanced samplers choose a stage uniformly before choosing one of its
        parameterized alternatives.
        """
        groups: dict[str, list[int]] = {}
        for alternative_index, alternative in enumerate(self.alternatives):
            groups.setdefault(alternative.stage, []).append(alternative_index)
        return tuple(tuple(indexes) for indexes in groups.values())


@dataclass(frozen=True, slots=True)
class SearchScenarioSpec:
    """Describe an expanded finite search scenario.

    Attributes:
        id: Stable scenario identifier used in individuals and generated IDs.
        slots: Ordered pipeline dimensions of the search space.
        design_spaces: Additional ordered domains and their allowed values. Their
            genes are appended after all slot genes.
        metadata: Scenario fields not interpreted by the search-space loader.
    """

    id: str
    slots: tuple[SlotSpec, ...]
    design_spaces: dict[str, dict[str, tuple[Any, ...]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
