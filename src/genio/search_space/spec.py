from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from genio.core import StageChoice


@dataclass(frozen=True, slots=True)
class SlotSpec:
    """Valid concrete alternatives available for one pipeline slot."""

    index: int
    alternatives: tuple[StageChoice, ...]


@dataclass(frozen=True, slots=True)
class SearchScenarioSpec:
    """Finite search-space description for one scenario."""

    id: str
    slots: tuple[SlotSpec, ...]
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
