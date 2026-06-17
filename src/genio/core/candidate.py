from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Candidate:
    """Configuration to evaluate inside a search space."""

    id: str
    parameters: dict[str, Any] = field(default_factory=dict)
