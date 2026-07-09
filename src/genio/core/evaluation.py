from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from genio.core.individual import Individual
from genio.core.result import Result


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Raw backend result associated with the individual that produced it."""

    individual: Individual
    result: Result
    metadata: dict[str, Any] = field(default_factory=dict)
