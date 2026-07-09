from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from genio.core.evaluation import Evaluation
from genio.core.individual import Individual


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Final result of an optimization session."""

    session_id: str
    evaluations: tuple[Evaluation, ...]
    best_individuals: tuple[Individual, ...] = ()
    statistics: dict[str, Any] = field(default_factory=dict)
