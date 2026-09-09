"""Aggregate result returned after an optimization session completes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from genio.core.evaluation import Evaluation
from genio.core.individual import Individual

@dataclass(frozen=True, slots=True)
class SearchResult:
    """Summarize a completed optimization session.

    Attributes:
        session_id: Logical identifier of the optimization configuration.
        evaluations: Committed evaluations in proposal order.
        best_individuals: Candidates selected by the search algorithm. Algorithms
            that do not implement best-candidate selection leave this empty.
        statistics: Final snapshot produced by the statistics collector.
        run_id: Identifier of this concrete execution or resumed lineage.
    """

    session_id: str
    evaluations: tuple[Evaluation, ...]
    best_individuals: tuple[Individual, ...] = ()
    statistics: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
