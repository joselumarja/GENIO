"""Association between a proposed individual and its normalized result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from genio.core.individual import Individual
from genio.core.result import Result


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Associate an individual with the result produced by its workflow.

    Attributes:
        individual: Candidate submitted for evaluation.
        result: Success or failure normalized by the evaluation executor.
        metadata: Proposal provenance and session-level evaluation information.

    Note:
        Task processes produce evaluation outputs. The executor collects their
        metrics into the normalized ``result`` stored here.
    """

    individual: Individual
    result: Result
    metadata: dict[str, Any] = field(default_factory=dict)
