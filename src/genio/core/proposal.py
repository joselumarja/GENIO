from __future__ import annotations

from dataclasses import dataclass

from genio.core.individual import Individual


@dataclass(frozen=True, slots=True)
class Proposal:
    """Identify one occurrence of an individual proposed during a search run."""

    proposal_id: str
    proposal_sequence: int
    batch_index: int | None
    batch_position: int
    individual: Individual

    def evaluation_metadata(self) -> dict[str, int | str | None]:
        """Return proposal provenance suitable for evaluation metadata."""

        return {
            "proposal_id": self.proposal_id,
            "proposal_sequence": self.proposal_sequence,
            "batch_index": self.batch_index,
            "batch_position": self.batch_position,
        }


__all__ = ["Proposal"]
