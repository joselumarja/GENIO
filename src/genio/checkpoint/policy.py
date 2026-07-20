from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    """Configure periodic checkpoint storage and optional session restoration."""

    directory: Path
    every_batches: int = 1
    keep_last: int = 3
    save_on_completion: bool = True
    resume_from: Path | None = None
    strict: bool = True
    compatibility_tag: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if self.resume_from is not None:
            object.__setattr__(self, "resume_from", Path(self.resume_from))
        if (
            isinstance(self.every_batches, bool)
            or not isinstance(self.every_batches, int)
            or self.every_batches <= 0
        ):
            raise ValueError("every_batches must be a positive integer.")
        if (
            isinstance(self.keep_last, bool)
            or not isinstance(self.keep_last, int)
            or self.keep_last <= 0
        ):
            raise ValueError("keep_last must be a positive integer.")


__all__ = ["CheckpointPolicy"]
