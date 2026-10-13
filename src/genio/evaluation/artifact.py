"""Artifacts exchanged between evaluation steps."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Artifact(Generic[T]):
    """Small data and named files produced by an evaluation step.

    Generated file paths belong in ``files``, not ``data``, so they can be
    relocated when an output is restored from the evaluation cache.
    """

    data: T | None = None
    files: Mapping[str, Path] = field(default_factory=dict)


__all__ = ["Artifact"]
