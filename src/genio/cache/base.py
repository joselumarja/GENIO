from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from genio.artifacts import Artifact


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Store one successful artifact bundle and its LFU access metadata."""

    namespace: str
    key: str
    artifacts: tuple[Artifact, ...]
    source_individual_id: str
    read_count: int
    last_access: int

    def artifacts_for(
        self,
        individual_id: str,
        *,
        coalesced: bool = False,
    ) -> list[Artifact]:
        """Clone cached artifacts for another individual and mark their origin."""

        return [
            artifact.for_individual(
                individual_id,
                metadata={
                    "cache_hit": True,
                    "cache_key": self.key,
                    "cache_namespace": self.namespace,
                    "cache_source_individual_id": self.source_individual_id,
                    "cache_coalesced": coalesced,
                },
            )
            for artifact in self.artifacts
        ]


class ArtifactCache(ABC):
    """Session-scoped cache for successful artifact bundles."""

    @staticmethod
    def build_key(namespace: str, inputs: Mapping[str, Any]) -> str:
        """Build a deterministic SHA-256 key from a namespace and semantic inputs."""

        payload = json.dumps(
            {"namespace": namespace, "inputs": inputs},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @abstractmethod
    def capacity(self, namespace: str) -> int:
        """Return the configured number of entries for a namespace."""

        raise NotImplementedError

    @abstractmethod
    def get(self, namespace: str, key: str, *, reads: int = 1) -> CacheEntry | None:
        """Return and account for a cached entry, or record one miss."""

        raise NotImplementedError

    @abstractmethod
    def put(
        self,
        namespace: str,
        key: str,
        artifacts: Sequence[Artifact],
        *,
        source_individual_id: str,
        initial_reads: int = 1,
    ) -> CacheEntry:
        """Store a successful artifact bundle and evict according to policy."""

        raise NotImplementedError

    @abstractmethod
    def record_bypass(self, namespace: str, *, count: int = 1) -> None:
        """Record logical requests made by tasks that did not use this cache."""

        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Discard every cached artifact bundle and reset telemetry."""

        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return cache capacity, occupancy and access telemetry."""

        raise NotImplementedError


__all__ = ["ArtifactCache", "CacheEntry"]
