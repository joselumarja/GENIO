"""Thread-safe LFU artifact cache with per-step capacities and telemetry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any

from genio.artifacts import Artifact
from genio.cache.base import ArtifactCache, CacheEntry


@dataclass(slots=True)
class _NamespaceStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    coalesced: int = 0
    bypasses: int = 0


class LFUArtifactCache(ArtifactCache):
    """Bound artifact entries by namespace using LFU with an LRU tie-breaker.

    Each workflow step normally forms one namespace. Eviction first chooses the
    entry with the smallest logical read count, then the oldest access sequence,
    and finally the lexicographically smallest key for deterministic ties.
    """

    def __init__(
        self,
        capacities: Mapping[str, int] | None = None,
        *,
        default_capacity: int = 0,
    ) -> None:
        """Configure per-namespace entry limits.

        Args:
            capacities: Explicit capacities keyed by namespace.
            default_capacity: Limit used for namespaces not listed explicitly. A
                value of zero disables caching for those namespaces.

        Raises:
            ValueError: If any capacity is negative.
        """
        if default_capacity < 0:
            raise ValueError("default_capacity cannot be negative.")
        normalized_capacities = dict(capacities or {})
        if any(capacity < 0 for capacity in normalized_capacities.values()):
            raise ValueError("Cache capacities cannot be negative.")
        self.capacities = normalized_capacities
        self.default_capacity = default_capacity
        self._entries: dict[str, dict[str, CacheEntry]] = {}
        self._stats: dict[str, _NamespaceStats] = {}
        self._access_sequence = 0
        self._lock = Lock()

    def capacity(self, namespace: str) -> int:
        """Return the configured entry limit for a namespace."""

        return self.capacities.get(namespace, self.default_capacity)

    def get(self, namespace: str, key: str, *, reads: int = 1) -> CacheEntry | None:
        """Read an entry and increase both logical hits and LFU frequency."""

        if reads <= 0:
            raise ValueError("reads must be positive.")
        with self._lock:
            stats = self._namespace_stats(namespace)
            entry = self._entries.get(namespace, {}).get(key)
            if entry is None:
                stats.misses += 1
                return None
            self._access_sequence += 1
            updated = replace(
                entry,
                read_count=entry.read_count + reads,
                last_access=self._access_sequence,
            )
            self._entries[namespace][key] = updated
            stats.hits += reads
            return updated

    def put(
        self,
        namespace: str,
        key: str,
        artifacts: Sequence[Artifact],
        *,
        source_individual_id: str,
        initial_reads: int = 1,
    ) -> CacheEntry:
        """Insert artifacts and evict the least-frequent, least-recent entry.

        Artifacts are deep-copied when first stored. If ``key`` already exists,
        the original entry is returned unchanged. ``initial_reads`` accounts for
        all equivalent requests coalesced into the representative execution.
        """

        if initial_reads <= 0:
            raise ValueError("initial_reads must be positive.")
        with self._lock:
            capacity = self.capacity(namespace)
            if capacity <= 0:
                raise ValueError(f"Cache namespace {namespace!r} is disabled.")
            entries = self._entries.setdefault(namespace, {})
            stats = self._namespace_stats(namespace)
            existing = entries.get(key)
            if existing is not None:
                return existing

            if len(entries) >= capacity:
                victim = min(
                    entries.values(),
                    key=lambda entry: (
                        entry.read_count,
                        entry.last_access,
                        entry.key,
                    ),
                )
                del entries[victim.key]
                stats.evictions += 1

            self._access_sequence += 1
            entry = CacheEntry(
                namespace=namespace,
                key=key,
                artifacts=tuple(deepcopy(tuple(artifacts))),
                source_individual_id=source_individual_id,
                read_count=initial_reads,
                last_access=self._access_sequence,
            )
            entries[key] = entry
            stats.stores += 1
            stats.coalesced += max(0, initial_reads - 1)
            return entry

    def record_bypass(self, namespace: str, *, count: int = 1) -> None:
        """Record requests that intentionally skipped caching."""

        if count < 0:
            raise ValueError("count cannot be negative.")
        with self._lock:
            self._namespace_stats(namespace).bypasses += count

    def clear(self) -> None:
        """Remove all entries and reset counters."""

        with self._lock:
            self._entries.clear()
            self._stats.clear()
            self._access_sequence = 0

    def snapshot(self) -> dict[str, Any]:
        """Return aggregate and per-namespace LFU telemetry.

        ``hits`` count reads served by entries that already existed, while
        ``coalesced`` counts same-wave executions avoided before a new entry was
        stored. ``executions_avoided`` combines both values.
        """

        with self._lock:
            namespaces = sorted(
                set(self.capacities) | set(self._entries) | set(self._stats)
            )
            per_namespace = {
                namespace: self._namespace_snapshot(namespace)
                for namespace in namespaces
            }
            totals: dict[str, int | float] = {
                name: sum(int(values[name]) for values in per_namespace.values())
                for name in (
                    "entries",
                    "hits",
                    "misses",
                    "stores",
                    "evictions",
                    "coalesced",
                    "bypasses",
                )
            }
            requests = totals["hits"] + totals["misses"] + totals["coalesced"]
            totals["executions_avoided"] = totals["hits"] + totals["coalesced"]
            totals["hit_rate"] = (
                totals["executions_avoided"] / requests if requests else 0.0
            )
            return {"totals": totals, "namespaces": per_namespace}

    def _namespace_stats(self, namespace: str) -> _NamespaceStats:
        return self._stats.setdefault(namespace, _NamespaceStats())

    def _namespace_snapshot(self, namespace: str) -> dict[str, int | float]:
        stats = self._stats.get(namespace, _NamespaceStats())
        entries = self._entries.get(namespace, {})
        requests = stats.hits + stats.misses + stats.coalesced
        avoided = stats.hits + stats.coalesced
        return {
            "capacity": self.capacity(namespace),
            "entries": len(entries),
            "hits": stats.hits,
            "misses": stats.misses,
            "stores": stats.stores,
            "evictions": stats.evictions,
            "coalesced": stats.coalesced,
            "bypasses": stats.bypasses,
            "executions_avoided": avoided,
            "hit_rate": avoided / requests if requests else 0.0,
        }


__all__ = ["LFUArtifactCache"]
