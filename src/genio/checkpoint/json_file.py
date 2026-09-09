"""Durable JSON checkpoint snapshots, manifests, checksums, and lineage locks."""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .errors import CheckpointFormatError
from .policy import CheckpointPolicy


class JSONCheckpointStore:
    """Persist session checkpoint snapshots as atomically replaced JSON files.

    A save writes an immutable numbered snapshot and then atomically replaces
    ``latest.json`` with a checksummed reference to it. Directory locks serialize
    individual file operations, while the longer-lived session lease prevents two
    optimization sessions from advancing the same checkpoint lineage.

    The durability and locking implementation relies on POSIX ``fsync``, atomic
    rename semantics, file permissions, and ``fcntl`` advisory locks.
    """

    FORMAT = "genio.optimization-checkpoint"
    LATEST_FORMAT = "genio.optimization-checkpoint.latest"
    _SNAPSHOT_NAME = re.compile(
        r"^checkpoint-[0-9]{6,}-(?:running|completed)-[0-9a-f]{32}\.json$"
    )

    def __init__(self, policy: CheckpointPolicy) -> None:
        """Create a store governed by one checkpoint policy.

        Filesystem directories and locks are created lazily by save, load, or
        lease operations.
        """
        self.policy = policy
        self._lease_descriptor: int | None = None

    @property
    def latest_path(self) -> Path:
        """Return the path of the latest-checkpoint manifest."""

        return self.policy.directory / "latest.json"

    @property
    def session_lease_held(self) -> bool:
        """Return whether this store currently owns the session lineage lease."""

        return self._lease_descriptor is not None

    def should_save(self, completed_batches: int) -> bool:
        """Return whether a periodic snapshot is due after this batch count."""

        return completed_batches % self.policy.every_batches == 0

    def save(self, payload: dict[str, Any], *, sequence: int) -> Path:
        """Atomically persist one snapshot and update the latest manifest.

        Args:
            payload: JSON-compatible session state with status ``running`` or
                ``completed``.
            sequence: Completed-batch sequence embedded in the snapshot filename.

        Returns:
            Path of the immutable numbered snapshot.

        Raises:
            CheckpointFormatError: If the payload status is invalid.
            TypeError: If canonical fingerprinting encounters a non-JSON value.
            ValueError: If canonical fingerprinting encounters a non-finite number.
            OSError: If snapshot, manifest, permission, or lock operations fail.

        Note:
            This method serializes writes but does not require the caller to hold
            the longer-lived session lease.
        """

        status = str(payload.get("status"))
        if status not in {"running", "completed"}:
            raise CheckpointFormatError("Checkpoint payload has an invalid status.")
        snapshot_without_checksum = {
            "format": self.FORMAT,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        snapshot = {
            **snapshot_without_checksum,
            "content_sha256": self.fingerprint(snapshot_without_checksum),
        }
        encoded = self._encode(snapshot)
        checksum = hashlib.sha256(encoded).hexdigest()
        snapshot_path = self.policy.directory / (
            f"checkpoint-{sequence:06d}-{status}-{uuid4().hex}.json"
        )

        latest = {
            "format": self.LATEST_FORMAT,
            "checkpoint": snapshot_path.name,
            "sha256": checksum,
        }
        with self._directory_lock():
            self._write_atomic(snapshot_path, encoded)
            self._write_atomic(self.latest_path, self._encode(latest))
            self._prune(snapshot_path)
        return snapshot_path

    def load(self, path: str | Path | None = None) -> dict[str, Any]:
        """Load and validate a numbered snapshot or latest manifest.

        Args:
            path: Snapshot or manifest path. Defaults to this store's
                ``latest.json``.

        Returns:
            Decoded snapshot including framework metadata and content checksum.

        Raises:
            CheckpointFormatError: If files are missing, malformed, unsupported,
                or fail manifest or content checksum validation.
        """

        requested = Path(path) if path is not None else self.latest_path
        if not requested.exists():
            raise CheckpointFormatError(f"Checkpoint path does not exist: {requested}.")
        with self._directory_lock():
            value = self._read_json(requested)
            if value.get("format") == self.LATEST_FORMAT:
                try:
                    snapshot_name = str(value["checkpoint"])
                    expected_checksum = str(value["sha256"])
                except KeyError as exc:
                    raise CheckpointFormatError("Invalid latest checkpoint manifest.") from exc
                if not self._SNAPSHOT_NAME.fullmatch(snapshot_name):
                    raise CheckpointFormatError("Latest manifest has an invalid snapshot name.")
                snapshot_path = requested.parent / snapshot_name
                try:
                    encoded = snapshot_path.read_bytes()
                except OSError as exc:
                    raise CheckpointFormatError(
                        f"Cannot read latest checkpoint snapshot: {exc}."
                    ) from exc
                actual_checksum = hashlib.sha256(encoded).hexdigest()
                if actual_checksum != expected_checksum:
                    raise CheckpointFormatError("Latest checkpoint checksum does not match.")
                value = self._decode(encoded)

        if value.get("format") != self.FORMAT:
            raise CheckpointFormatError("Unknown checkpoint format.")
        try:
            expected_content_checksum = str(value["content_sha256"])
        except KeyError as exc:
            raise CheckpointFormatError("Checkpoint has no content checksum.") from exc
        content = dict(value)
        content.pop("content_sha256", None)
        if self.fingerprint(content) != expected_content_checksum:
            raise CheckpointFormatError("Checkpoint content checksum does not match.")
        return value

    def acquire_session_lease(self) -> None:
        """Acquire non-blocking exclusive ownership of a checkpoint lineage.

        Repeated calls from the owning store are idempotent. A different process
        or store already holding the lock causes :class:`CheckpointFormatError`.
        """

        if self._lease_descriptor is not None:
            return
        self.policy.directory.mkdir(parents=True, exist_ok=True)
        self.policy.directory.chmod(0o700)
        lease_path = self.policy.directory / ".session.lock"
        descriptor = os.open(lease_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(descriptor)
            raise CheckpointFormatError(
                f"Checkpoint directory is already owned by another session: "
                f"{self.policy.directory}."
            ) from exc
        self._lease_descriptor = descriptor

    def release_session_lease(self) -> None:
        """Release this process's checkpoint-lineage ownership lease."""

        descriptor = self._lease_descriptor
        if descriptor is None:
            return
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            self._lease_descriptor = None

    @staticmethod
    def fingerprint(value: Any) -> str:
        """Return a SHA-256 fingerprint for canonical JSON-compatible data."""

        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _encode(value: Any) -> bytes:
        try:
            return json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CheckpointFormatError(
                f"Checkpoint contains a non-JSON value: {exc}."
            ) from exc

    @staticmethod
    def _decode(value: bytes) -> dict[str, Any]:
        try:
            decoded = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointFormatError("Checkpoint is not valid JSON.") from exc
        if not isinstance(decoded, dict):
            raise CheckpointFormatError("Checkpoint root must be an object.")
        return decoded

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            return self._decode(path.read_bytes())
        except OSError as exc:
            raise CheckpointFormatError(f"Cannot read checkpoint {path}: {exc}.") from exc

    def _write_atomic(self, path: Path, value: bytes) -> None:
        """Durably replace a file using a private temporary and directory fsync."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            descriptor = os.open(
                temporary_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as file:
                file.write(value)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _prune(self, latest_snapshot: Path) -> None:
        """Remove oldest numbered snapshots beyond the configured retention."""
        snapshots = sorted(
            (
                path
                for path in self.policy.directory.glob("checkpoint-*.json")
                if self._SNAPSHOT_NAME.fullmatch(path.name)
            ),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        removable = snapshots[: max(0, len(snapshots) - self.policy.keep_last)]
        for snapshot in removable:
            if snapshot != latest_snapshot:
                snapshot.unlink(missing_ok=True)

    @contextmanager
    def _directory_lock(self):
        """Serialize one checkpoint-directory operation with a POSIX file lock."""
        self.policy.directory.mkdir(parents=True, exist_ok=True)
        self.policy.directory.chmod(0o700)
        lock_path = self.policy.directory / ".checkpoint.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


__all__ = ["JSONCheckpointStore"]
