"""Session-local cache for evaluation outputs and their artifact files."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath
import pickle
import shutil
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import uuid4

from genio.evaluation.artifact import Artifact
from genio.evaluation.task import EvaluationOutput


def _segment(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be a safe path segment.")
    return value


def _file_name(value: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("Artifact file name must be a safe relative path.")
    return Path(value)


def _directory(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_dir() or path.absolute() != path.resolve():
        raise ValueError(f"Directory must exist without symlinks: {path!r}.")
    return path.resolve()


def _copy_file(source: Path, source_root: Path, destination: Path, target_root: Path) -> None:
    if not source.is_file() or source.is_symlink() or not source.resolve().is_relative_to(source_root):
        raise ValueError(f"Artifact file must be inside its source workspace: {source!r}.")
    for parent in (source, *source.parents):
        if parent == source_root:
            break
        if parent.is_symlink():
            raise ValueError(f"Artifact file traverses a symlink: {source!r}.")
    if not destination.is_relative_to(target_root):
        raise ValueError("Artifact destination escapes its workspace.")
    for parent in (destination, *destination.parents):
        if parent == target_root:
            break
        if parent.is_symlink():
            raise ValueError(f"Artifact destination traverses a symlink: {destination!r}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


class EvaluationCache:
    """Keep per-step LRU outputs for the lifetime of one evaluation session.

    ``root`` must be a new, dedicated directory: ``clear`` deletes it entirely.
    Callers must not mutate producer files concurrently with ``put`` or target
    workspaces concurrently with ``get``.
    """

    def __init__(self, root: Path, capacities: Mapping[str, int | None]) -> None:
        if not isinstance(root, Path):
            raise TypeError("Cache root must be a Path.")
        if not isinstance(capacities, Mapping):
            raise TypeError("Cache capacities must be a mapping.")
        self.capacities = dict(capacities)
        for step_id, capacity in self.capacities.items():
            _segment(step_id, "Step ID")
            if capacity is not None and (
                isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0
            ):
                raise ValueError("Cache capacity must be a non-negative integer or None.")
        self.root = root.absolute()
        if self.root.exists() or self.root.is_symlink():
            raise ValueError("Cache root must be a new, dedicated directory.")
        if self.root.parent.resolve() != self.root.parent:
            raise ValueError("Cache root parent must not traverse a symlink.")
        self.root.mkdir(parents=True)
        self._entries: dict[str, OrderedDict[str, tuple[Path, EvaluationOutput]]] = {}
        self._lock = Lock()

    def _workspace(self, workspace: Path) -> Path:
        path = _directory(workspace)
        if self.root.is_relative_to(path) or path.is_relative_to(self.root):
            raise ValueError("Cache root must be outside task workspaces.")
        return path

    def _root(self) -> Path:
        if (
            self.root.parent.resolve() != self.root.parent
            or self.root.is_symlink()
            or (self.root.exists() and self.root.resolve() != self.root)
        ):
            raise ValueError("Cache root must not traverse a symlink.")
        if not self.root.exists():
            self.root.mkdir(parents=True)
        return _directory(self.root)

    def enabled(self, step_id: str) -> bool:
        """Return whether a step has a finite or unlimited cache capacity."""
        return step_id in self.capacities and self.capacities[step_id] != 0

    def get(self, step_id: str, key: str, workspace: Path) -> EvaluationOutput | None:
        """Restore a hit into this individual's step workspace."""
        with self._lock:
            _segment(step_id, "Step ID")
            if not isinstance(key, str):
                raise TypeError("Cache key must be a string.")
            entries = self._entries.get(step_id)
            if not entries or key not in entries:
                return None
            target = self._workspace(workspace)
            self._root()
            directory, output = entries[key]
            if _directory(directory) != directory:
                raise ValueError("Cached entry directory is invalid.")
            artifacts = {}
            for name, artifact in output.artifacts.items():
                files = {}
                for file_name, source in artifact.files.items():
                    destination = target / name / _file_name(file_name)
                    _copy_file(source, directory, destination, target)
                    files[file_name] = destination
                artifacts[name] = replace(
                    artifact, data=pickle.loads(pickle.dumps(artifact.data)), files=files
                )
            entries.move_to_end(key)
            return EvaluationOutput(
                metrics=pickle.loads(pickle.dumps(output.metrics)), artifacts=artifacts
            )

    def put(self, step_id: str, key: str, output: EvaluationOutput, workspace: Path) -> None:
        """Snapshot an output, atomically publishing its copied artifact files."""
        with self._lock:
            _segment(step_id, "Step ID")
            if not isinstance(key, str):
                raise TypeError("Cache key must be a string.")
            if not isinstance(output, EvaluationOutput):
                raise TypeError("Cache output must be an EvaluationOutput.")
            if self.capacities.get(step_id, 0) == 0:
                return
            producer = self._workspace(workspace)
            root = self._root()
            with TemporaryDirectory(prefix=".stage-", dir=root) as staging_name:
                staging = Path(staging_name)
                artifacts = {}
                for name, artifact in output.artifacts.items():
                    _segment(name, "Artifact name")
                    if "." in name:
                        raise ValueError("Artifact names must not contain dots.")
                    if not isinstance(artifact, Artifact):
                        raise TypeError("Cache artifacts must be Artifact instances.")
                    files = {}
                    for file_name, source in artifact.files.items():
                        relative = Path(name) / _file_name(file_name)
                        if not isinstance(source, Path):
                            raise TypeError("Artifact file paths must be Paths.")
                        _copy_file(source, producer, staging / relative, staging)
                        files[file_name] = relative
                    artifacts[name] = replace(
                        artifact, data=pickle.loads(pickle.dumps(artifact.data)), files=files
                    )
                metrics = pickle.loads(pickle.dumps(dict(output.metrics)))
                directory = root / uuid4().hex
                staging.rename(directory)
            cached_artifacts = {
                name: replace(
                    artifact,
                    files={
                        file_name: directory / relative
                        for file_name, relative in artifact.files.items()
                    },
                )
                for name, artifact in artifacts.items()
            }
            entries = self._entries.setdefault(step_id, OrderedDict())
            previous = entries.pop(key, None)
            entries[key] = (directory, EvaluationOutput(metrics=metrics, artifacts=cached_artifacts))
            if previous is not None:
                shutil.rmtree(previous[0])
            capacity = self.capacities[step_id]
            if capacity is not None:
                while len(entries) > capacity:
                    _, (evicted, _) = entries.popitem(last=False)
                    shutil.rmtree(evicted)

    def clear(self) -> None:
        """Discard all entries and the session-owned cache directory."""
        with self._lock:
            if self.root.is_symlink() or (self.root.exists() and self.root.resolve() != self.root):
                raise ValueError("Cache root must not traverse a symlink.")
            if self.root.exists():
                shutil.rmtree(_directory(self.root))
            self._entries.clear()

    @property
    def entry_count(self) -> int:
        with self._lock:
            return sum(len(entries) for entries in self._entries.values())


__all__ = ["EvaluationCache"]
