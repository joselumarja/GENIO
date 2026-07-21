from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from threading import Event, Lock
from typing import Any, TYPE_CHECKING

from genio.artifacts import Artifact
from genio.core.individual import Individual

if TYPE_CHECKING:
    from genio.composer import ExecutionPackage


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Result of a command executed through an evaluation context."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    cwd: Path | None = None


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Runtime context provided by a backend to executable tasks."""

    base_work_dir: Path
    run_id: str | None = None
    backend_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _cancel_requested: Event = field(
        default_factory=Event,
        init=False,
        repr=False,
        compare=False,
    )
    _process_lock: Lock = field(
        default_factory=Lock,
        init=False,
        repr=False,
        compare=False,
    )
    _active_processes: dict[int, subprocess.Popen[str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a path against the base working directory."""

        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self.base_work_dir / resolved
        return resolved

    def resolve_resource_path(self, path: str | Path, *parts: str) -> Path:
        """Resolve an execution-host resource path and append optional parts."""

        return Path(path).expanduser().resolve().joinpath(*parts)

    def resource_exists(self, path: str | Path) -> bool:
        """Return whether a resource exists on the execution host."""

        return Path(path).exists()

    def resource_is_dir(self, path: str | Path) -> bool:
        """Return whether a resource is a directory on the execution host."""

        return Path(path).is_dir()

    def task_dir(self, task: EvaluationTask, *parts: str | Path) -> Path:
        """Return a path within a task's working directory."""

        step_id = task.step_id or "task"
        return self.base_work_dir.joinpath(task.individual.id, step_id, *parts)

    def artifact_path(self, task: EvaluationTask, *parts: str | Path) -> Path:
        """Return a path within a task's artifact directory."""

        return self.task_dir(task, "artifacts", *parts)

    def package_dir(self, task: EvaluationTask, *parts: str | Path) -> Path:
        """Return a path within a task's package directory."""

        return self.task_dir(task, "package", *parts)

    def materialize_package(
        self,
        task: EvaluationTask,
        package: ExecutionPackage,
    ) -> Path:
        """Materialize an execution package in a task's package directory."""

        return package.materialize(self.package_dir(task))

    def log_path(self, task: EvaluationTask, *parts: str | Path) -> Path:
        """Return a path within a task's log directory."""

        return self.task_dir(task, "logs", *parts)

    def ensure_dir(self, path: str | Path) -> Path:
        """Create a directory if needed and return its resolved path."""

        resolved = self.resolve_path(path)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def ensure_parent(self, path: str | Path) -> Path:
        """Create a path's parent directory and return the resolved path."""

        resolved = self.resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def write_text(self, path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
        """Write text to a resolved path and return that path."""

        resolved = self.ensure_parent(path)
        resolved.write_text(content, encoding=encoding)
        return resolved

    def read_text(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        """Read text from a resolved path."""

        return self.resolve_path(path).read_text(encoding=encoding)

    def write_bytes(self, path: str | Path, content: bytes) -> Path:
        """Write bytes to a resolved path and return that path."""

        resolved = self.ensure_parent(path)
        resolved.write_bytes(content)
        return resolved

    def read_bytes(self, path: str | Path) -> bytes:
        """Read bytes from a resolved path."""

        return self.resolve_path(path).read_bytes()

    def write_json(
        self,
        path: str | Path,
        data: Any,
        *,
        encoding: str = "utf-8",
        indent: int | None = 2,
    ) -> Path:
        """Serialize JSON data to a resolved path and return that path."""

        resolved = self.ensure_parent(path)
        with resolved.open("w", encoding=encoding) as file:
            json.dump(data, file, indent=indent)
        return resolved

    def read_json(self, path: str | Path, *, encoding: str = "utf-8") -> Any:
        """Deserialize JSON data from a resolved path."""

        with self.resolve_path(path).open("r", encoding=encoding) as file:
            return json.load(file)

    def copy_file(self, source: str | Path, target: str | Path) -> Path:
        """Copy a file to a resolved target path and return that path."""

        resolved_source = Path(source)
        resolved_target = self.ensure_parent(target)
        shutil.copy2(resolved_source, resolved_target)
        return resolved_target

    def copy_tree(
        self,
        source: str | Path,
        target: str | Path,
        *,
        dirs_exist_ok: bool = True,
    ) -> Path:
        """Copy a directory tree to a resolved target path."""

        resolved_source = Path(source)
        resolved_target = self.resolve_path(target)
        resolved_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(resolved_source, resolved_target, dirs_exist_ok=dirs_exist_ok)
        return resolved_target

    def write_log(self, task: EvaluationTask, name: str, content: str) -> Path:
        """Write content to a task log and return its path."""

        return self.write_text(self.log_path(task, name), content)

    def merged_env(self, env: Mapping[str, str] | None = None) -> dict[str, str]:
        """Merge environment overrides with the process environment."""

        merged = dict(os.environ)
        merged.update(env or {})
        return merged

    def cancel(self) -> bool:
        """Reject future commands and terminate command groups currently running."""

        with self._process_lock:
            if self._cancel_requested.is_set():
                return False
            self._cancel_requested.set()
            processes = tuple(self._active_processes.values())

        for process in processes:
            self._terminate_process_group(process)
        return True

    def run_command(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> CommandResult:
        """Run a command within the execution context and return its result."""

        resolved_cwd = self.resolve_path(cwd) if cwd is not None else None
        normalized_command = tuple(command)
        with self._process_lock:
            if self._cancel_requested.is_set():
                raise CancelledError("Execution context was cancelled.")
            process = subprocess.Popen(
                normalized_command,
                cwd=resolved_cwd,
                env=self.merged_env(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=os.name == "posix",
            )
            self._active_processes[process.pid] = process
        try:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                self._terminate_process_group(process)
                stdout, stderr = process.communicate()
                raise subprocess.TimeoutExpired(
                    normalized_command,
                    timeout,
                    output=stdout,
                    stderr=stderr,
                ) from exc
            except BaseException:
                self._terminate_process_group(process)
                raise
        finally:
            with self._process_lock:
                self._active_processes.pop(process.pid, None)

        if self._cancel_requested.is_set():
            raise CancelledError("Execution context was cancelled.")

        result = CommandResult(
            command=normalized_command,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            cwd=resolved_cwd,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command {result.command!r} failed with return code "
                f"{result.returncode}: {result.stderr}"
            )
        return result

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str]) -> None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.terminate()

        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass

        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()

        process.wait()


@dataclass(frozen=True, slots=True)
class EvaluationTask(ABC):
    """Executable unit of work created by an evaluation step."""

    individual: Individual
    id: str | None = None
    step_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        """Return the stable identifier for this evaluation task."""

        if self.id is not None:
            return self.id
        if self.step_id is not None:
            return f"{self.individual.id}:{self.step_id}"
        return self.individual.id

    def cache_inputs(self) -> Mapping[str, Any] | None:
        """Return semantic variable inputs for caching, or None to bypass it."""

        return None

    def execution_timeout_seconds(self) -> float | None:
        """Return a positive timeout from execution metadata, or None if disabled."""

        execution = self.metadata.get("execution", {})
        if not isinstance(execution, Mapping):
            raise ValueError("metadata.execution must be a mapping.")
        timeout = execution.get("timeout_seconds")
        if timeout is None:
            return None
        if isinstance(timeout, bool) or not isinstance(timeout, Real) or timeout <= 0:
            raise ValueError(
                "metadata.execution.timeout_seconds must be a positive number."
            )
        return float(timeout)

    def _pipeline_cache_inputs(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "slot": choice.slot,
                "stage": choice.stage,
                "parameters": choice.parameters,
                "wrapper_inputs": choice.wrapper_inputs,
            }
            for choice in self.individual.slots
        )

    @abstractmethod
    def run(self, context: ExecutionContext) -> list[Artifact]:
        """Execute this task using backend-provided runtime context."""
