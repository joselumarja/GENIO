"""Executable evaluation tasks and their backend-provided runtime context."""

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
    """Capture a command invocation executed through an evaluation context.

    Attributes:
        command: Exact argument vector passed to the process launcher.
        returncode: Process exit code.
        stdout: Complete captured standard output.
        stderr: Complete captured standard error.
        cwd: Resolved working directory, when one was supplied.
    """

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    cwd: Path | None = None


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Provide filesystem, process, and cancellation services to tasks.

    Backends may override these operations to execute them on another host. Tasks
    should therefore use the context instead of direct filesystem or subprocess
    calls whenever their work must be backend-portable.

    The conventional workspace layout is
    ``base_work_dir/individual_id/step_id/{package,artifacts,logs}``.

    Attributes:
        base_work_dir: Root directory for all task workspaces.
        run_id: Optional optimization-run identifier supplied by the backend.
        backend_id: Optional identity of the backend instance.
        metadata: Backend resources and execution configuration available to tasks.
    """

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
        """Resolve a relative path against the base working directory.

        Absolute paths are returned unchanged. This is a convenience resolver,
        not a sandbox boundary; callers are responsible for trusted paths.
        """

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
        """Return a path below the task's individual and step workspace."""

        step_id = task.step_id or "task"
        workspace = self.individual_dir(task.individual.id) / _workspace_segment(
            step_id, "step_id"
        )
        target = workspace.joinpath(*parts).resolve()
        try:
            target.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("Task workspace path escapes its step directory.") from exc
        return target

    def individual_dir(self, individual_id: str) -> Path:
        """Return a validated individual workspace below ``base_work_dir``."""

        return self.base_work_dir.resolve() / _workspace_segment(
            individual_id, "individual_id"
        )

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
        """Materialize an execution package in the task's package directory.

        Returns:
            The concrete package directory returned by the package implementation.
        """

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
        symlinks: bool = False,
    ) -> Path:
        """Copy a directory tree to a resolved target path."""

        resolved_source = Path(source)
        resolved_target = self.resolve_path(target)
        resolved_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            resolved_source,
            resolved_target,
            dirs_exist_ok=dirs_exist_ok,
            symlinks=symlinks,
        )
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
        """Reject future commands and terminate active command groups.

        Cancellation is irreversible for this context. The first request returns
        ``True``; subsequent requests return ``False``. Python work that does not
        use :meth:`run_command` cannot be interrupted by this mechanism.
        """

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
        """Run a subprocess with captured output and cooperative cancellation.

        Args:
            command: Program and arguments. Shell parsing is never used.
            cwd: Optional working directory, resolved through :meth:`resolve_path`.
            env: Values merged over the current process environment.
            timeout: Maximum execution time in seconds, or ``None`` for no limit.
            check: Raise when the process returns a non-zero status.

        Returns:
            The command, status, output streams, and resolved working directory.

        Raises:
            concurrent.futures.CancelledError: If cancellation was requested before
                or during execution.
            subprocess.TimeoutExpired: If the process exceeds ``timeout``. Captured
                output is attached after its process group is terminated.
            RuntimeError: If ``check`` is true and the process exits unsuccessfully.

        Note:
            On POSIX, each command starts a new process session so cancellation and
            timeouts terminate the complete process group rather than only its root.
        """

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
                assert timeout is not None
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
    """Executable unit of work created for one individual and workflow step.

    Attributes:
        individual: Candidate whose configuration the task evaluates.
        id: Optional explicit task identifier.
        step_id: Producing workflow-step identifier used for workspace names.
        metadata: Fixed task configuration, including optional execution settings.
    """

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
        """Return semantic variable inputs for caching, or ``None`` to bypass it.

        Subclasses should include every value that can vary between otherwise
        identical task instances and change produced artifacts. Values must have a
        deterministic JSON/string representation accepted by the configured cache.
        Fixed step configuration may remain in the cache namespace contract.
        """

        return None

    def execution_timeout_seconds(self) -> float | None:
        """Return a positive timeout from execution metadata, or ``None``.

        The value is read from ``metadata.execution.timeout_seconds``.

        Raises:
            ValueError: If execution metadata is malformed or the timeout is not a
                positive real number.
        """

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
        """Serialize the individual's ordered pipeline as cache-key input."""
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
        """Execute this task using a backend-provided runtime context.

        Implementations should place persistent outputs below ``artifact_path``
        and return artifacts bound to this task's individual. Raised exceptions
        are propagated by the backend and converted to failed results by the
        standard executor.

        Returns:
            Artifacts produced by the task for downstream steps and metrics.
        """


def _workspace_segment(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).is_absolute()
        or len(Path(value).parts) != 1
    ):
        raise ValueError(f"{name} must be a safe workspace path segment.")
    return value
