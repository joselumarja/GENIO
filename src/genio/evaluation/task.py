from __future__ import annotations

import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
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

    def resolve_path(self, path: str | Path) -> Path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self.base_work_dir / resolved
        return resolved

    def task_dir(self, task: EvaluationTask, *parts: str | Path) -> Path:
        step_id = task.step_id or "task"
        return self.base_work_dir.joinpath(task.individual.id, step_id, *parts)

    def artifact_path(self, task: EvaluationTask, *parts: str | Path) -> Path:
        return self.task_dir(task, "artifacts", *parts)

    def package_dir(self, task: EvaluationTask, *parts: str | Path) -> Path:
        return self.task_dir(task, "package", *parts)

    def materialize_package(
        self,
        task: EvaluationTask,
        package: ExecutionPackage,
    ) -> Path:
        return package.materialize(self.package_dir(task))

    def log_path(self, task: EvaluationTask, *parts: str | Path) -> Path:
        return self.task_dir(task, "logs", *parts)

    def ensure_dir(self, path: str | Path) -> Path:
        resolved = self.resolve_path(path)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def ensure_parent(self, path: str | Path) -> Path:
        resolved = self.resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def write_text(self, path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
        resolved = self.ensure_parent(path)
        resolved.write_text(content, encoding=encoding)
        return resolved

    def read_text(self, path: str | Path, *, encoding: str = "utf-8") -> str:
        return self.resolve_path(path).read_text(encoding=encoding)

    def write_bytes(self, path: str | Path, content: bytes) -> Path:
        resolved = self.ensure_parent(path)
        resolved.write_bytes(content)
        return resolved

    def read_bytes(self, path: str | Path) -> bytes:
        return self.resolve_path(path).read_bytes()

    def write_json(
        self,
        path: str | Path,
        data: Any,
        *,
        encoding: str = "utf-8",
        indent: int | None = 2,
    ) -> Path:
        resolved = self.ensure_parent(path)
        with resolved.open("w", encoding=encoding) as file:
            json.dump(data, file, indent=indent)
        return resolved

    def read_json(self, path: str | Path, *, encoding: str = "utf-8") -> Any:
        with self.resolve_path(path).open("r", encoding=encoding) as file:
            return json.load(file)

    def copy_file(self, source: str | Path, target: str | Path) -> Path:
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
        resolved_source = Path(source)
        resolved_target = self.resolve_path(target)
        resolved_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(resolved_source, resolved_target, dirs_exist_ok=dirs_exist_ok)
        return resolved_target

    def write_log(self, task: EvaluationTask, name: str, content: str) -> Path:
        return self.write_text(self.log_path(task, name), content)

    def merged_env(self, env: Mapping[str, str] | None = None) -> dict[str, str]:
        merged = dict(os.environ)
        merged.update(env or {})
        return merged

    def run_command(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> CommandResult:
        resolved_cwd = self.resolve_path(cwd) if cwd is not None else None
        process = subprocess.run(
            tuple(command),
            cwd=resolved_cwd,
            env=self.merged_env(env),
            timeout=timeout,
            check=False,
            capture_output=True,
            text=True,
        )
        result = CommandResult(
            command=tuple(command),
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            cwd=resolved_cwd,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command {result.command!r} failed with return code "
                f"{result.returncode}: {result.stderr}"
            )
        return result


@dataclass(frozen=True, slots=True)
class EvaluationTask(ABC):
    """Executable unit of work created by an evaluation step."""

    individual: Individual
    id: str | None = None
    step_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        if self.id is not None:
            return self.id
        if self.step_id is not None:
            return f"{self.individual.id}:{self.step_id}"
        return self.individual.id

    @abstractmethod
    def run(self, context: ExecutionContext) -> list[Artifact]:
        """Execute this task using backend-provided runtime context."""
