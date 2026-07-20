from __future__ import annotations

import re
import shlex
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from tempfile import mkdtemp
from typing import Any
from uuid import uuid4

from genio.artifacts import Artifact
from genio.backend.base import (
    Backend,
    BackendError,
    BackendShutdownError,
    EvaluationHandle,
    EvaluationState,
    UnknownEvaluationHandleError,
)
from genio.evaluation.task import CommandResult, EvaluationTask, ExecutionContext


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class _SSHExecutionContext(ExecutionContext):
    ssh_target: str
    remote_base_work_dir: PurePosixPath
    ssh_command: tuple[str, ...] = ("ssh",)
    rsync_command: tuple[str, ...] = ("rsync",)
    transfer_timeout: float | None = 60.0

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve relative paths within this task's local staging workspace."""

        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self._task_workspace() / resolved
        return resolved

    def resolve_resource_path(self, path: str | Path, *parts: str) -> Path:
        """Resolve an absolute resource path on the remote execution host."""

        remote_path = PurePosixPath(str(path))
        if not remote_path.is_absolute():
            raise BackendError("SSH resource paths must be absolute POSIX paths.")
        return Path(str(remote_path.joinpath(*parts)))

    def resource_exists(self, path: str | Path) -> bool:
        """Return whether a resource exists on the remote execution host."""

        return self._test_remote_resource(path, "-e")

    def resource_is_dir(self, path: str | Path) -> bool:
        """Return whether a resource is a directory on the remote execution host."""

        return self._test_remote_resource(path, "-d")

    def _test_remote_resource(self, path: str | Path, operator: str) -> bool:
        result = self._run_remote_shell(
            f"test {operator} {shlex.quote(str(path))}",
            timeout=self.transfer_timeout,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise BackendError(
                f"Could not inspect remote resource {str(path)!r}: {result.stderr}"
            )
        return result.returncode == 0

    def run_command(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> CommandResult:
        """Synchronize a workspace, run a command through SSH, and retrieve outputs."""

        normalized_command = tuple(str(part) for part in command)
        if not normalized_command:
            raise ValueError("command cannot be empty.")
        self._validate_environment(env)

        result_cwd, local_cwd = self._resolve_command_cwd(cwd)
        local_cwd.mkdir(parents=True, exist_ok=True)
        remote_cwd = self._remote_path(local_cwd)
        self._sync_to_remote(local_cwd, remote_cwd)

        remote_argv = (
            ("env", *(f"{key}={value}" for key, value in (env or {}).items()), *normalized_command)
            if env
            else normalized_command
        )
        runner = f"echo $$ > .genio.pgid; exec {shlex.join(remote_argv)}"
        remote_command = (
            f"cd {shlex.quote(str(remote_cwd))} && "
            "rm -f .genio.pgid && "
            f"exec setsid --wait sh -c {shlex.quote(runner)}"
        )

        try:
            ssh_result = self._run_remote_shell(
                remote_command,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._terminate_remote_process_group(remote_cwd)
            self._best_effort_remove_pid(remote_cwd)
            self._best_effort_sync_from_remote(remote_cwd, local_cwd)
            raise subprocess.TimeoutExpired(
                normalized_command,
                timeout,
                output=exc.output,
                stderr=exc.stderr,
            ) from exc

        self._best_effort_remove_pid(remote_cwd)
        self._sync_from_remote(remote_cwd, local_cwd)
        result = CommandResult(
            command=normalized_command,
            returncode=ssh_result.returncode,
            stdout=ssh_result.stdout,
            stderr=ssh_result.stderr,
            cwd=result_cwd,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Remote command {result.command!r} failed with return code "
                f"{result.returncode}: {result.stderr}"
            )
        return result

    def _remote_path(self, local_path: str | Path) -> PurePosixPath:
        resolved_path = self.resolve_path(local_path).resolve()
        try:
            relative_path = resolved_path.relative_to(self.base_work_dir.resolve())
        except ValueError as exc:
            raise BackendError(
                f"SSH command path {resolved_path} is outside staging directory "
                f"{self.base_work_dir}."
            ) from exc
        return self.remote_base_work_dir.joinpath(*relative_path.parts)

    def _resolve_command_cwd(
        self,
        cwd: str | Path | None,
    ) -> tuple[Path | None, Path]:
        task_workspace = self._task_workspace()
        if cwd is None:
            return None, task_workspace

        requested_path = Path(cwd)
        resolved_path = (
            requested_path.resolve()
            if requested_path.is_absolute()
            else (task_workspace / requested_path).resolve()
        )
        try:
            resolved_path.relative_to(task_workspace)
        except ValueError as exc:
            raise BackendError(
                f"SSH command cwd {resolved_path} is outside task workspace "
                f"{task_workspace}."
            ) from exc
        return resolved_path, resolved_path

    def _task_workspace(self) -> Path:
        return self.base_work_dir.joinpath(
            str(self.metadata["individual_id"]),
            str(self.metadata["step_id"] or "task"),
        ).resolve()

    def _sync_to_remote(self, local_path: Path, remote_path: PurePosixPath) -> None:
        self._run_remote_shell(
            f"mkdir -p {shlex.quote(str(remote_path))}",
            timeout=self.transfer_timeout,
        )
        self._run_local_command(
            (
                *self.rsync_command,
                "--archive",
                "--delete",
                "--protect-args",
                "--rsh",
                shlex.join(self.ssh_command),
                f"{local_path}/",
                f"{self.ssh_target}:{remote_path}/",
            ),
            timeout=self.transfer_timeout,
        )

    def _sync_from_remote(self, remote_path: PurePosixPath, local_path: Path) -> None:
        local_path.mkdir(parents=True, exist_ok=True)
        self._run_local_command(
            (
                *self.rsync_command,
                "--archive",
                "--delete",
                "--protect-args",
                "--rsh",
                shlex.join(self.ssh_command),
                f"{self.ssh_target}:{remote_path}/",
                f"{local_path}/",
            ),
            timeout=self.transfer_timeout,
        )

    def _run_remote_shell(
        self,
        command: str,
        *,
        timeout: float | None,
        check: bool = True,
    ) -> CommandResult:
        return self._run_local_command(
            (*self.ssh_command, self.ssh_target, command),
            timeout=timeout,
            check=check,
        )

    def _run_local_command(
        self,
        command: Sequence[str],
        *,
        timeout: float | None,
        check: bool = True,
    ) -> CommandResult:
        return ExecutionContext.run_command(
            self,
            command,
            timeout=timeout,
            check=check,
        )

    def _terminate_remote_process_group(self, remote_path: PurePosixPath) -> None:
        pid_path = shlex.quote(str(remote_path / ".genio.pgid"))
        for signal_name in ("TERM", "KILL"):
            command = (
                f"if test -f {pid_path}; then "
                f"pgid=$(cat {pid_path}); "
                f'kill -{signal_name} -"$pgid" 2>/dev/null || true; '
                "fi"
            )
            try:
                self._run_remote_shell(
                    command,
                    timeout=self.transfer_timeout,
                    check=False,
                )
            except Exception:
                pass
            if signal_name == "TERM":
                time.sleep(0.5)

    def _best_effort_remove_pid(self, remote_path: PurePosixPath) -> None:
        try:
            self._run_remote_shell(
                f"rm -f {shlex.quote(str(remote_path / '.genio.pgid'))}",
                timeout=self.transfer_timeout,
                check=False,
            )
        except Exception:
            pass

    def _best_effort_sync_from_remote(
        self,
        remote_path: PurePosixPath,
        local_path: Path,
    ) -> None:
        try:
            self._sync_from_remote(remote_path, local_path)
        except Exception:
            pass

    @staticmethod
    def _validate_environment(env: Mapping[str, str] | None) -> None:
        for name in env or {}:
            if not _ENVIRONMENT_NAME.fullmatch(name):
                raise ValueError(f"Invalid environment variable name: {name!r}.")


@dataclass(slots=True)
class _SSHEvaluationRecord:
    state: EvaluationState
    artifacts: list[Artifact] = field(default_factory=list)
    exception: Exception | None = None
    error: str | None = None


class _SSHExecutionBackend(Backend):
    """Share SSH transport configuration and execution-context construction."""

    def __init__(
        self,
        host: str,
        *,
        remote_base_work_dir: str | PurePosixPath,
        username: str | None = None,
        local_staging_dir: str | Path | None = None,
        port: int | None = None,
        identity_file: str | Path | None = None,
        ssh_options: Sequence[str] = (),
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        transfer_timeout: float | None = 60.0,
        ssh_executable: str = "ssh",
        rsync_executable: str = "rsync",
    ) -> None:
        self.host = self._validate_host(host)
        self.username = username
        self.remote_base_work_dir = self._validate_remote_base_work_dir(
            remote_base_work_dir
        )
        self.port = self._validate_port(port)
        self.identity_file = (
            str(Path(identity_file).expanduser().resolve())
            if identity_file is not None
            else None
        )
        self.ssh_options = tuple(ssh_options)
        self.run_id = self._validate_workspace_segment(
            run_id or uuid4().hex,
            "run_id",
        )
        self.metadata = dict(metadata or {})
        self.transfer_timeout = self._validate_transfer_timeout(transfer_timeout)
        self.ssh_executable = ssh_executable
        self.rsync_executable = rsync_executable
        self.local_staging_dir = self._resolve_staging_dir(local_staging_dir)
        self._shutdown = False

    def create_context(self, task: EvaluationTask) -> _SSHExecutionContext:
        """Create a staging context mapped to the configured remote run directory."""

        individual_id = self._validate_workspace_segment(
            task.individual.id,
            "individual.id",
        )
        step_id = self._validate_workspace_segment(
            task.step_id or "task",
            "step_id",
        )
        return _SSHExecutionContext(
            base_work_dir=self.local_staging_dir,
            run_id=self.run_id,
            backend_id=self.__class__.__name__,
            metadata={
                **self.metadata,
                "task_id": task.task_id,
                "individual_id": individual_id,
                "step_id": step_id,
            },
            ssh_target=self._ssh_target,
            remote_base_work_dir=self.remote_base_work_dir / self.run_id,
            ssh_command=self._ssh_command,
            rsync_command=(self.rsync_executable,),
            transfer_timeout=self.transfer_timeout,
        )

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return remote transport and execution-context compatibility settings."""

        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "host": self.host,
            "username": self.username,
            "remote_base_work_dir": str(self.remote_base_work_dir),
            "port": self.port,
            "identity_file": self.identity_file,
            "ssh_options": list(self.ssh_options),
            "metadata": self.metadata,
            "transfer_timeout": self.transfer_timeout,
            "ssh_executable": self.ssh_executable,
            "rsync_executable": self.rsync_executable,
        }

    def _handle_metadata(
        self,
        task: EvaluationTask,
        context: _SSHExecutionContext,
    ) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "host": self.host,
            "remote_task_dir": str(context._remote_path(context.task_dir(task))),
        }

    @property
    def _ssh_target(self) -> str:
        return f"{self.username}@{self.host}" if self.username else self.host

    @property
    def _ssh_command(self) -> tuple[str, ...]:
        command = [self.ssh_executable]
        if self.port is not None:
            command.extend(("-p", str(self.port)))
        if self.identity_file is not None:
            command.extend(("-i", self.identity_file))
        command.extend(self.ssh_options)
        return tuple(command)

    @staticmethod
    def _validate_host(host: str) -> str:
        normalized = host.strip()
        if not normalized:
            raise ValueError("host cannot be empty.")
        return normalized

    @staticmethod
    def _validate_remote_base_work_dir(
        path: str | PurePosixPath,
    ) -> PurePosixPath:
        normalized = PurePosixPath(path)
        if not normalized.is_absolute():
            raise ValueError("remote_base_work_dir must be an absolute POSIX path.")
        return normalized

    @staticmethod
    def _validate_port(port: int | None) -> int | None:
        if port is not None and (
            isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
        ):
            raise ValueError("port must be an integer between 1 and 65535.")
        return port

    @staticmethod
    def _validate_workspace_segment(value: str, name: str) -> str:
        if not value or value in {".", ".."} or "/" in value:
            raise ValueError(f"{name} must be a non-empty path segment.")
        return value

    @staticmethod
    def _validate_transfer_timeout(timeout: float | None) -> float | None:
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("transfer_timeout must be a positive number or None.")
        return float(timeout) if timeout is not None else None

    @staticmethod
    def _resolve_staging_dir(path: str | Path | None) -> Path:
        resolved = (
            Path(path).expanduser().resolve()
            if path is not None
            else Path(mkdtemp(prefix="genio-ssh-")).resolve()
        )
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved


class SSHBackend(_SSHExecutionBackend):
    """Synchronously stage tasks and execute their external commands through SSH."""

    def __init__(
        self,
        host: str,
        *,
        remote_base_work_dir: str | PurePosixPath,
        username: str | None = None,
        local_staging_dir: str | Path | None = None,
        port: int | None = None,
        identity_file: str | Path | None = None,
        ssh_options: Sequence[str] = (),
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        transfer_timeout: float | None = 60.0,
        ssh_executable: str = "ssh",
        rsync_executable: str = "rsync",
    ) -> None:
        super().__init__(
            host,
            remote_base_work_dir=remote_base_work_dir,
            username=username,
            local_staging_dir=local_staging_dir,
            port=port,
            identity_file=identity_file,
            ssh_options=ssh_options,
            run_id=run_id,
            metadata=metadata,
            transfer_timeout=transfer_timeout,
            ssh_executable=ssh_executable,
            rsync_executable=rsync_executable,
        )
        self._records: dict[str, _SSHEvaluationRecord] = {}

    def submit(self, task: EvaluationTask) -> EvaluationHandle:
        """Stage and execute a task synchronously through the SSH gateway."""

        if self._shutdown:
            raise BackendShutdownError("Cannot submit work after backend shutdown.")
        context = self.create_context(task)
        handle = EvaluationHandle(
            id=uuid4().hex,
            task_id=task.task_id,
            backend_id=self.__class__.__name__,
            metadata=self._handle_metadata(task, context),
            payload=task,
        )
        record = _SSHEvaluationRecord(state=EvaluationState.RUNNING)
        self._records[handle.id] = record
        try:
            record.artifacts = list(task.run(context))
            record.state = EvaluationState.DONE
        except Exception as exc:
            record.exception = exc
            record.error = f"{type(exc).__name__}: {exc}"
            record.state = EvaluationState.FAILED
        return handle

    def collect(self, handle: EvaluationHandle) -> list[Artifact]:
        """Return locally materialized artifacts for an SSH evaluation."""

        record = self._require_record(handle)
        if record.state is EvaluationState.FAILED:
            assert record.exception is not None
            raise record.exception
        if record.state is EvaluationState.CANCELLED:
            raise BackendError(f"Evaluation {handle.id!r} was cancelled.")
        return list(record.artifacts)

    def status(self, handle: EvaluationHandle) -> EvaluationState:
        """Return the state of a synchronously submitted SSH evaluation."""

        return self._require_record(handle).state

    def error(self, handle: EvaluationHandle) -> str | None:
        """Return the error captured for an SSH evaluation."""

        return self._require_record(handle).error

    def cancel(self, handle: EvaluationHandle) -> bool:
        """Report that completed synchronous submissions cannot be cancelled."""

        self._require_record(handle)
        return False

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Reject future submissions to this SSH backend."""

        self._shutdown = True

    def _require_record(self, handle: EvaluationHandle) -> _SSHEvaluationRecord:
        try:
            return self._records[handle.id]
        except KeyError as exc:
            raise UnknownEvaluationHandleError(
                f"Evaluation handle {handle.id!r} does not belong to this backend."
            ) from exc


__all__ = ["SSHBackend"]
