from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from threading import Barrier, Event, Lock, Thread

import pytest

from genio import (
    Artifact,
    BackendError,
    BackendShutdownError,
    EvaluationHandle,
    EvaluationState,
    EvaluationTask,
    ExecutionContext,
    Individual,
    ParallelSSHBackend,
    SSHBackend,
    StageChoice,
    UnknownEvaluationHandleError,
)


class FileArtifact(Artifact):
    def load(self):
        return (Path(self.metadata["path"]).read_text(encoding="utf-8"),)


class RemoteFileTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        package_dir = context.ensure_dir(context.package_dir(self))
        context.write_text(package_dir / "input.txt", "input")
        result = context.run_command(
            (
                "sh",
                "-c",
                "printf '%s' \"$GENIO_TEST\" > output.txt; printf 'remote-output'",
            ),
            cwd=package_dir,
            env={"GENIO_TEST": "result"},
        )
        return [
            FileArtifact(
                name="remote-file",
                producer="ssh-test",
                individual_id=self.individual.id,
                metadata={
                    "path": str(package_dir / "output.txt"),
                    "stdout": result.stdout,
                },
            )
        ]


class FailingRemoteTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        package_dir = context.ensure_dir(context.package_dir(self))
        context.run_command(
            ("sh", "-c", "printf 'preserved' > failure.txt; exit 7"),
            cwd=package_dir,
        )
        return []


class RemoteResourceTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        resource = context.resolve_resource_path(
            context.metadata["toolchain_path"],
            "include",
        )
        return [
            FileArtifact(
                name="resource",
                producer="ssh-test",
                individual_id=self.individual.id,
                metadata={
                    "path": str(
                        context.write_text(
                            "resource.txt",
                            str(
                                context.resource_exists(resource)
                                and context.resource_is_dir(resource)
                            ),
                        )
                    ),
                },
            )
        ]


class TimeoutRemoteTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        package_dir = context.ensure_dir(context.package_dir(self))
        context.run_command(
            (
                "sh",
                "-c",
                "trap '' TERM; sleep 1; printf 'alive' > survived.txt",
            ),
            cwd=package_dir,
            timeout=0.1,
        )
        return []


class DeletingRemoteTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        package_dir = context.ensure_dir(context.package_dir(self))
        context.write_text(package_dir / "stale.txt", "stale")
        context.run_command(
            ("sh", "-c", "rm stale.txt; printf 'current' > current.txt"),
            cwd=package_dir,
        )
        return []


class ConcurrencyProbe:
    def __init__(self, parties: int) -> None:
        self.barrier = Barrier(parties)
        self.lock = Lock()
        self.active = 0
        self.max_active = 0

    def run(self) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.barrier.wait(timeout=5)
        finally:
            with self.lock:
                self.active -= 1


class ParallelRemoteFileTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        self.metadata["probe"].run()
        task_dir = context.ensure_dir(context.task_dir(self))
        context.run_command(
            (
                "sh",
                "-c",
                "mkdir -p \"$GATE\"; "
                ": > \"$GATE/$VALUE.started\"; "
                "while test \"$(find \"$GATE\" -name '*.started' -type f | wc -l)\" "
                "-lt 2; do sleep 0.01; done; "
                "printf '%s' \"$VALUE\" > output.txt",
            ),
            env={
                "GATE": str(self.metadata["gate"]),
                "VALUE": self.individual.id,
            },
            timeout=2,
        )
        return [
            FileArtifact(
                name="remote-file",
                producer="parallel-ssh-test",
                individual_id=self.individual.id,
                metadata={"path": str(task_dir / "output.txt")},
            )
        ]


class BlockingRemoteTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        self.metadata["started"].set()
        if not self.metadata["release"].wait(timeout=5):
            raise TimeoutError("Blocking SSH task was not released.")
        return []


class RelativeWorkspaceTask(EvaluationTask):
    def run(self, context: ExecutionContext) -> list[Artifact]:
        context.write_text("work/input.txt", self.individual.id)
        context.run_command(
            ("sh", "-c", "cp input.txt output.txt"),
            cwd="work",
        )
        return [
            FileArtifact(
                name="relative-workspace",
                producer="parallel-ssh-test",
                individual_id=self.individual.id,
                metadata={"path": str(context.resolve_path("work/output.txt"))},
            )
        ]


def make_individual(identifier: str = "candidate") -> Individual:
    return Individual.from_slots(
        id=identifier,
        scenario="ssh_backend_test",
        slots=[StageChoice(slot=0, stage="nop")],
    )


def install_fake_ssh_tools(tmp_path: Path) -> tuple[Path, Path]:
    ssh_path = tmp_path / "fake_ssh"
    ssh_path.write_text(
        f"""#!{sys.executable}
import subprocess
import sys

completed = subprocess.run(
    sys.argv[-1],
    shell=True,
    capture_output=True,
    text=True,
    start_new_session=True,
)
sys.stdout.write(completed.stdout)
sys.stderr.write(completed.stderr)
raise SystemExit(completed.returncode)
""",
        encoding="utf-8",
    )
    ssh_path.chmod(0o755)

    rsync_path = tmp_path / "fake_rsync"
    rsync_path.write_text(
        f"""#!{sys.executable}
import shutil
import sys
from pathlib import Path

def transfer_path(value):
    return Path(value.split(':', 1)[1] if ':' in value else value)

source = transfer_path(sys.argv[-2])
target = transfer_path(sys.argv[-1])
if '--delete' in sys.argv and target.exists():
    shutil.rmtree(target)
target.mkdir(parents=True, exist_ok=True)
for child in source.iterdir():
    destination = target / child.name
    if child.is_dir():
        shutil.copytree(child, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(child, destination)
""",
        encoding="utf-8",
    )
    rsync_path.chmod(0o755)
    return ssh_path, rsync_path


def make_backend(tmp_path: Path, **kwargs) -> SSHBackend:
    ssh_path, rsync_path = install_fake_ssh_tools(tmp_path)
    return SSHBackend(
        "test-host",
        remote_base_work_dir=tmp_path / "remote",
        local_staging_dir=tmp_path / "staging",
        run_id="run-001",
        ssh_executable=str(ssh_path),
        rsync_executable=str(rsync_path),
        **kwargs,
    )


def make_parallel_backend(tmp_path: Path, **kwargs) -> ParallelSSHBackend:
    ssh_path, rsync_path = install_fake_ssh_tools(tmp_path)
    return ParallelSSHBackend(
        "test-host",
        remote_base_work_dir=tmp_path / "remote",
        local_staging_dir=tmp_path / "staging",
        run_id="run-001",
        ssh_executable=str(ssh_path),
        rsync_executable=str(rsync_path),
        **kwargs,
    )


def test_ssh_backend_stages_executes_and_retrieves_workspace(tmp_path) -> None:
    backend = make_backend(tmp_path)
    task = RemoteFileTask(individual=make_individual(), step_id="remote")

    handle = backend.submit(task)
    artifact = backend.collect(handle)[0]

    assert backend.status(handle) is EvaluationState.DONE
    assert artifact.load() == ("result",)
    assert artifact.metadata["stdout"] == "remote-output"
    assert handle.metadata == {
        "run_id": "run-001",
        "host": "test-host",
        "remote_task_dir": str(tmp_path / "remote/run-001/candidate/remote"),
    }
    assert (
        tmp_path / "remote/run-001/candidate/remote/package/output.txt"
    ).read_text(encoding="utf-8") == "result"


def test_ssh_backend_retrieves_workspace_when_remote_command_fails(tmp_path) -> None:
    backend = make_backend(tmp_path)
    task = FailingRemoteTask(individual=make_individual("failed"), step_id="remote")

    handle = backend.submit(task)

    assert backend.status(handle) is EvaluationState.FAILED
    assert backend.error(handle).startswith("RuntimeError: Remote command")
    with pytest.raises(RuntimeError, match="return code 7"):
        backend.collect(handle)
    assert (
        tmp_path / "staging/failed/remote/package/failure.txt"
    ).read_text(encoding="utf-8") == "preserved"


def test_ssh_backend_resolves_resources_on_remote_host(tmp_path) -> None:
    resource_path = tmp_path / "remote-toolchain"
    (resource_path / "include").mkdir(parents=True)
    backend = make_backend(
        tmp_path,
        metadata={"toolchain_path": str(resource_path)},
    )
    task = RemoteResourceTask(individual=make_individual(), step_id="resource")

    artifact = backend.collect(backend.submit(task))[0]

    assert artifact.load() == ("True",)


@pytest.mark.skipif(sys.platform == "win32", reason="Remote process groups require POSIX.")
def test_ssh_backend_terminates_remote_process_group_on_timeout(tmp_path) -> None:
    backend = make_backend(tmp_path, transfer_timeout=0.25)
    task = TimeoutRemoteTask(individual=make_individual("timeout"), step_id="remote")

    handle = backend.submit(task)

    assert backend.status(handle) is EvaluationState.FAILED
    with pytest.raises(subprocess.TimeoutExpired):
        backend.collect(handle)
    time.sleep(0.6)
    assert not (
        tmp_path / "remote/run-001/timeout/remote/package/survived.txt"
    ).exists()
    assert not (tmp_path / "staging/timeout/remote/package/survived.txt").exists()


def test_ssh_backend_mirrors_remote_deletions_to_local_staging(tmp_path) -> None:
    backend = make_backend(tmp_path)
    task = DeletingRemoteTask(individual=make_individual("deletion"), step_id="remote")

    backend.collect(backend.submit(task))

    package_dir = tmp_path / "staging/deletion/remote/package"
    assert not (package_dir / "stale.txt").exists()
    assert (package_dir / "current.txt").read_text(encoding="utf-8") == "current"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"host": ""}, "host"),
        ({"remote_base_work_dir": "relative"}, "remote_base_work_dir"),
        ({"port": 0}, "port"),
        ({"run_id": "../escape"}, "run_id"),
        ({"transfer_timeout": 0}, "transfer_timeout"),
    ),
)
def test_ssh_backend_validates_configuration(tmp_path, kwargs, message) -> None:
    parameters = {
        "host": "test-host",
        "remote_base_work_dir": "/remote",
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        SSHBackend(local_staging_dir=tmp_path, **parameters)


def test_ssh_backend_rejects_unknown_handles_and_submission_after_shutdown(tmp_path) -> None:
    backend = make_backend(tmp_path)

    with pytest.raises(UnknownEvaluationHandleError):
        backend.status(EvaluationHandle(id="unknown"))

    backend.shutdown()
    with pytest.raises(BackendShutdownError):
        backend.submit(RemoteFileTask(individual=make_individual(), step_id="remote"))


def test_parallel_ssh_backend_executes_remote_tasks_concurrently(tmp_path) -> None:
    probe = ConcurrencyProbe(parties=2)
    gate = tmp_path / "remote-gate"
    tasks = [
        ParallelRemoteFileTask(
            individual=make_individual(identifier),
            step_id="remote",
            metadata={"gate": gate, "probe": probe},
        )
        for identifier in ("first", "second")
    ]

    with make_parallel_backend(tmp_path, max_workers=2) as backend:
        handles = backend.submit_batch(tasks)
        artifacts = backend.collect_batch(handles)

    assert probe.max_active == 2
    assert [task_artifacts[0].load()[0] for task_artifacts in artifacts] == [
        "first",
        "second",
    ]
    assert all(backend.status(handle) is EvaluationState.DONE for handle in handles)
    assert (
        tmp_path / "remote/run-001/first/remote/output.txt"
    ).read_text(encoding="utf-8") == "first"
    assert (
        tmp_path / "remote/run-001/second/remote/output.txt"
    ).read_text(encoding="utf-8") == "second"


def test_parallel_ssh_backend_exposes_original_remote_failure(tmp_path) -> None:
    with make_parallel_backend(tmp_path, max_workers=1) as backend:
        handle = backend.submit(
            FailingRemoteTask(
                individual=make_individual("failed-parallel"),
                step_id="remote",
            )
        )

        with pytest.raises(RuntimeError, match="return code 7"):
            backend.collect(handle)

        assert backend.status(handle) is EvaluationState.FAILED
        assert backend.error(handle).startswith("RuntimeError: Remote command")


def test_parallel_ssh_backend_scopes_relative_helpers_per_task(tmp_path) -> None:
    tasks = [
        RelativeWorkspaceTask(
            individual=make_individual(identifier),
            step_id="relative",
        )
        for identifier in ("first-relative", "second-relative")
    ]

    with make_parallel_backend(tmp_path, max_workers=2) as backend:
        artifacts = backend.collect_batch(backend.submit_batch(tasks))

    assert [task_artifacts[0].load()[0] for task_artifacts in artifacts] == [
        "first-relative",
        "second-relative",
    ]
    assert not (tmp_path / "staging/work").exists()


def test_parallel_ssh_backend_cancels_only_queued_tasks(tmp_path) -> None:
    started = Event()
    release = Event()
    backend = make_parallel_backend(tmp_path, max_workers=1)
    running_handle = backend.submit(
        BlockingRemoteTask(
            individual=make_individual("running"),
            step_id="remote",
            metadata={"started": started, "release": release},
        )
    )
    assert started.wait(timeout=2)
    queued_handle = backend.submit(
        BlockingRemoteTask(
            individual=make_individual("queued"),
            step_id="remote",
            metadata={"started": Event(), "release": Event()},
        )
    )

    try:
        assert backend.status(running_handle) is EvaluationState.RUNNING
        assert backend.status(queued_handle) is EvaluationState.PENDING
        assert backend.cancel(running_handle) is False
        assert backend.cancel(queued_handle) is True
        assert backend.status(queued_handle) is EvaluationState.CANCELLED
        with pytest.raises(BackendError, match="cancelled"):
            backend.collect(queued_handle)
    finally:
        release.set()
        backend.shutdown()

    assert backend.status(running_handle) is EvaluationState.DONE


def test_parallel_ssh_backend_rejects_duplicate_active_workspace(tmp_path) -> None:
    started = Event()
    release = Event()
    task = BlockingRemoteTask(
        individual=make_individual("duplicate"),
        step_id="remote",
        metadata={"started": started, "release": release},
    )
    backend = make_parallel_backend(tmp_path, max_workers=2)
    first_handle = backend.submit(task)
    assert started.wait(timeout=2)

    try:
        with pytest.raises(BackendError, match="same SSH workspace"):
            backend.submit(task)
    finally:
        release.set()
        backend.collect(first_handle)
        backend.shutdown()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"max_workers": 0}, "max_workers"),
        ({"max_workers": True}, "max_workers"),
        ({"max_workers": 1.5}, "max_workers"),
        ({"max_workers": 1, "max_pending": 0}, "max_pending"),
        ({"max_workers": 1, "max_pending": 1.5}, "max_pending"),
    ),
)
def test_parallel_ssh_backend_validates_capacity(tmp_path, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        make_parallel_backend(tmp_path, **kwargs)


def test_parallel_ssh_backend_rejects_unknown_handle_and_late_submission(tmp_path) -> None:
    backend = make_parallel_backend(tmp_path, max_workers=1)

    with pytest.raises(UnknownEvaluationHandleError):
        backend.status(EvaluationHandle(id="unknown"))

    backend.shutdown()
    with pytest.raises(BackendShutdownError):
        backend.submit(RemoteFileTask(individual=make_individual(), step_id="remote"))


def test_parallel_ssh_backend_releases_capacity_after_completion(tmp_path) -> None:
    started = Event()
    release = Event()
    submitted = Event()
    submitted_handles: list[EvaluationHandle] = []
    backend = make_parallel_backend(tmp_path, max_workers=1, max_pending=1)
    first_handle = backend.submit(
        BlockingRemoteTask(
            individual=make_individual("capacity-running"),
            step_id="remote",
            metadata={"started": started, "release": release},
        )
    )
    assert started.wait(timeout=2)

    def submit_second() -> None:
        submitted_handles.append(
            backend.submit(
                RemoteFileTask(
                    individual=make_individual("capacity-next"),
                    step_id="remote",
                )
            )
        )
        submitted.set()

    submitter = Thread(target=submit_second)
    submitter.start()
    assert not submitted.wait(timeout=0.2)

    release.set()
    backend.collect(first_handle)
    assert submitted.wait(timeout=2)
    backend.collect(submitted_handles[0])
    backend.shutdown()
    submitter.join(timeout=2)


def test_parallel_ssh_backend_wakes_capacity_waiters_on_shutdown(tmp_path) -> None:
    started = Event()
    release = Event()
    submit_finished = Event()
    submit_errors: list[Exception] = []
    backend = make_parallel_backend(tmp_path, max_workers=1, max_pending=1)
    running_handle = backend.submit(
        BlockingRemoteTask(
            individual=make_individual("shutdown-running"),
            step_id="remote",
            metadata={"started": started, "release": release},
        )
    )
    assert started.wait(timeout=2)

    def submit_after_capacity() -> None:
        try:
            backend.submit(
                RemoteFileTask(
                    individual=make_individual("shutdown-waiter"),
                    step_id="remote",
                )
            )
        except Exception as exc:
            submit_errors.append(exc)
        finally:
            submit_finished.set()

    submitter = Thread(target=submit_after_capacity)
    submitter.start()
    assert not submit_finished.wait(timeout=0.2)

    backend.shutdown(wait=False)
    assert submit_finished.wait(timeout=1)
    assert len(submit_errors) == 1
    assert isinstance(submit_errors[0], BackendShutdownError)

    release.set()
    backend.shutdown(wait=True)
    submitter.join(timeout=2)
    assert backend.status(running_handle) is EvaluationState.DONE
