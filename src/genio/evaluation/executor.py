"""Workflow executor for supervised local task processes."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from math import isfinite
from multiprocessing import get_context
from multiprocessing.connection import Connection
from numbers import Real
import os
from pathlib import Path, PurePosixPath
import pickle
import signal
import sys
from threading import Lock
import traceback

from genio.evaluation.artifact import Artifact
from genio.checkpoint.codec import signature_value
from genio.core.individual import Individual
from genio.core.result import Result
from genio.evaluation.cache import EvaluationCache
from genio.evaluation.step import EvaluationStep
from genio.evaluation.task import (
    EvaluationOutput,
    EvaluationTask,
    TaskContext,
    _TaskTimeoutInterrupted,
    _workspace_segment,
)
from genio.evaluation.workflow import EvaluationWorkflow
from genio.search_space import SearchSpace


class EvaluationExecutionError(Exception):
    """Raised when an evaluation step creates an invalid task."""


class _TaskFailure(Exception):
    """An error raised by an isolated evaluation task."""


_TIMEOUT_GRACE_SECONDS = 10.0


def _execute_task(
    task: EvaluationTask,
    workspace: Path,
    resources: Mapping[str, object],
    log_dir: Path,
    connection: Connection,
) -> None:
    """Run one task in its own process group and return one evaluation output."""
    if os.name == "posix":
        os.setsid()
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "stdout.log").open("w", encoding="utf-8", buffering=1) as stdout, (
        log_dir / "stderr.log"
    ).open("w", encoding="utf-8", buffering=1) as stderr:
        os.dup2(stdout.fileno(), 1)
        os.dup2(stderr.fileno(), 2)
        sys.stdout = stdout
        sys.stderr = stderr
        previous_handler = signal.getsignal(signal.SIGTERM)

        def interrupt(_signum: int, _frame: object) -> None:
            raise _TaskTimeoutInterrupted()

        signal.signal(signal.SIGTERM, interrupt)
        try:
            context = TaskContext(
                workspace=workspace,
                resources=resources,
            )
            output = task._execute(context)
            try:
                pickle.dumps(output)
            except Exception as exc:
                connection.send(
                    ("invalid_output", f"{type(exc).__name__}: {exc}")
                )
            else:
                connection.send(("success", output))
        except _TaskTimeoutInterrupted:
            try:
                connection.send(("interrupted", None))
            except (OSError, TypeError, ValueError):
                pass
        except BaseException as exc:
            traceback.print_exc()
            try:
                connection.send(("failure", f"{type(exc).__name__}: {exc}"))
            except (OSError, TypeError, ValueError):
                pass
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            connection.close()


@dataclass(slots=True)
class _IndividualEvaluationState:
    individual: Individual
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    result: Result | None = None


class EvaluationExecutor:
    """Execute an evaluation workflow in supervised local processes.

    Steps are processed in topological order. Within each step up to
    ``max_workers`` tasks run concurrently. Failed individuals retain metrics from
    prior steps and are skipped by every later step.

    """

    def __init__(
        self,
        workflow: EvaluationWorkflow,
        *,
        max_workers: int = 1,
        work_dir: str | Path,
        run_id: str,
        execution_metadata: Mapping[str, object] | None = None,
        search_space: SearchSpace | None = None,
        cache: EvaluationCache | None = None,
    ) -> None:
        """Configure a workflow executor.

        Args:
            workflow: Valid dependency graph to execute for every individual.
            max_workers: Maximum number of simultaneous task processes.
            work_dir: Root for task workspaces and retained logs.
            run_id: Identifier separating logs from independent executions.
            execution_metadata: Local resource hints available to each task.
            search_space: Scenario used to select cache key domains.
            cache: Optional session-owned per-step evaluation cache.
        """
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer.")
        if os.name != "posix":
            raise RuntimeError("Supervised task process groups require a POSIX host.")
        self.workflow = workflow
        self.max_workers = max_workers
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_root = self.work_dir / "workspaces"
        self.run_id = _workspace_segment(run_id, "run_id")
        self.execution_metadata = dict(execution_metadata or {})
        self._process_context = get_context("spawn")
        self._active_processes: set[object] = set()
        self._process_lock = Lock()
        self._stopping = False
        if cache is not None and search_space is None:
            raise ValueError("A search space is required when caching evaluations.")
        self.search_space = search_space
        self.cache = cache
        self._cache_domains: dict[str, frozenset[str]] = {}
        self._cache_signatures: dict[str, str] = {}
        if cache is not None:
            assert search_space is not None
            steps = workflow.execution_order()
            needed = {step.id for step in steps if cache.enabled(step.id)}
            for step in reversed(steps):
                if step.id in needed:
                    needed.update(
                        artifact_key.partition(".")[0]
                        for artifact_key in step.required_artifacts
                    )
            all_domains = set(search_space.scenario.domain_order or (
                (("pipeline",) if search_space.scenario.slots else ())
                + tuple(search_space.scenario.design_spaces)
            ))
            for step in steps:
                if step.id not in needed:
                    continue
                declared = step.task_class.uses_domains
                if declared is None:
                    domains = set(all_domains)
                elif isinstance(declared, (tuple, list, set, frozenset)) and all(
                    isinstance(domain, str) for domain in declared
                ):
                    domains = set(declared)
                else:
                    raise ValueError(
                        f"Task {step.task_class.__name__}.uses_domains must be "
                        "a collection of domain names or None."
                    )
                unknown = domains - all_domains
                if unknown:
                    raise ValueError(
                        f"Evaluation step {step.id!r} uses unknown domains: {sorted(unknown)!r}."
                    )
                for artifact_key in step.required_artifacts:
                    domains.update(self._cache_domains[artifact_key.partition(".")[0]])
                self._cache_domains[step.id] = frozenset(domains)
                if cache.enabled(step.id):
                    self._cache_signatures[step.id] = json.dumps(
                        signature_value(step.checkpoint_signature()),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )

    def evaluate(self, individual: Individual) -> Result:
        """Evaluate one individual through the configured workflow.

        Returns:
            The normalized result for ``individual``.
        """

        return self.evaluate_many((individual,))[0]

    def evaluate_many(self, individuals: Sequence[Individual]) -> list[Result]:
        """Evaluate an ordered batch using one process wave per workflow step.

        Task exceptions raised in a worker are converted into a
        failed result for its individual. Structural errors,
        such as invalid tasks, missing artifacts, duplicate artifact names, or
        non-numeric metrics, abort the complete call instead.

        Args:
            individuals: Candidates with IDs unique within this batch.

        Returns:
            One result per input individual in the original order. Successful
            metric keys use the form ``step_id.metric_name``.

        Raises:
            EvaluationExecutionError: If candidate IDs or a workflow/task/artifact
                contract is invalid.
        """

        individual_batch = tuple(individuals)
        self._validate_unique_individual_ids(individual_batch)
        states = [_IndividualEvaluationState(individual) for individual in individual_batch]

        for step in self.workflow.execution_order():
            active_states = [state for state in states if state.result is None]
            if not active_states:
                break

            if self.cache is not None and self.cache.enabled(step.id):
                self._evaluate_cached_step(step, active_states)
                continue

            tasks = [
                self._create_task(step, state.individual, state.artifacts)
                for state in active_states
            ]
            for state, outcome in zip(
                active_states, self._run_wave(step, tasks), strict=True
            ):
                if isinstance(outcome, Exception):
                    self._fail_state(state, outcome)
                else:
                    self._accumulate_output(step, outcome, state)

        return [
            state.result
            or Result.success(
                state.individual.id,
                metrics=dict(state.metrics),
            )
            for state in states
        ]

    def _run_wave(
        self,
        step: EvaluationStep,
        tasks: Sequence[EvaluationTask],
    ) -> list[EvaluationOutput | Exception]:
        """Run one barrier of tasks while preserving individual failure results."""
        if not tasks:
            return []
        timeout = step.timeout_seconds
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not isfinite(timeout)
            or timeout <= 0
        ):
            raise EvaluationExecutionError(
                f"Evaluation step {step.id!r} timeout_seconds must be positive and finite."
            )
        for task in tasks:
            try:
                pickle.dumps((task, self.execution_metadata))
            except Exception as exc:
                raise EvaluationExecutionError(
                    f"Evaluation task {task.task_id!r} cannot be sent to a process."
                ) from exc

        pool = ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            futures = [
                pool.submit(self._run_task_process, task, timeout)
                for task in tasks
            ]
            outcomes: list[EvaluationOutput | Exception] = []
            for future in futures:
                try:
                    outcomes.append(future.result())
                except EvaluationExecutionError:
                    raise
                except Exception as exc:
                    outcomes.append(exc)
        except BaseException:
            with self._process_lock:
                self._stopping = True
            pool.shutdown(wait=False, cancel_futures=True)
            self._stop_active_processes()
            raise
        else:
            pool.shutdown(wait=True)
            return outcomes

    @staticmethod
    def _fail_state(state: _IndividualEvaluationState, exc: Exception) -> None:
        state.result = Result.failed(
            state.individual.id,
            error=(str(exc) if isinstance(exc, _TaskFailure)
                   else f"{type(exc).__name__}: {exc}"),
            metrics=dict(state.metrics),
        )

    def _cache_key(self, step: EvaluationStep, individual: Individual) -> str:
        assert self.search_space is not None
        if individual.scenario != self.search_space.scenario_id:
            raise EvaluationExecutionError(
                f"Individual {individual.id!r} belongs to another search scenario."
            )
        try:
            genotype = self.search_space.to_genotype(individual)
        except ValueError as exc:
            raise EvaluationExecutionError(
                f"Individual {individual.id!r} has inconsistent design values."
            ) from exc
        genes = self.search_space.genes_for_domains(
            genotype, self._cache_domains[step.id]
        )
        return json.dumps(
            (self._cache_signatures[step.id], genes),
            separators=(",", ":"),
            allow_nan=False,
        )

    def _task_workspace(self, individual_id: str, step_id: str) -> Path:
        relative_dir = Path(_workspace_segment(individual_id, "individual_id")) / (
            _workspace_segment(step_id, "step_id")
        )
        workspace = (self.workspace_root / relative_dir).resolve()
        try:
            workspace.relative_to(self.workspace_root.resolve())
        except ValueError as exc:
            raise EvaluationExecutionError("Task workspace escapes its root directory.") from exc
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _evaluate_cached_step(
        self,
        step: EvaluationStep,
        active_states: Sequence[_IndividualEvaluationState],
    ) -> None:
        """Reuse successful outputs and coalesce equal keys within a batch."""
        assert self.cache is not None
        groups: OrderedDict[str, list[_IndividualEvaluationState]] = OrderedDict()
        for state in active_states:
            self._resolve_required_artifacts(step, state.artifacts)
            key = self._cache_key(step, state.individual)
            workspace = self._task_workspace(state.individual.id, step.id)
            cached = self.cache.get(step.id, key, workspace)
            if cached is not None:
                self._accumulate_output(step, cached, state)
            else:
                groups.setdefault(key, []).append(state)

        primary = [members[0] for members in groups.values()]
        tasks = [
            self._create_task(step, state.individual, state.artifacts)
            for state in primary
        ]
        fallback: list[tuple[str, list[_IndividualEvaluationState]]] = []
        for (key, members), outcome in zip(
            groups.items(), self._run_wave(step, tasks), strict=True
        ):
            state = members[0]
            if isinstance(outcome, Exception):
                self._fail_state(state, outcome)
                if len(members) > 1:
                    fallback.append((key, members[1:]))
                continue
            self._store_and_restore(step, key, outcome, state, members[1:])

        for key, members in fallback:
            for position, state in enumerate(members):
                task = self._create_task(step, state.individual, state.artifacts)
                outcome = self._run_wave(step, (task,))[0]
                if isinstance(outcome, Exception):
                    self._fail_state(state, outcome)
                    continue
                self._store_and_restore(step, key, outcome, state, members[position + 1:])
                break

    def _store_and_restore(
        self,
        step: EvaluationStep,
        key: str,
        output: EvaluationOutput,
        source: _IndividualEvaluationState,
        equivalents: Sequence[_IndividualEvaluationState],
    ) -> None:
        """Publish one successful output and materialize its equal-key peers."""
        assert self.cache is not None
        self._accumulate_output(step, output, source)
        self.cache.put(
            step.id, key, output, self._task_workspace(source.individual.id, step.id)
        )
        for state in equivalents:
            restored = self.cache.get(
                step.id, key, self._task_workspace(state.individual.id, step.id)
            )
            if restored is None:
                raise EvaluationExecutionError(
                    f"Cache entry disappeared for step {step.id!r}."
                )
            self._accumulate_output(step, restored, state)

    def _run_task_process(
        self,
        task: EvaluationTask,
        timeout_seconds: float | None,
    ) -> EvaluationOutput:
        """Collect one process result, terminating the group at the deadline."""
        relative_dir = Path(_workspace_segment(task.individual.id, "individual_id")) / (
            _workspace_segment(task.step_id or "task", "step_id")
        )
        workspace = self._task_workspace(task.individual.id, task.step_id or "task")
        log_dir = self.work_dir / "logs" / self.run_id / relative_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        receiver, sender = self._process_context.Pipe(duplex=False)
        process = self._process_context.Process(
            target=_execute_task,
            args=(task, workspace, self.execution_metadata, log_dir, sender),
        )
        try:
            with self._process_lock:
                if self._stopping:
                    raise _TaskFailure("Evaluation was interrupted.")
                process.start()
                self._active_processes.add(process)
            sender.close()
            if not receiver.poll(timeout_seconds):
                self._terminate_process(process, cooperative=True)
                raise _TaskFailure(
                    f"TimeoutError: Task {task.task_id!r} exceeded {timeout_seconds:g} seconds."
                )
            try:
                status, payload = receiver.recv()
            except EOFError as exc:
                raise _TaskFailure(
                    f"Task {task.task_id!r} exited without an evaluation output."
                ) from exc
            process.join(timeout=1.0)
            if process.is_alive():
                self._terminate_process(process)
                raise _TaskFailure(f"Task {task.task_id!r} did not exit after returning.")
            if status == "interrupted":
                raise _TaskFailure(f"Task {task.task_id!r} was interrupted.")
            if status == "invalid_output":
                raise EvaluationExecutionError(
                    f"Task {task.task_id!r} returned an untransportable output: {payload}"
                )
            if status != "success":
                raise _TaskFailure(str(payload))
            return payload
        finally:
            if process.pid is not None and process.is_alive():
                self._terminate_process(process)
            receiver.close()
            sender.close()
            with self._process_lock:
                self._active_processes.discard(process)

    @staticmethod
    def _terminate_process(process: object, *, cooperative: bool = False) -> None:
        """Offer timeout cleanup, then kill any remaining task process group."""
        if process.pid is None:
            return
        if cooperative and process.is_alive():
            try:
                os.kill(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.join(timeout=_TIMEOUT_GRACE_SECONDS)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.is_alive():
            process.kill()
        process.join(timeout=1.0)

    def _stop_active_processes(self) -> None:
        with self._process_lock:
            processes = tuple(self._active_processes)
        for process in processes:
            self._terminate_process(process)

    def _accumulate_output(
        self,
        step: EvaluationStep,
        output: EvaluationOutput,
        state: _IndividualEvaluationState,
    ) -> None:
        """Validate and collect a task's artifacts and local metrics."""
        if not isinstance(output, EvaluationOutput):
            raise EvaluationExecutionError(
                f"Evaluation step {step.id!r} returned a non-EvaluationOutput value."
            )
        if not isinstance(output.artifacts, Mapping):
            raise EvaluationExecutionError(
                f"Evaluation step {step.id!r} output artifacts must be a mapping."
            )
        for name, artifact in output.artifacts.items():
            self._accumulate_artifact(
                step,
                name,
                artifact,
                state.individual.id,
                state.artifacts,
            )
        self._accumulate_metrics(step, output.metrics, state.metrics)

    @staticmethod
    def _validate_unique_individual_ids(individuals: Sequence[Individual]) -> None:
        ids = [individual.id for individual in individuals]
        duplicate_ids = sorted(
            individual_id
            for individual_id, count in Counter(ids).items()
            if count > 1
        )
        if duplicate_ids:
            raise EvaluationExecutionError(
                f"Duplicate individual ids in evaluation batch: {duplicate_ids!r}."
            )

    @staticmethod
    def _create_task(
        step: EvaluationStep,
        individual: Individual,
        artifacts: Mapping[str, Artifact],
    ) -> EvaluationTask:
        required_artifacts = EvaluationExecutor._resolve_required_artifacts(
            step,
            artifacts,
        )
        task = step.create_task(individual, required_artifacts)
        if not isinstance(task, EvaluationTask):
            raise EvaluationExecutionError(
                f"Evaluation step {step.id!r} created {type(task).__name__}, "
                "not an EvaluationTask."
            )
        if task.individual != individual:
            raise EvaluationExecutionError(
                f"Evaluation step {step.id!r} created a task for a different individual."
            )
        if task.step_id != step.id:
            raise EvaluationExecutionError(
                f"Evaluation step {step.id!r} created a task with step_id "
                f"{task.step_id!r}."
            )
        return task

    @staticmethod
    def _resolve_required_artifacts(
        step: EvaluationStep,
        artifacts: Mapping[str, Artifact],
    ) -> dict[str, Artifact]:
        resolved: dict[str, Artifact] = {}
        for artifact_key, artifact_type in step.required_artifacts.items():
            artifact = artifacts.get(artifact_key)
            if artifact is None:
                raise EvaluationExecutionError(
                    f"Evaluation step {step.id!r} requires artifact "
                    f"{artifact_key!r}, but its successful producer did not return it."
                )
            if not isinstance(artifact, artifact_type):
                raise EvaluationExecutionError(
                    f"Evaluation step {step.id!r} requires artifact "
                    f"{artifact_key!r} of type {artifact_type.__name__}, but received "
                    f"{type(artifact).__name__}."
                )
            resolved[artifact_key] = artifact
        return resolved

    def _accumulate_artifact(
        self,
        step: EvaluationStep,
        name: str,
        artifact: Artifact,
        individual_id: str,
        accumulated_artifacts: dict[str, Artifact],
    ) -> None:
        if not isinstance(artifact, Artifact):
            raise EvaluationExecutionError(
                f"Evaluation step {step.id!r} returned a non-Artifact value."
            )
        if not isinstance(name, str) or not name or "." in name:
            raise EvaluationExecutionError(
                f"Evaluation step {step.id!r} artifact names must be non-empty local strings."
            )
        if not isinstance(artifact.files, Mapping):
            raise EvaluationExecutionError(
                f"Evaluation step {step.id!r} artifact {name!r} files must be a mapping."
            )
        workspace = (
            self.workspace_root
            / _workspace_segment(individual_id, "individual_id")
            / _workspace_segment(step.id, "step_id")
        ).resolve()
        for file_name, path in artifact.files.items():
            if (
                not isinstance(file_name, str)
                or not file_name
                or file_name in {".", ".."}
                or PurePosixPath(file_name).is_absolute()
                or ".." in PurePosixPath(file_name).parts
                or file_name != PurePosixPath(file_name).as_posix()
            ):
                raise EvaluationExecutionError(
                    f"Evaluation step {step.id!r} artifact {name!r} has an invalid file name."
                )
            if not isinstance(path, Path) or not path.is_file():
                raise EvaluationExecutionError(
                    f"Evaluation step {step.id!r} artifact {name!r} file {file_name!r} "
                    f"does not exist: {path!r}."
                )
            if not path.resolve().is_relative_to(workspace):
                raise EvaluationExecutionError(
                    f"Evaluation step {step.id!r} artifact {name!r} file {file_name!r} "
                    "must be inside the producing task's workspace."
                )
        artifact_key = f"{step.id}.{name}"
        if artifact_key in accumulated_artifacts:
            msg = f"Duplicate artifact key {artifact_key!r}."
            raise EvaluationExecutionError(msg)
        accumulated_artifacts[artifact_key] = artifact

    def _accumulate_metrics(
        self,
        step: EvaluationStep,
        metrics: Mapping[str, float],
        accumulated_metrics: dict[str, float],
    ) -> None:
        if not isinstance(metrics, Mapping):
            raise EvaluationExecutionError(
                f"Evaluation step {step.id!r} output metrics must be a mapping."
            )
        for metric_name, value in metrics.items():
            if not isinstance(metric_name, str) or not metric_name:
                raise EvaluationExecutionError(
                    f"Evaluation step {step.id!r} metric names must be non-empty strings."
                )
            metric_key = f"{step.id}.{metric_name}"
            if metric_key in accumulated_metrics:
                msg = f"Duplicate metric key {metric_key!r}."
                raise EvaluationExecutionError(msg)
            if isinstance(value, bool) or not isinstance(value, Real):
                msg = f"Metric {metric_key!r} must be numeric, got {value!r}."
                raise EvaluationExecutionError(msg)
            accumulated_metrics[metric_key] = float(value)
