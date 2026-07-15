from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real

from genio.backend.base import Backend
from genio.artifacts import Artifact, MetricArtifact
from genio.cache import ArtifactCache, CacheEntry
from genio.core.individual import Individual
from genio.core.result import Result
from genio.evaluation.step import EvaluationStep
from genio.evaluation.task import EvaluationTask
from genio.evaluation.workflow import EvaluationWorkflow


class EvaluationExecutionError(Exception):
    """Raised when an evaluation step creates an invalid task."""


@dataclass(slots=True)
class _IndividualEvaluationState:
    individual: Individual
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    cache_metadata: dict[str, dict[str, object]] = field(default_factory=dict)
    result: Result | None = None


@dataclass(slots=True)
class _TaskExecutionGroup:
    task: EvaluationTask
    members: list[_IndividualEvaluationState]
    namespace: str | None = None
    cache_key: str | None = None


class EvaluationExecutor:
    """Executes an evaluation workflow for individuals using a backend."""

    def __init__(
        self,
        workflow: EvaluationWorkflow,
        backend: Backend,
        artifact_cache: ArtifactCache | None = None,
    ) -> None:
        self.workflow = workflow
        self.backend = backend
        self.artifact_cache = artifact_cache

    def evaluate(self, individual: Individual) -> Result:
        """Evaluate an individual through the configured workflow."""

        return self.evaluate_many((individual,))[0]

    def evaluate_many(self, individuals: Sequence[Individual]) -> list[Result]:
        """Evaluate individuals in parallel waves while preserving input order."""

        individual_batch = tuple(individuals)
        self._validate_unique_individual_ids(individual_batch)
        states = [_IndividualEvaluationState(individual) for individual in individual_batch]

        for step in self.workflow.execution_order():
            active_states = [state for state in states if state.result is None]
            if not active_states:
                break

            execution_groups = self._prepare_execution_groups(step, active_states)
            handles = self.backend.submit_batch(
                [group.task for group in execution_groups]
            )
            if len(handles) != len(execution_groups):
                raise EvaluationExecutionError(
                    f"Backend returned {len(handles)} handles for "
                    f"{len(execution_groups)} tasks."
                )

            for group, handle in zip(execution_groups, handles, strict=True):
                try:
                    artifacts = self.backend.collect(handle)
                except Exception as exc:
                    self._fail_group(group, exc)
                    continue

                representative = group.members[0]
                self._accumulate_artifacts(step, artifacts, representative)
                if group.cache_key is None or self.artifact_cache is None:
                    continue

                entry = self.artifact_cache.put(
                    group.namespace or step.id,
                    group.cache_key,
                    artifacts,
                    source_individual_id=representative.individual.id,
                    initial_reads=len(group.members),
                )
                for state in group.members:
                    state.cache_metadata[step.id]["cache_entry_read_count"] = (
                        entry.read_count
                    )
                for state in group.members[1:]:
                    cached_artifacts = entry.artifacts_for(
                        state.individual.id,
                        coalesced=True,
                    )
                    self._accumulate_artifacts(step, cached_artifacts, state)

        return [
            state.result
            or Result.success(
                state.individual.id,
                metrics=dict(state.metrics),
                metadata=self._result_metadata(state),
            )
            for state in states
        ]

    def _prepare_execution_groups(
        self,
        step: EvaluationStep,
        states: Sequence[_IndividualEvaluationState],
    ) -> list[_TaskExecutionGroup]:
        bypass_groups: list[_TaskExecutionGroup] = []
        cache_groups: dict[str, _TaskExecutionGroup] = {}

        for state in states:
            task = self._create_task(step, state.individual, state.artifacts)
            cache_inputs = task.cache_inputs()
            namespace = step.id
            if (
                self.artifact_cache is None
                or cache_inputs is None
                or self.artifact_cache.capacity(namespace) <= 0
            ):
                if self.artifact_cache is not None:
                    self.artifact_cache.record_bypass(namespace)
                    state.cache_metadata[step.id] = {"status": "bypass"}
                bypass_groups.append(_TaskExecutionGroup(task=task, members=[state]))
                continue
            if not isinstance(cache_inputs, Mapping):
                raise EvaluationExecutionError(
                    f"Task {type(task).__name__}.cache_inputs() must return a mapping or None."
                )

            cache_key = self.artifact_cache.build_key(namespace, cache_inputs)
            group = cache_groups.get(cache_key)
            if group is None:
                cache_groups[cache_key] = _TaskExecutionGroup(
                    task=task,
                    members=[state],
                    namespace=namespace,
                    cache_key=cache_key,
                )
            else:
                group.members.append(state)

        pending_groups: list[_TaskExecutionGroup] = []
        for group in cache_groups.values():
            assert group.cache_key is not None
            entry = self.artifact_cache.get(
                group.namespace or step.id,
                group.cache_key,
                reads=len(group.members),
            )
            if entry is None:
                self._mark_cache_miss(step, group)
                pending_groups.append(group)
                continue
            self._apply_cache_hit(step, group, entry)

        return [*bypass_groups, *pending_groups]

    def _mark_cache_miss(
        self,
        step: EvaluationStep,
        group: _TaskExecutionGroup,
    ) -> None:
        assert group.cache_key is not None
        source_individual_id = group.members[0].individual.id
        for index, state in enumerate(group.members):
            state.cache_metadata[step.id] = {
                "status": "miss" if index == 0 else "coalesced",
                "cache_hit": index > 0,
                "cache_key": group.cache_key,
                "cache_source_individual_id": source_individual_id,
            }

    def _apply_cache_hit(
        self,
        step: EvaluationStep,
        group: _TaskExecutionGroup,
        entry: CacheEntry,
    ) -> None:
        for state in group.members:
            state.cache_metadata[step.id] = {
                "status": "hit",
                "cache_hit": True,
                "cache_key": entry.key,
                "cache_source_individual_id": entry.source_individual_id,
                "cache_entry_read_count": entry.read_count,
            }
            self._accumulate_artifacts(
                step,
                entry.artifacts_for(state.individual.id),
                state,
            )

    def _fail_group(self, group: _TaskExecutionGroup, exc: Exception) -> None:
        for state in group.members:
            state.result = Result.failed(
                state.individual.id,
                error=f"{type(exc).__name__}: {exc}",
                metrics=dict(state.metrics),
                metadata=self._result_metadata(state),
            )

    def _accumulate_artifacts(
        self,
        step: EvaluationStep,
        artifacts: Sequence[Artifact],
        state: _IndividualEvaluationState,
    ) -> None:
        for artifact in artifacts:
            self._accumulate_artifact(step, artifact, state.artifacts)
            if isinstance(artifact, MetricArtifact):
                self._accumulate_metrics(step, artifact, state.metrics)

    @staticmethod
    def _result_metadata(state: _IndividualEvaluationState) -> dict[str, object]:
        return {"cache": dict(state.cache_metadata)} if state.cache_metadata else {}

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
        task = step.create_task(individual, artifacts)
        if not isinstance(task, step.task_type):
            msg = (
                f"Evaluation step {step.id!r} declared task type "
                f"{step.task_type.__name__}, but created "
                f"{type(task).__name__}."
            )
            raise EvaluationExecutionError(msg)
        return task

    def _accumulate_artifact(
        self,
        step: EvaluationStep,
        artifact: Artifact,
        accumulated_artifacts: dict[str, Artifact],
    ) -> None:
        artifact_key = f"{step.id}.{artifact.name}"
        if artifact_key in accumulated_artifacts:
            msg = f"Duplicate artifact key {artifact_key!r}."
            raise EvaluationExecutionError(msg)
        accumulated_artifacts[artifact_key] = artifact

    def _accumulate_metrics(
        self,
        step: EvaluationStep,
        artifact: MetricArtifact,
        accumulated_metrics: dict[str, float],
    ) -> None:
        for metric_name, value in artifact.metrics().items():
            metric_key = f"{step.id}.{metric_name}"
            if metric_key in accumulated_metrics:
                msg = f"Duplicate metric key {metric_key!r}."
                raise EvaluationExecutionError(msg)
            if isinstance(value, bool) or not isinstance(value, Real):
                msg = f"Metric {metric_key!r} must be numeric, got {value!r}."
                raise EvaluationExecutionError(msg)
            accumulated_metrics[metric_key] = float(value)
