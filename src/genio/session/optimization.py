"""Optimization-session orchestration and checkpoint restoration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import warnings
from typing import Any
from uuid import uuid4

from genio.algorithm.base import SearchAlgorithm, SearchContext
from genio.backend.base import Backend
from genio.cache import ArtifactCache
from genio.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointNotSupportedError,
    CheckpointPolicy,
    CheckpointStateError,
    JSONCheckpointStore,
)
from genio.checkpoint.codec import (
    decode_evaluation,
    decode_evaluated_batch,
    encode_evaluation,
    encode_evaluated_batch,
    qualified_name,
    signature_value,
)
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual
from genio.core.proposal import Proposal
from genio.core.search_result import SearchResult
from genio.evaluation.executor import EvaluationExecutor
from genio.evaluation.step import EvaluationStep
from genio.evaluation.workflow import EvaluationWorkflow
from genio.objective import ObjectiveSet
from genio.objective.runtime import (
    EvaluatedBatch,
    ObjectiveRuntime,
)
from genio.search_space.space import SearchSpace
from genio.statistics.base import InMemoryStatistics, StatisticsCollector


class OptimizationSession:
    """Coordinate the complete lifecycle of one optimization run.

    A session drives the stateful algorithm through its ``ask``/``tell``
    protocol, evaluates each proposed batch, notifies the statistics collector,
    and optionally persists completed-batch checkpoints. It is the integration
    point between the otherwise independent framework components.

    The session owns run state, but it does not own the backend's lifecycle. Use
    the backend as a context manager or call ``backend.shutdown()`` when its
    resources are no longer needed.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        algorithm: SearchAlgorithm,
        backend: Backend,
        evaluation_workflow: EvaluationWorkflow,
        statistics: StatisticsCollector | None = None,
        id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        artifact_cache: ArtifactCache | None = None,
        checkpoint_policy: CheckpointPolicy | None = None,
        objective_set: ObjectiveSet | None = None,
    ) -> None:
        """Configure an optimization session without starting it.

        Args:
            search_space: Factory and finite domain from which candidates come.
            algorithm: Stateful strategy implementing the ``ask``/``tell``
                protocol.
            backend: Execution mechanism used by the workflow executor.
            evaluation_workflow: Dependency-ordered steps run for every candidate.
            statistics: Hook receiver for session events. Defaults to an in-memory
                counter and history collector.
            id: Logical session identifier. Defaults to the scenario identifier.
            run_id: Identifier for this concrete execution lineage. A random UUID
                is generated when omitted and restored from a checkpoint on resume.
            metadata: Application-defined values included in compatibility checks.
            artifact_cache: Optional session-scoped cache shared by evaluation
                steps.
            checkpoint_policy: Optional persistence and restoration policy.
            objective_set: Objective definitions and transformation strategies.
                Algorithms that do not consume objectives may omit it.

        Note:
            Checkpoint compatibility is validated when :meth:`run` starts. Active
            artifact caching and checkpointing cannot currently be combined.
        """
        self.id = id or search_space.scenario_id
        self.run_id = run_id or uuid4().hex
        self._configured_run_id = run_id
        self.search_space = search_space
        self.algorithm = algorithm
        self.backend = backend
        self.evaluation_workflow = evaluation_workflow
        self.artifact_cache = artifact_cache
        self.evaluation_executor = EvaluationExecutor(
            self.evaluation_workflow,
            backend,
            artifact_cache=artifact_cache,
        )
        self.statistics = statistics if statistics is not None else InMemoryStatistics()
        self.metadata = metadata or {}
        self.checkpoint_policy = checkpoint_policy
        self.objective_set = objective_set
        self._objective_runtime = (
            objective_set.bind() if objective_set is not None else None
        )
        normalizer = objective_set.normalizer if objective_set is not None else None
        normalization_scope = getattr(normalizer, "scope", None)
        if normalization_scope is not None:
            normalization_scope = getattr(normalization_scope, "value", normalization_scope)
        self._algorithm_context = SearchContext(
            search_space=search_space,
            objective_schema=(
                objective_set.schema if objective_set is not None else None
            ),
            has_normalizer=normalizer is not None,
            has_scalarizer=(
                objective_set is not None and objective_set.scalarizer is not None
            ),
            normalization_scope=normalization_scope,
        )
        self._algorithm_configured = False
        self._checkpoint_store = (
            JSONCheckpointStore(checkpoint_policy)
            if checkpoint_policy is not None
            else None
        )
        self._checkpoint_compatibility: dict[str, Any] | None = None
        self._next_proposal_sequence = 0
        self._next_batch_index = 0
        self._evaluations: list[Evaluation] = []
        self._evaluated_batches: list[EvaluatedBatch] = []
        self._started = False
        self._restored = False
        self._batch_in_progress = False
        self._finalized = False
        self._completed = False
        self._failed = False

    @property
    def objective_runtime(self) -> ObjectiveRuntime | None:
        """Return the session-owned objective runtime without allowing replacement."""

        return self._objective_runtime

    @property
    def failed(self) -> bool:
        """Return whether execution crossed the terminal failure boundary."""

        return self._failed

    def run(self) -> SearchResult:
        """Run or resume the optimization loop and return its final result.

        Each iteration calls ``ask``, starts and evaluates the complete batch,
        emits per-evaluation hooks, calls ``tell``, and finally emits the batch
        completion hook. Checkpoints are written only after that sequence has
        completed, making a completed batch the transactional boundary of a run.

        Returns:
            Evaluation history, algorithm-selected best individuals, and the
            final statistics snapshot.

        Note:
            Calling this method again after successful completion returns the
            existing result. Backend resources are not shut down automatically.
        """

        try:
            return self._run()
        finally:
            if self._checkpoint_store is not None:
                self._checkpoint_store.release_session_lease()

    def _run(self) -> SearchResult:
        """Execute the loop while :meth:`run` owns lease cleanup."""

        self._ensure_not_failed("run")
        self._prepare_run()
        if self._completed:
            return self._build_result()
        try:
            if self._finalized:
                result = self._build_result()
                if (
                    self._checkpoint_store is not None
                    and self.checkpoint_policy is not None
                    and self.checkpoint_policy.save_on_completion
                ):
                    self._persist_checkpoint(status="completed")
                self._completed = True
                return result

            while not self.algorithm.should_stop():
                self._batch_in_progress = True
                try:
                    individuals = tuple(self.algorithm.ask())
                    if not individuals:
                        if not self.algorithm.should_stop():
                            raise RuntimeError(
                                f"{type(self.algorithm).__name__}.ask() returned an empty "
                                "batch without reaching its stopping condition."
                            )
                        break

                    batch_index = self._next_batch_index
                    self.statistics.on_batch_started(batch_index, individuals)
                    batch_evaluations = self.evaluate(
                        individuals,
                        batch_index=batch_index,
                    )
                    evaluated_batch = self._evaluate_objectives(
                        batch_evaluations,
                        batch_index=batch_index,
                    )

                    for evaluation in batch_evaluations:
                        self.statistics.on_evaluation_completed(evaluation)
                    self.algorithm.tell(evaluated_batch)

                    # Commit order is intentional: tell, authoritative session
                    # history, observer hooks, then checkpoint publication.
                    self._evaluations.extend(batch_evaluations)
                    self._evaluated_batches.append(evaluated_batch)
                    self._next_batch_index += 1
                    self.statistics.on_evaluated_batch(evaluated_batch)
                    self.statistics.on_batch_completed(batch_index, batch_evaluations)

                    if (
                        self._checkpoint_store is not None
                        and self._checkpoint_store.should_save(self._next_batch_index)
                    ):
                        self._persist_checkpoint(status="running")
                finally:
                    self._batch_in_progress = False

            result = self._build_result()
            self.statistics.on_session_completed(result)
            result = replace(result, statistics=self.statistics.snapshot())
            self._finalized = True
            if (
                self._checkpoint_store is not None
                and self.checkpoint_policy is not None
                and self.checkpoint_policy.save_on_completion
            ):
                self._persist_checkpoint(status="completed")
            self._completed = True
            return result
        except BaseException:
            self._failed = True
            raise

    def save_checkpoint(self) -> Any:
        """Persist the current completed-batch state immediately.

        Returns:
            The numbered checkpoint path. If the policy is non-strict, a failed
            save emits a warning and returns ``None``.

        Raises:
            CheckpointStateError: If checkpointing is disabled, the session has
                not started, or an evaluation batch is currently in progress.
            Exception: The original serialization or storage error when persistence
                fails under a strict policy.
        """

        self._ensure_not_failed("save a checkpoint")
        if self._checkpoint_store is None or self.checkpoint_policy is None:
            raise CheckpointStateError("This session has no checkpoint policy.")
        if not self._started:
            raise CheckpointStateError("The session must be started before checkpointing.")
        if self._batch_in_progress:
            raise CheckpointStateError("Cannot checkpoint while a batch is in progress.")
        status = "completed" if self._finalized else "running"
        acquired_here = not self._checkpoint_store.session_lease_held
        if acquired_here:
            self._checkpoint_store.acquire_session_lease()
        try:
            return self._persist_checkpoint(status=status)
        finally:
            if acquired_here:
                self._checkpoint_store.release_session_lease()

    def _persist_checkpoint(self, *, status: str) -> Any:
        assert self._checkpoint_store is not None
        assert self.checkpoint_policy is not None
        try:
            payload = self._checkpoint_payload(status=status)
            return self._checkpoint_store.save(
                payload,
                sequence=self._next_batch_index,
            )
        except Exception as exc:
            if self.checkpoint_policy.strict:
                raise
            warnings.warn(f"Could not save optimization checkpoint: {exc}", stacklevel=2)
            return None

    def evaluate(
        self,
        individuals: Sequence[Individual],
        *,
        batch_index: int | None = None,
    ) -> list[Evaluation]:
        """Evaluate individuals and attach proposal provenance to their results.

        This lower-level entry point allocates proposal IDs, invokes
        ``on_proposals_generated``, and executes the workflow. It deliberately
        does not call ``algorithm.tell``, append to session history, or emit the
        completion hooks; :meth:`run` performs those commit operations.

        Args:
            individuals: Ordered candidates to evaluate. Their IDs must be unique
                within the batch.
            batch_index: Optional batch number included in proposal metadata.

        Returns:
            Evaluations in the same order as ``individuals``.
        """
        proposals = self._create_proposals(individuals, batch_index=batch_index)
        self.statistics.on_proposals_generated(proposals)
        results = self.evaluation_executor.evaluate_many(
            tuple(proposal.individual for proposal in proposals)
        )
        return [
            Evaluation(
                individual=proposal.individual,
                result=result,
                metadata=proposal.evaluation_metadata(),
            )
            for proposal, result in zip(proposals, results, strict=True)
        ]

    def _create_proposals(
        self,
        individuals: Sequence[Individual],
        *,
        batch_index: int | None,
    ) -> tuple[Proposal, ...]:
        """Allocate run-scoped provenance for each candidate occurrence."""
        proposals = tuple(
            Proposal(
                proposal_id=f"{self.run_id}:{self._next_proposal_sequence + position:06d}",
                proposal_sequence=self._next_proposal_sequence + position,
                batch_index=batch_index,
                batch_position=position,
                individual=individual,
            )
            for position, individual in enumerate(individuals)
        )
        self._next_proposal_sequence += len(proposals)
        return proposals

    def _prepare_run(self) -> None:
        """Initialize a fresh run or restore its configured checkpoint lineage."""
        if not self._algorithm_configured:
            self.algorithm.configure(self._algorithm_context)
            self._algorithm_configured = True
        if self._started:
            if self._checkpoint_store is not None:
                self._checkpoint_store.acquire_session_lease()
            return
        if self._checkpoint_store is None or self.checkpoint_policy is None:
            if self.artifact_cache is not None:
                self.artifact_cache.clear()
            self.statistics.on_session_started(self)
            self._started = True
            return

        self._validate_checkpoint_support()
        self._checkpoint_store.acquire_session_lease()
        if self.checkpoint_policy.resume_from is not None:
            checkpoint = self._checkpoint_store.load(self.checkpoint_policy.resume_from)
            self._restore_checkpoint(checkpoint)
            self._restored = True
        else:
            if self._checkpoint_store.latest_path.exists():
                raise CheckpointStateError(
                    f"Checkpoint directory already contains {self._checkpoint_store.latest_path}; "
                    "configure resume_from or use another directory."
                )
            if self.artifact_cache is not None:
                self.artifact_cache.clear()
            self._checkpoint_compatibility = self._compatibility_signature()
            self.statistics.on_session_started(self)
        self._started = True

    def _validate_checkpoint_support(self) -> None:
        """Ensure every stateful component can participate in restoration."""
        if not self.algorithm.supports_checkpointing:
            raise CheckpointNotSupportedError(
                f"Algorithm {type(self.algorithm).__name__} does not support checkpointing."
            )
        if not self.statistics.supports_checkpointing:
            raise CheckpointNotSupportedError(
                f"Statistics collector {type(self.statistics).__name__} does not support "
                "checkpointing."
            )
        if self.artifact_cache is not None:
            raise CheckpointNotSupportedError(
                "Checkpointing with an artifact cache is not supported until cache "
                "entries and telemetry can be restored deterministically."
            )
        if type(self.backend).checkpoint_signature is Backend.checkpoint_signature:
            raise CheckpointNotSupportedError(
                f"Backend {type(self.backend).__name__} must override "
                "checkpoint_signature() to support checkpointing."
            )
        unsupported_steps = [
            step.id
            for step in self.evaluation_workflow.steps
            if type(step).checkpoint_signature is EvaluationStep.checkpoint_signature
        ]
        if unsupported_steps:
            raise CheckpointNotSupportedError(
                "Evaluation steps must override checkpoint_signature(): "
                + ", ".join(unsupported_steps)
            )

    def _checkpoint_payload(self, *, status: str) -> dict[str, Any]:
        """Assemble snapshot data for a committed session boundary.

        The checkpoint store performs the final JSON-compatibility validation when
        this payload is persisted.
        """
        self._assert_checkpoint_compatibility_stable()
        assert self._checkpoint_compatibility is not None
        return {
            "status": status,
            "compatibility": dict(self._checkpoint_compatibility),
            "session": {
                "session_id": self.id,
                "run_id": self.run_id,
                "next_batch_index": self._next_batch_index,
                "next_proposal_sequence": self._next_proposal_sequence,
                "backend_run_id": getattr(self.backend, "run_id", None),
                "metadata": self.metadata,
            },
            "search_space_state": self.search_space.checkpoint_state(),
            "algorithm": self.algorithm.checkpoint_state(),
            "statistics": self.statistics.checkpoint_state(),
            "evaluations": [
                encode_evaluation(evaluation) for evaluation in self._evaluations
            ],
            "evaluated_batches": [
                encode_evaluated_batch(batch) for batch in self._evaluated_batches
            ],
            "objective_runtime": (
                self._objective_runtime.checkpoint_state()
                if self._objective_runtime is not None
                else None
            ),
        }

    def _restore_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Validate and restore a complete cross-component session snapshot.

        Compatibility fingerprints are checked before mutable component state is
        restored. Evaluation counters, proposal provenance, algorithm counters,
        search-space IDs, and statistics state are then cross-validated against
        the session-owned batch history.

        Raises:
            CheckpointCompatibilityError: If configuration or persisted state no
                longer describes the configured session.
        """
        try:
            checkpoint_compatibility = dict(checkpoint["compatibility"])
            session_state = dict(checkpoint["session"])
            algorithm_state = dict(checkpoint["algorithm"])
            statistics_state = dict(checkpoint["statistics"])
            status = str(checkpoint["status"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError("Checkpoint session payload is invalid.") from exc
        if status not in {"running", "completed"}:
            raise CheckpointCompatibilityError(f"Unknown checkpoint status {status!r}.")

        expected_compatibility = self._compatibility_signature()
        for name, expected in expected_compatibility.items():
            actual = checkpoint_compatibility.get(name)
            if actual != expected:
                raise CheckpointCompatibilityError(
                    f"Checkpoint compatibility mismatch for {name!r}: "
                    f"expected {expected!r}, got {actual!r}."
                )
        if session_state.get("session_id") != self.id:
            raise CheckpointCompatibilityError("Checkpoint session_id does not match.")
        checkpoint_run_id = session_state.get("run_id")
        if not isinstance(checkpoint_run_id, str) or not checkpoint_run_id:
            raise CheckpointCompatibilityError("Checkpoint run_id is invalid.")
        if self._configured_run_id is not None and self._configured_run_id != checkpoint_run_id:
            raise CheckpointCompatibilityError("Configured run_id does not match checkpoint.")

        try:
            evaluations = [
                decode_evaluation(value, self.search_space)
                for value in checkpoint.get("evaluations", [])
            ]
            next_batch_index = session_state["next_batch_index"]
            next_proposal_sequence = session_state["next_proposal_sequence"]
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError("Checkpoint counters are invalid.") from exc
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                next_batch_index,
                next_proposal_sequence,
            )
        ):
            raise CheckpointCompatibilityError("Checkpoint counters cannot be negative.")
        self._validate_restored_evaluations(
            evaluations,
            next_batch_index=next_batch_index,
            next_proposal_sequence=next_proposal_sequence,
            run_id=checkpoint_run_id,
        )
        self._validate_component_state(
            checkpoint,
            evaluations=evaluations,
            next_batch_index=next_batch_index,
            next_proposal_sequence=next_proposal_sequence,
        )
        try:
            evaluations_by_batch: dict[int, list[Evaluation]] = {}
            for evaluation in evaluations:
                evaluations_by_batch.setdefault(
                    int(evaluation.metadata["batch_index"]), []
                ).append(evaluation)
            evaluated_batch_values = checkpoint.get("evaluated_batches", [])
            if len(evaluated_batch_values) != next_batch_index:
                raise ValueError("Evaluated batch count is inconsistent.")
            evaluated_batches = [
                decode_evaluated_batch(
                    value,
                    evaluations=evaluations_by_batch[index],
                    normalizer=(
                        self.objective_set.normalizer
                        if self.objective_set is not None
                        else None
                    ),
                )
                for index, value in enumerate(evaluated_batch_values)
            ]
            if any(
                batch.batch_index != index
                for index, batch in enumerate(evaluated_batches)
            ):
                raise ValueError("Evaluated batch indexes are inconsistent.")
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                "Checkpoint objective batch history is invalid."
            ) from exc

        objective_runtime_state = checkpoint.get("objective_runtime")
        if self._objective_runtime is None:
            if objective_runtime_state is not None:
                raise CheckpointCompatibilityError(
                    "Checkpoint contains objective state but the session has no objectives."
                )
        else:
            if not isinstance(objective_runtime_state, Mapping):
                raise CheckpointCompatibilityError(
                    "Checkpoint has no valid objective runtime state."
                )
            try:
                self._objective_runtime.restore_checkpoint_state(objective_runtime_state)
            except (TypeError, ValueError) as exc:
                raise CheckpointCompatibilityError(
                    "Checkpoint objective runtime state is invalid."
                ) from exc
        self.algorithm.restore_checkpoint_state(
            algorithm_state,
            search_space=self.search_space,
            evaluated_batches=evaluated_batches,
        )
        if status == "completed" and not self.algorithm.should_stop():
            raise CheckpointCompatibilityError(
                "Completed checkpoint contains an unfinished algorithm state."
            )

        self.search_space.restore_checkpoint_state(checkpoint["search_space_state"])
        self._checkpoint_compatibility = expected_compatibility
        self.run_id = checkpoint_run_id
        backend_run_id = session_state.get("backend_run_id")
        if backend_run_id is not None and hasattr(self.backend, "run_id"):
            self.backend.run_id = str(backend_run_id)
        self._evaluations = evaluations
        self._evaluated_batches = evaluated_batches
        self._next_batch_index = next_batch_index
        self._next_proposal_sequence = next_proposal_sequence
        self._finalized = status == "completed"
        self._completed = status == "completed"
        if self.artifact_cache is not None:
            self.artifact_cache.clear()
        self.statistics.restore_checkpoint_state(
            statistics_state,
            session=self,
            evaluations=evaluations,
            evaluated_batches=evaluated_batches,
            completed=status == "completed",
        )

    def _compatibility_signature(self) -> dict[str, Any]:
        """Fingerprint configuration that must remain stable across a resume."""
        assert self._checkpoint_store is not None
        assert self.checkpoint_policy is not None
        workflow = [
            signature_value(step.checkpoint_signature())
            for step in self.evaluation_workflow.execution_order()
        ]
        return {
            "algorithm_type": qualified_name(self.algorithm),
            "algorithm_configuration": self._checkpoint_store.fingerprint(
                signature_value(self.algorithm.checkpoint_signature())
            ),
            "objective_set": (
                self._checkpoint_store.fingerprint(
                    signature_value(self.objective_set.checkpoint_signature())
                )
                if self.objective_set is not None
                else None
            ),
            "search_space": self._checkpoint_store.fingerprint(
                signature_value(self.search_space.checkpoint_signature())
            ),
            "workflow": self._checkpoint_store.fingerprint(workflow),
            "backend": self._checkpoint_store.fingerprint(
                signature_value(self.backend.checkpoint_signature())
            ),
            "statistics_type": qualified_name(self.statistics),
            "statistics_configuration": self._checkpoint_store.fingerprint(
                signature_value(self.statistics.checkpoint_signature())
            ),
            "session_metadata": self._checkpoint_store.fingerprint(
                signature_value(self.metadata)
            ),
            "compatibility_tag": self.checkpoint_policy.compatibility_tag,
        }

    def _assert_checkpoint_compatibility_stable(self) -> None:
        """Reject checkpoint-relevant configuration mutated during a run."""
        current = self._compatibility_signature()
        if self._checkpoint_compatibility is None:
            self._checkpoint_compatibility = current
            return
        if current != self._checkpoint_compatibility:
            raise CheckpointCompatibilityError(
                "Checkpoint-relevant configuration changed during the session."
            )

    def _build_result(self) -> SearchResult:
        """Create a result snapshot from committed session state."""
        return SearchResult(
            session_id=self.id,
            evaluations=tuple(self._evaluations),
            run_id=self.run_id,
            best_individuals=tuple(self.algorithm.best_individuals()),
            statistics=self.statistics.snapshot(),
        )

    def _evaluate_objectives(
        self,
        evaluations: Sequence[Evaluation],
        *,
        batch_index: int,
    ) -> EvaluatedBatch:
        """Build the objective-aware batch delivered to search algorithms."""

        if self._objective_runtime is not None:
            return self._objective_runtime.evaluate_batch(
                evaluations,
                batch_index=batch_index,
            )
        return EvaluatedBatch.from_evaluations(
            evaluations,
            batch_index=batch_index,
        )

    @staticmethod
    def _validate_restored_evaluations(
        evaluations: Sequence[Evaluation],
        *,
        next_batch_index: int,
        next_proposal_sequence: int,
        run_id: str,
    ) -> None:
        if len(evaluations) != next_proposal_sequence:
            raise CheckpointCompatibilityError(
                "Checkpoint proposal counter does not match evaluation history."
            )
        sequences: list[int] = []
        proposal_ids: list[str] = []
        batches: dict[int, list[int]] = {}
        for evaluation in evaluations:
            sequence = evaluation.metadata.get("proposal_sequence")
            batch_index = evaluation.metadata.get("batch_index")
            batch_position = evaluation.metadata.get("batch_position")
            proposal_id = evaluation.metadata.get("proposal_id")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or isinstance(batch_index, bool)
                or not isinstance(batch_index, int)
                or isinstance(batch_position, bool)
                or not isinstance(batch_position, int)
                or not isinstance(proposal_id, str)
            ):
                raise CheckpointCompatibilityError(
                    "Checkpoint evaluation proposal metadata is invalid."
                )
            sequences.append(sequence)
            proposal_ids.append(proposal_id)
            batches.setdefault(batch_index, []).append(batch_position)
        if sequences != list(range(next_proposal_sequence)):
            raise CheckpointCompatibilityError(
                "Checkpoint proposal sequences are not contiguous."
            )
        expected_proposal_ids = [
            f"{run_id}:{sequence:06d}" for sequence in range(next_proposal_sequence)
        ]
        if proposal_ids != expected_proposal_ids:
            raise CheckpointCompatibilityError(
                "Checkpoint proposal IDs are inconsistent with its run lineage."
            )
        if sorted(batches) != list(range(next_batch_index)):
            raise CheckpointCompatibilityError(
                "Checkpoint batch indexes are not contiguous."
            )
        for positions in batches.values():
            if positions != list(range(len(positions))):
                raise CheckpointCompatibilityError(
                    "Checkpoint batch positions are not contiguous."
                )

    @staticmethod
    def _validate_component_state(
        checkpoint: dict[str, Any],
        *,
        evaluations: Sequence[Evaluation],
        next_batch_index: int,
        next_proposal_sequence: int,
    ) -> None:
        algorithm_state = checkpoint["algorithm"]
        statistics_state = checkpoint["statistics"]
        search_state = checkpoint.get("search_space_state", {})
        if not isinstance(algorithm_state, Mapping):
            raise CheckpointCompatibilityError("Algorithm checkpoint state is invalid.")
        if not isinstance(statistics_state, Mapping):
            raise CheckpointCompatibilityError("Statistics checkpoint state is invalid.")
        if not isinstance(search_state, Mapping):
            raise CheckpointCompatibilityError("Search-space checkpoint state is invalid.")
        next_id = search_state.get("next_id")
        if isinstance(next_id, bool) or not isinstance(next_id, int):
            raise CheckpointCompatibilityError("Search-space next_id is invalid.")
        if next_id < next_proposal_sequence:
            raise CheckpointCompatibilityError(
                "Search-space ID allocator precedes committed proposals."
            )

        completed_batches = statistics_state.get("completed_batches")
        completed_evaluations = statistics_state.get("completed_evaluations")
        if completed_batches is not None and completed_batches != next_batch_index:
            raise CheckpointCompatibilityError(
                "Statistics batch count differs from session state."
            )
        if completed_evaluations is not None and completed_evaluations != len(evaluations):
            raise CheckpointCompatibilityError(
                "Statistics evaluation count differs from session state."
            )
        proposal_order = statistics_state.get("proposal_order")
        if proposal_order is not None:
            expected_order = [
                str(evaluation.metadata["proposal_id"])
                for evaluation in evaluations
            ]
            if list(proposal_order) != expected_order:
                raise CheckpointCompatibilityError(
                    "Statistics proposal order differs from session history."
                )

    def _ensure_not_failed(self, operation: str) -> None:
        if self._failed:
            raise CheckpointStateError(
                f"OptimizationSession is FAILED and cannot {operation}."
            )
