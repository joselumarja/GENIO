from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import warnings
from typing import Any
from uuid import uuid4

from genio.algorithm.base import SearchAlgorithm
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
    encode_evaluation,
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
from genio.search_space.space import SearchSpace
from genio.statistics.base import InMemoryStatistics, StatisticsCollector


class OptimizationSession:
    """Coordinates search space, algorithm, backend execution and statistics."""

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
    ) -> None:
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
        self._checkpoint_store = (
            JSONCheckpointStore(checkpoint_policy)
            if checkpoint_policy is not None
            else None
        )
        self._checkpoint_compatibility: dict[str, Any] | None = None
        self._next_proposal_sequence = 0
        self._next_batch_index = 0
        self._evaluations: list[Evaluation] = []
        self._started = False
        self._restored = False
        self._batch_in_progress = False
        self._finalized = False
        self._completed = False

    def run(self) -> SearchResult:
        """Run the optimization loop and return its search result."""

        try:
            return self._run()
        finally:
            if self._checkpoint_store is not None:
                self._checkpoint_store.release_session_lease()

    def _run(self) -> SearchResult:
        """Execute the optimization loop while run() owns lifecycle cleanup."""

        self._prepare_run()
        if self._completed:
            return self._build_result()
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
                individuals = tuple(self.algorithm.ask(self))
            except Exception:
                raise
            if not individuals:
                self._batch_in_progress = False
                break

            batch_index = self._next_batch_index
            self.statistics.on_batch_started(batch_index, individuals)
            batch_evaluations = self.evaluate(individuals, batch_index=batch_index)

            for evaluation in batch_evaluations:
                self.statistics.on_evaluation_completed(evaluation)
            self.algorithm.tell(batch_evaluations)
            self.statistics.on_batch_completed(batch_index, batch_evaluations)
            self._evaluations.extend(batch_evaluations)
            self._next_batch_index += 1
            self._batch_in_progress = False
            if (
                self._checkpoint_store is not None
                and self._checkpoint_store.should_save(self._next_batch_index)
            ):
                self._persist_checkpoint(status="running")

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

    def save_checkpoint(self) -> Any:
        """Persist the current completed-batch session state immediately."""

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
        """Evaluate a batch of individuals and wrap their results."""
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
            "algorithm": {
                "version": self.algorithm.checkpoint_version,
                "state": self.algorithm.checkpoint_state(),
            },
            "statistics": {
                "version": self.statistics.checkpoint_version,
                "state": self.statistics.checkpoint_state(),
            },
            "evaluations": [
                encode_evaluation(evaluation) for evaluation in self._evaluations
            ],
        }

    def _restore_checkpoint(self, checkpoint: dict[str, Any]) -> None:
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
        self._checkpoint_compatibility = expected_compatibility
        if session_state.get("session_id") != self.id:
            raise CheckpointCompatibilityError("Checkpoint session_id does not match.")
        checkpoint_run_id = str(session_state.get("run_id"))
        if self._configured_run_id is not None and self._configured_run_id != checkpoint_run_id:
            raise CheckpointCompatibilityError("Configured run_id does not match checkpoint.")

        try:
            evaluations = [
                decode_evaluation(value, self.search_space)
                for value in checkpoint.get("evaluations", [])
            ]
            next_batch_index = int(session_state["next_batch_index"])
            next_proposal_sequence = int(session_state["next_proposal_sequence"])
            algorithm_version = int(algorithm_state["version"])
            collector_version = int(statistics_state["version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError("Checkpoint counters are invalid.") from exc
        if next_batch_index < 0 or next_proposal_sequence < 0:
            raise CheckpointCompatibilityError("Checkpoint counters cannot be negative.")
        self._validate_restored_evaluations(
            evaluations,
            next_batch_index=next_batch_index,
            next_proposal_sequence=next_proposal_sequence,
        )
        self._validate_cross_component_state(
            checkpoint,
            evaluations=evaluations,
            next_batch_index=next_batch_index,
            next_proposal_sequence=next_proposal_sequence,
        )
        if collector_version != self.statistics.checkpoint_version:
            raise CheckpointCompatibilityError(
                f"Unsupported statistics checkpoint version {collector_version}."
            )

        self.run_id = checkpoint_run_id
        backend_run_id = session_state.get("backend_run_id")
        if backend_run_id is not None and hasattr(self.backend, "run_id"):
            self.backend.run_id = str(backend_run_id)
        self.search_space.restore_checkpoint_state(checkpoint["search_space_state"])
        self.algorithm.restore_checkpoint_state(
            algorithm_state["state"],
            version=algorithm_version,
            search_space=self.search_space,
        )
        self._evaluations = evaluations
        self._next_batch_index = next_batch_index
        self._next_proposal_sequence = next_proposal_sequence
        self._finalized = status == "completed"
        self._completed = status == "completed"
        if self._completed and not self.algorithm.should_stop():
            raise CheckpointCompatibilityError(
                "Completed checkpoint contains an unfinished algorithm state."
            )
        if self.artifact_cache is not None:
            self.artifact_cache.clear()
        self.statistics.restore_checkpoint_state(
            statistics_state["state"],
            session=self,
            evaluations=evaluations,
            completed=status == "completed",
        )

    def _compatibility_signature(self) -> dict[str, Any]:
        assert self._checkpoint_store is not None
        workflow = [
            signature_value(step.checkpoint_signature())
            for step in self.evaluation_workflow.execution_order()
        ]
        return {
            "algorithm_type": qualified_name(self.algorithm),
            "algorithm_version": self.algorithm.checkpoint_version,
            "algorithm_configuration": self._checkpoint_store.fingerprint(
                signature_value(self.algorithm.checkpoint_signature())
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
        current = self._compatibility_signature()
        if self._checkpoint_compatibility is None:
            self._checkpoint_compatibility = current
            return
        if current != self._checkpoint_compatibility:
            raise CheckpointCompatibilityError(
                "Checkpoint-relevant configuration changed during the session."
            )

    def _build_result(self) -> SearchResult:
        return SearchResult(
            session_id=self.id,
            evaluations=tuple(self._evaluations),
            run_id=self.run_id,
            best_individuals=tuple(self.algorithm.best_individuals()),
            statistics=self.statistics.snapshot(),
        )

    @staticmethod
    def _validate_restored_evaluations(
        evaluations: Sequence[Evaluation],
        *,
        next_batch_index: int,
        next_proposal_sequence: int,
    ) -> None:
        if len(evaluations) != next_proposal_sequence:
            raise CheckpointCompatibilityError(
                "Checkpoint proposal counter does not match evaluation history."
            )
        sequences: list[int] = []
        batches: dict[int, list[int]] = {}
        for evaluation in evaluations:
            sequence = evaluation.metadata.get("proposal_sequence")
            batch_index = evaluation.metadata.get("batch_index")
            batch_position = evaluation.metadata.get("batch_position")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or isinstance(batch_index, bool)
                or not isinstance(batch_index, int)
                or isinstance(batch_position, bool)
                or not isinstance(batch_position, int)
            ):
                raise CheckpointCompatibilityError(
                    "Checkpoint evaluation proposal metadata is invalid."
                )
            sequences.append(sequence)
            batches.setdefault(batch_index, []).append(batch_position)
        if sorted(sequences) != list(range(next_proposal_sequence)):
            raise CheckpointCompatibilityError(
                "Checkpoint proposal sequences are not contiguous."
            )
        if sorted(batches) != list(range(next_batch_index)):
            raise CheckpointCompatibilityError(
                "Checkpoint batch indexes are not contiguous."
            )
        for positions in batches.values():
            if sorted(positions) != list(range(len(positions))):
                raise CheckpointCompatibilityError(
                    "Checkpoint batch positions are not contiguous."
                )

    @staticmethod
    def _validate_cross_component_state(
        checkpoint: dict[str, Any],
        *,
        evaluations: Sequence[Evaluation],
        next_batch_index: int,
        next_proposal_sequence: int,
    ) -> None:
        algorithm_state = checkpoint["algorithm"]["state"]
        statistics_state = checkpoint["statistics"]["state"]
        search_state = checkpoint.get("search_space_state", {})
        if not isinstance(algorithm_state, Mapping):
            raise CheckpointCompatibilityError("Algorithm checkpoint state is invalid.")
        if not isinstance(statistics_state, Mapping):
            raise CheckpointCompatibilityError("Statistics checkpoint state is invalid.")
        if not isinstance(search_state, Mapping):
            raise CheckpointCompatibilityError("Search-space checkpoint state is invalid.")
        algorithm_evaluations = algorithm_state.get("evaluations")
        session_evaluations = checkpoint.get("evaluations", [])
        if algorithm_evaluations != session_evaluations:
            raise CheckpointCompatibilityError(
                "Algorithm and session evaluation histories differ."
            )

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
