"""Lifecycle collector for population-oriented search analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any

from genio.checkpoint.codec import qualified_name
from genio.core.evaluation import Evaluation
from genio.core.individual import Individual
from genio.core.proposal import Proposal
from genio.core.search_result import SearchResult
from genio.objective.runtime import EvaluatedBatch, EvaluatedIndividual
from genio.statistics.base import StatisticsCollector
from genio.statistics.population_analysis import (
    duplicate_genotype_ratio,
    failure_rate_by_batch,
    gene_entropy,
    mean_pairwise_hamming_distance,
    objective_summary_by_batch,
    score_summary_by_batch,
    stage_frequency_by_slot,
    unique_genotype_ratio,
)
from genio.statistics.population_models import PopulationRecord, PopulationSnapshot
from genio.statistics.population_plots import (
    PopulationPlotConfig,
    PopulationPlotRenderer,
    PopulationPlotResult,
)

if TYPE_CHECKING:
    from genio.session.optimization import OptimizationSession


class PopulationAnalysisError(ValueError):
    """Raised when population lifecycle events are incomplete or inconsistent."""


class PopulationAnalysisCollector(StatisticsCollector):
    """Capture immutable population snapshots from session lifecycle events.

    This phase stores analysis-ready data only; it does not write files or render
    plots. Proposals and objective results are correlated by ``proposal_id`` so
    repeated individuals and genotypes remain distinct.

    Args:
        output_dir: Directory containing summaries, manifest, and plot folders.
        plots: Plot cadence, format, selected genes, and failure policy.
    """

    supports_checkpointing = True

    def __init__(
        self,
        output_dir: str | Path,
        *,
        plots: PopulationPlotConfig | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.plots = plots or PopulationPlotConfig()
        self._renderer = PopulationPlotRenderer(self.plots)
        self._records: dict[str, PopulationRecord] = {}
        self._batch_proposal_ids: dict[int, tuple[str, ...]] = {}
        self._started_individual_ids: dict[int, tuple[str, ...]] = {}
        self._snapshots: list[PopulationSnapshot] = []
        self._best_individual_ids: tuple[str, ...] = ()
        self._plot_results: dict[str, PopulationPlotResult] = {}

    @property
    def summary_path(self) -> Path:
        """Return the final numerical analysis summary path."""

        return self.output_dir / "analysis_summary.json"

    @property
    def manifest_path(self) -> Path:
        """Return the generated/skipped/warnings manifest path."""

        return self.output_dir / "analysis_manifest.json"

    @property
    def snapshots(self) -> tuple[PopulationSnapshot, ...]:
        """Return completed populations in batch order."""

        return tuple(self._snapshots)

    @property
    def records(self) -> tuple[PopulationRecord, ...]:
        """Return all proposal records in batch and position order."""

        return tuple(
            record
            for snapshot in self._snapshots
            for record in snapshot.records
        )

    @property
    def best_individual_ids(self) -> tuple[str, ...]:
        """Return IDs selected by the algorithm at session completion."""

        return self._best_individual_ids

    def on_session_started(self, session: OptimizationSession) -> None:
        """Reset capture state for a fresh optimization session."""

        del session
        self._records.clear()
        self._batch_proposal_ids.clear()
        self._started_individual_ids.clear()
        self._snapshots.clear()
        self._best_individual_ids = ()
        self._plot_results.clear()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = self.output_dir / "plots"
        if plots_dir.exists():
            shutil.rmtree(plots_dir)
        for path in (self.summary_path, self.manifest_path):
            path.unlink(missing_ok=True)

    def on_batch_started(
        self,
        batch_index: int,
        individuals: Sequence[Individual],
    ) -> None:
        """Record the expected individuals for one population batch."""

        if batch_index in self._started_individual_ids:
            raise PopulationAnalysisError(
                f"Population batch {batch_index} has already started."
            )
        individual_ids = tuple(individual.id for individual in individuals)
        if len(set(individual_ids)) != len(individual_ids):
            raise PopulationAnalysisError(
                f"Population batch {batch_index} contains duplicate individual IDs."
            )
        self._started_individual_ids[batch_index] = individual_ids

    def on_proposals_generated(self, proposals: Sequence[Proposal]) -> None:
        """Create pending records for one generated population."""

        normalized = tuple(proposals)
        if not normalized:
            raise PopulationAnalysisError("A population proposal batch must not be empty.")
        batch_indexes = {proposal.batch_index for proposal in normalized}
        if len(batch_indexes) != 1 or None in batch_indexes:
            raise PopulationAnalysisError(
                "Population proposals must belong to one numbered batch."
            )
        batch_index = normalized[0].batch_index
        assert batch_index is not None
        expected_individual_ids = self._started_individual_ids.get(batch_index)
        if expected_individual_ids is None:
            raise PopulationAnalysisError(
                f"Population batch {batch_index} received proposals before start."
            )
        proposal_ids = tuple(proposal.proposal_id for proposal in normalized)
        if len(set(proposal_ids)) != len(proposal_ids):
            raise PopulationAnalysisError(
                f"Population batch {batch_index} contains duplicate proposal IDs."
            )
        if any(proposal_id in self._records for proposal_id in proposal_ids):
            raise PopulationAnalysisError("Population proposal IDs must be run-unique.")
        if tuple(proposal.batch_position for proposal in normalized) != tuple(
            range(len(normalized))
        ):
            raise PopulationAnalysisError(
                "Population proposal positions must be contiguous and ordered."
            )
        if tuple(proposal.individual.id for proposal in normalized) != expected_individual_ids:
            raise PopulationAnalysisError(
                f"Population batch {batch_index} proposals do not match started individuals."
            )
        self._batch_proposal_ids[batch_index] = proposal_ids
        self._records.update(
            (proposal.proposal_id, PopulationRecord.from_proposal(proposal))
            for proposal in normalized
        )

    def on_evaluated_batch(self, batch: EvaluatedBatch) -> None:
        """Attach evaluation and objective results to pending proposal records."""

        if batch.batch_index is None:
            raise PopulationAnalysisError("Evaluated population batch has no batch_index.")
        proposal_ids = self._batch_proposal_ids.get(batch.batch_index)
        if proposal_ids is None:
            raise PopulationAnalysisError(
                f"Evaluated population batch {batch.batch_index} was not proposed."
            )
        evaluated_by_proposal: dict[str, EvaluatedIndividual] = {}
        for item in batch:
            proposal_id = item.evaluation.metadata.get("proposal_id")
            if not isinstance(proposal_id, str) or not proposal_id:
                raise PopulationAnalysisError(
                    "Evaluated population item has no valid proposal_id."
                )
            if proposal_id in evaluated_by_proposal:
                raise PopulationAnalysisError(
                    f"Evaluated population batch contains duplicate {proposal_id!r}."
                )
            evaluated_by_proposal[proposal_id] = item
        if set(evaluated_by_proposal) != set(proposal_ids):
            raise PopulationAnalysisError(
                f"Evaluated population batch {batch.batch_index} does not match proposals."
            )
        for proposal_id in proposal_ids:
            self._records[proposal_id] = self._records[proposal_id].with_evaluation(
                evaluated_by_proposal[proposal_id]
            )

    def on_batch_completed(
        self,
        batch_index: int,
        evaluations: Sequence[Evaluation],
    ) -> None:
        """Close one population snapshot after all records are evaluated."""

        proposal_ids = self._batch_proposal_ids.get(batch_index)
        if proposal_ids is None:
            raise PopulationAnalysisError(
                f"Population batch {batch_index} completed before proposals."
            )
        evaluation_proposal_ids = tuple(
            evaluation.metadata.get("proposal_id") for evaluation in evaluations
        )
        if evaluation_proposal_ids != proposal_ids:
            raise PopulationAnalysisError(
                f"Population batch {batch_index} evaluation order is inconsistent."
            )
        records = tuple(self._records[proposal_id] for proposal_id in proposal_ids)
        if any(not record.evaluated for record in records):
            raise PopulationAnalysisError(
                f"Population batch {batch_index} has incomplete objective results."
            )
        if self._snapshots and batch_index <= self._snapshots[-1].batch_index:
            raise PopulationAnalysisError(
                "Population batches must complete in increasing order."
            )
        self._snapshots.append(PopulationSnapshot(batch_index, records))
        del self._batch_proposal_ids[batch_index]
        del self._started_individual_ids[batch_index]
        every_batches = self.plots.every_batches
        if every_batches is not None and len(self._snapshots) % every_batches == 0:
            scope = f"batch_{batch_index:06d}"
            self._plot_results[scope] = self._render_plots(
                scope,
                self._snapshots,
            )

    def on_session_completed(self, result: SearchResult) -> None:
        """Capture the algorithm-selected individual IDs after the final batch."""

        if self._batch_proposal_ids or self._started_individual_ids:
            raise PopulationAnalysisError(
                "Cannot complete population statistics with open batches."
            )
        self._best_individual_ids = tuple(
            individual.id for individual in result.best_individuals
        )
        if self.plots.final_plots:
            self._plot_results["final"] = self._render_plots(
                "final",
                self._snapshots,
                best_individual_ids=self._best_individual_ids,
            )
        self._write_json(self.summary_path, self._analysis_summary())
        self._write_json(self.manifest_path, self._analysis_manifest())

    def snapshot(self) -> dict[str, Any]:
        """Return lightweight capture counts and selected IDs."""

        records = self.records
        return {
            "output_dir": str(self.output_dir),
            "batches": len(self._snapshots),
            "records": len(records),
            "evaluated_records": sum(record.evaluated for record in records),
            "valid_objective_records": sum(
                record.objectives_valid for record in records
            ),
            "best_individual_ids": list(self._best_individual_ids),
            "analysis_summary": str(self.summary_path),
            "analysis_manifest": str(self.manifest_path),
            "generated_plots": sum(
                len(result.generated) for result in self._plot_results.values()
            ),
            "plot_warnings": sum(
                len(result.warnings) for result in self._plot_results.values()
            ),
        }

    def checkpoint_signature(self) -> dict[str, Any]:
        """Return immutable output configuration for session compatibility."""

        return {
            "type": qualified_name(self),
            "output_dir": str(self.output_dir.resolve()),
            "plots": asdict(self.plots),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Persist only indexes and final selection not owned by session history."""

        if self._batch_proposal_ids or self._started_individual_ids:
            raise PopulationAnalysisError(
                "Cannot checkpoint population statistics with an open batch."
            )
        return {
            "batch_indexes": [snapshot.batch_index for snapshot in self._snapshots],
            "best_individual_ids": list(self._best_individual_ids),
        }

    def restore_checkpoint_state(
        self,
        state: dict[str, Any],
        *,
        session: OptimizationSession,
        evaluations: Sequence[Evaluation],
        evaluated_batches: Sequence[EvaluatedBatch],
        completed: bool,
    ) -> None:
        """Rebuild records from authoritative session batches and regenerate outputs."""

        del session, evaluations
        if set(state) != {"batch_indexes", "best_individual_ids"}:
            raise PopulationAnalysisError(
                "Population analysis checkpoint fields are invalid."
            )
        batch_indexes = state["batch_indexes"]
        best_individual_ids = state["best_individual_ids"]
        if isinstance(batch_indexes, (str, bytes)) or not isinstance(
            batch_indexes, Sequence
        ):
            raise PopulationAnalysisError("batch_indexes must be a sequence.")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in batch_indexes
        ):
            raise PopulationAnalysisError(
                "batch_indexes must contain non-negative integers."
            )
        expected_indexes = [batch.batch_index for batch in evaluated_batches]
        if list(batch_indexes) != expected_indexes:
            raise PopulationAnalysisError(
                "Population checkpoint batches do not match session history."
            )
        if isinstance(best_individual_ids, (str, bytes)) or not isinstance(
            best_individual_ids, Sequence
        ):
            raise PopulationAnalysisError("best_individual_ids must be a sequence.")
        if any(
            not isinstance(identifier, str) or not identifier
            for identifier in best_individual_ids
        ):
            raise PopulationAnalysisError(
                "best_individual_ids must contain non-empty strings."
            )
        if not completed and best_individual_ids:
            raise PopulationAnalysisError(
                "An incomplete session cannot contain final best individual IDs."
            )

        records: dict[str, PopulationRecord] = {}
        snapshots: list[PopulationSnapshot] = []
        for batch in evaluated_batches:
            if batch.batch_index is None:
                raise PopulationAnalysisError(
                    "Restored population batch has no batch_index."
                )
            batch_records: list[PopulationRecord] = []
            for item in batch:
                metadata = item.evaluation.metadata
                try:
                    proposal = Proposal(
                        proposal_id=_checkpoint_string(
                            metadata["proposal_id"], "proposal_id"
                        ),
                        proposal_sequence=_checkpoint_integer(
                            metadata["proposal_sequence"], "proposal_sequence"
                        ),
                        batch_index=_checkpoint_integer(
                            metadata["batch_index"], "batch_index"
                        ),
                        batch_position=_checkpoint_integer(
                            metadata["batch_position"], "batch_position"
                        ),
                        individual=item.individual,
                    )
                except KeyError as exc:
                    raise PopulationAnalysisError(
                        "Restored evaluation has incomplete proposal metadata."
                    ) from exc
                record = PopulationRecord.from_proposal(proposal).with_evaluation(item)
                if record.proposal_id in records:
                    raise PopulationAnalysisError(
                        "Restored population proposal IDs must be unique."
                    )
                records[record.proposal_id] = record
                batch_records.append(record)
            batch_records.sort(key=lambda record: record.batch_position)
            snapshots.append(PopulationSnapshot(batch.batch_index, tuple(batch_records)))

        best_ids = tuple(best_individual_ids)
        if len(set(best_ids)) != len(best_ids):
            raise PopulationAnalysisError(
                "Best individual IDs must be unique."
            )
        known_ids = {record.individual_id for record in records.values()}
        if any(identifier not in known_ids for identifier in best_ids):
            raise PopulationAnalysisError(
                "Best individual IDs must belong to restored population history."
            )
        self._records = records
        self._batch_proposal_ids.clear()
        self._started_individual_ids.clear()
        self._snapshots = snapshots
        self._best_individual_ids = best_ids
        self._plot_results.clear()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        every_batches = self.plots.every_batches
        if every_batches is not None:
            for count, snapshot in enumerate(self._snapshots, start=1):
                if count % every_batches == 0:
                    scope = f"batch_{snapshot.batch_index:06d}"
                    self._plot_results[scope] = self._render_plots(
                        scope,
                        self._snapshots[:count],
                    )
        if completed and self.plots.final_plots:
            self._plot_results["final"] = self._render_plots(
                "final",
                self._snapshots,
                best_individual_ids=self._best_individual_ids,
            )
        self._write_json(self.summary_path, self._analysis_summary())
        self._write_json(self.manifest_path, self._analysis_manifest())

    def _render_plots(
        self,
        scope: str,
        snapshots: Sequence[PopulationSnapshot],
        *,
        best_individual_ids: Sequence[str] = (),
    ) -> PopulationPlotResult:
        target_dir = self.output_dir / "plots" / scope
        if target_dir.exists():
            shutil.rmtree(target_dir)
        return self._renderer.render(
            snapshots,
            best_individual_ids=best_individual_ids,
            target_dir=target_dir,
        )

    def _analysis_summary(self) -> dict[str, Any]:
        objective_summaries = objective_summary_by_batch(self._snapshots)
        score_summaries = score_summary_by_batch(self._snapshots)
        return {
            "batches": len(self._snapshots),
            "records": len(self.records),
            "best_individual_ids": list(self._best_individual_ids),
            "diversity_by_batch": {
                str(snapshot.batch_index): {
                    "unique_genotype_ratio": unique_genotype_ratio(snapshot),
                    "duplicate_genotype_ratio": duplicate_genotype_ratio(snapshot),
                    "mean_pairwise_hamming_distance": (
                        mean_pairwise_hamming_distance(snapshot)
                    ),
                    "gene_entropy": list(gene_entropy(snapshot)),
                }
                for snapshot in self._snapshots
            },
            "failure_rate_by_batch": {
                str(batch): value
                for batch, value in failure_rate_by_batch(self._snapshots).items()
            },
            "objectives_by_batch": {
                str(batch): {
                    name: asdict(summary) for name, summary in summaries.items()
                }
                for batch, summaries in objective_summaries.items()
            },
            "scores_by_batch": {
                str(batch): asdict(summary) if summary is not None else None
                for batch, summary in score_summaries.items()
            },
            "final_stage_frequency": (
                stage_frequency_by_slot(self._snapshots[-1])
                if self._snapshots
                else {}
            ),
        }

    def _analysis_manifest(self) -> dict[str, Any]:
        generated: list[str] = []
        skipped: dict[str, dict[str, str]] = {}
        warnings: list[str] = []
        for scope, result in self._plot_results.items():
            generated.extend(
                str(Path("plots") / scope / filename)
                for filename in result.generated
            )
            if result.skipped:
                skipped[scope] = dict(result.skipped)
            warnings.extend(f"{scope}/{warning}" for warning in result.warnings)
        return {
            "generated": generated,
            "skipped": skipped,
            "warnings": warnings,
        }

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _checkpoint_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PopulationAnalysisError(
            f"Restored {name} must be a non-negative integer."
        )
    return value


def _checkpoint_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PopulationAnalysisError(
            f"Restored {name} must be a non-empty string."
        )
    return value


__all__ = ["PopulationAnalysisCollector", "PopulationAnalysisError"]
