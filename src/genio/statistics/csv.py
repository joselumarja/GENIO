from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from genio.core.evaluation import Evaluation
from genio.core.proposal import Proposal
from genio.core.search_result import SearchResult
from genio.statistics.base import StatisticsCollector

if TYPE_CHECKING:
    from genio.session.optimization import OptimizationSession


class CSVStatisticsCollector(StatisticsCollector):
    """Persist one CSV row per proposed individual and run-level summaries."""

    SCHEMA_VERSION = 2
    supports_checkpointing = True
    _BASE_COLUMNS = (
        "schema_version",
        "run_id",
        "session_id",
        "proposal_id",
        "proposal_sequence",
        "individual_id",
        "scenario_id",
        "batch_index",
        "batch_position",
        "generation",
        "population_index",
        "proposal_origin",
        "search_index",
        "genotype_json",
        "genotype_hash",
        "individual_fingerprint",
        "duplicate_genotype_seen",
        "pipeline_json",
        "design_json",
        "individual_metadata_json",
        "algorithm_metadata_json",
        "algorithm_name",
        "evaluation_status",
        "partial_metrics",
        "cache_hit",
        "cache_json",
        "metrics_json",
        "error_type",
        "error_message",
    )

    def __init__(
        self,
        output_dir: str | Path,
        *,
        individuals_filename: str = "individuals.csv",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.individuals_path = self.output_dir / individuals_filename
        self.run_manifest_path = self.output_dir / "run_manifest.json"
        self.run_summary_path = self.output_dir / "run_summary.json"
        self._rows: dict[str, dict[str, Any]] = {}
        self._proposal_order: list[str] = []
        self._seen_genotypes: set[str] = set()
        self._run_id: str | None = None
        self._session_id: str | None = None
        self._algorithm_name: str | None = None
        self._started_at: datetime | None = None
        self._started_monotonic: float | None = None
        self._completed_batches = 0
        self._completed_evaluations = 0
        self._artifact_cache = None

    def on_session_started(self, session: OptimizationSession) -> None:
        """Initialize output files and write the reproducibility manifest."""

        self._reset_runtime_state()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = session.run_id
        self._session_id = session.id
        self._algorithm_name = self._qualified_name(session.algorithm)
        self._artifact_cache = session.artifact_cache
        self._started_at = datetime.now(timezone.utc)
        self._started_monotonic = time.perf_counter()
        self._write_manifest(session)

    def _write_manifest(self, session: OptimizationSession) -> None:
        self._write_json(
            self.run_manifest_path,
            {
                "schema_version": self.SCHEMA_VERSION,
                "run_id": session.run_id,
                "session_id": session.id,
                "started_at": self._isoformat(self._started_at),
                "scenario_id": session.search_space.scenario_id,
                "search_space_size": session.search_space.search_space_size,
                "algorithm": self._algorithm_name,
                "backend": self._qualified_name(session.backend),
                "backend_run_id": getattr(session.backend, "run_id", None),
                "artifact_cache": (
                    {
                        "type": self._qualified_name(session.artifact_cache),
                        "initial_state": session.artifact_cache.snapshot(),
                    }
                    if session.artifact_cache is not None
                    else None
                ),
                "workflow_steps": [
                    {
                        "id": step.id,
                        "type": self._qualified_name(step),
                        "depends_on": list(step.depends_on),
                    }
                    for step in session.evaluation_workflow.execution_order()
                ],
                "session_metadata": dict(session.metadata),
            },
        )

    def _reset_runtime_state(self) -> None:
        self._rows = {}
        self._proposal_order = []
        self._seen_genotypes = set()
        self._completed_batches = 0
        self._completed_evaluations = 0

    def on_proposals_generated(self, proposals: Sequence[Proposal]) -> None:
        """Register generated proposals before their evaluations begin."""

        for proposal in proposals:
            if proposal.proposal_id in self._rows:
                raise ValueError(f"Duplicate proposal id {proposal.proposal_id!r}.")
            row = self._proposal_row(proposal)
            self._rows[proposal.proposal_id] = row
            self._proposal_order.append(proposal.proposal_id)
        self._write_csv()

    def on_evaluation_completed(self, evaluation: Evaluation) -> None:
        """Merge a completed evaluation into its proposal row."""

        proposal_id = str(evaluation.metadata["proposal_id"])
        try:
            row = self._rows[proposal_id]
        except KeyError as exc:
            raise ValueError(
                f"Evaluation references unknown proposal id {proposal_id!r}."
            ) from exc

        result = evaluation.result
        row["evaluation_status"] = result.status.value
        row["partial_metrics"] = bool(result.metrics and result.error)
        cache_metadata = result.metadata.get("cache", {})
        if not isinstance(cache_metadata, Mapping):
            cache_metadata = {}
        row["cache_hit"] = any(
            bool(values.get("cache_hit"))
            for values in cache_metadata.values()
            if isinstance(values, Mapping)
        )
        row["cache_json"] = self._canonical_json(cache_metadata)
        self._flatten_mapping("cache", cache_metadata, row)
        row["metrics_json"] = self._canonical_json(result.metrics)
        for metric_name, value in sorted(result.metrics.items()):
            row[f"metric.{metric_name}"] = value

        if result.error:
            error_type, separator, error_message = result.error.partition(": ")
            row["error_type"] = error_type if separator else ""
            row["error_message"] = error_message if separator else result.error
        self._completed_evaluations += 1

    def on_batch_completed(
        self,
        batch_index: int,
        evaluations: Sequence[Evaluation],
    ) -> None:
        """Flush all rows atomically after a completed batch."""

        self._completed_batches += 1
        self._write_csv()

    def on_session_completed(self, result: SearchResult) -> None:
        """Finalize the CSV and write aggregate run statistics."""

        self._write_csv()
        finished_at = datetime.now(timezone.utc)
        wall_seconds = (
            time.perf_counter() - self._started_monotonic
            if self._started_monotonic is not None
            else None
        )
        status_counts = Counter(
            str(row["evaluation_status"]) for row in self._ordered_rows()
        )
        self._write_json(
            self.run_summary_path,
            {
                "schema_version": self.SCHEMA_VERSION,
                "run_id": self._run_id,
                "session_id": self._session_id,
                "started_at": self._isoformat(self._started_at),
                "finished_at": self._isoformat(finished_at),
                "wall_seconds": wall_seconds,
                "generated_individuals": len(self._rows),
                "evaluated_individuals": self._completed_evaluations,
                "batches": self._completed_batches,
                "status_counts": dict(sorted(status_counts.items())),
                "duplicate_genotypes": sum(
                    bool(row["duplicate_genotype_seen"])
                    for row in self._ordered_rows()
                ),
                "best_individual_ids": [
                    individual.id for individual in result.best_individuals
                ],
                "metric_summary": self._metric_summary(),
                "cache": (
                    self._artifact_cache.snapshot()
                    if self._artifact_cache is not None
                    else None
                ),
            },
        )

    def snapshot(self) -> dict[str, Any]:
        """Return paths and counts for the generated statistics files."""

        return {
            "run_id": self._run_id,
            "generated_individuals": len(self._rows),
            "evaluated_individuals": self._completed_evaluations,
            "batches": self._completed_batches,
            "individuals_csv": str(self.individuals_path),
            "run_manifest": str(self.run_manifest_path),
            "run_summary": str(self.run_summary_path),
            "cache": (
                self._artifact_cache.snapshot()
                if self._artifact_cache is not None
                else None
            ),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Return rows, counters and timing needed to resume CSV reporting."""

        elapsed_seconds = (
            time.perf_counter() - self._started_monotonic
            if self._started_monotonic is not None
            else 0.0
        )
        return {
            "rows": deepcopy(self._rows),
            "proposal_order": list(self._proposal_order),
            "seen_genotypes": sorted(self._seen_genotypes),
            "run_id": self._run_id,
            "session_id": self._session_id,
            "algorithm_name": self._algorithm_name,
            "started_at": self._isoformat(self._started_at),
            "elapsed_seconds": elapsed_seconds,
            "completed_batches": self._completed_batches,
            "completed_evaluations": self._completed_evaluations,
            "run_manifest": self._read_json_if_exists(self.run_manifest_path),
            "run_summary": self._read_json_if_exists(self.run_summary_path),
        }

    def checkpoint_signature(self) -> dict[str, Any]:
        """Return CSV output location and schema configuration."""

        return {
            **StatisticsCollector.checkpoint_signature(self),
            "schema_version": self.SCHEMA_VERSION,
            "output_dir": str(self.output_dir.expanduser().resolve()),
            "individuals_filename": self.individuals_path.name,
        }

    def restore_checkpoint_state(
        self,
        state: dict[str, Any],
        *,
        session: OptimizationSession,
        evaluations: Sequence[Evaluation],
        completed: bool,
    ) -> None:
        """Restore authoritative CSV rows and discard uncommitted on-disk rows."""

        try:
            rows = {
                str(proposal_id): dict(row)
                for proposal_id, row in dict(state["rows"]).items()
            }
            proposal_order = [str(value) for value in state["proposal_order"]]
            seen_genotypes = {str(value) for value in state["seen_genotypes"]}
            started_at_value = state.get("started_at")
            started_at = (
                datetime.fromisoformat(str(started_at_value))
                if started_at_value is not None
                else datetime.now(timezone.utc)
            )
            elapsed_seconds = float(state.get("elapsed_seconds", 0.0))
            completed_batches = int(state["completed_batches"])
            completed_evaluations = int(state["completed_evaluations"])
            run_manifest = state.get("run_manifest")
            run_summary = state.get("run_summary")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid CSV statistics checkpoint state.") from exc
        if set(proposal_order) != set(rows) or len(proposal_order) != len(rows):
            raise ValueError("CSV checkpoint proposal order is inconsistent.")
        if completed_evaluations != len(evaluations):
            raise ValueError("CSV checkpoint evaluation count is inconsistent.")

        self._rows = rows
        self._proposal_order = proposal_order
        self._seen_genotypes = seen_genotypes
        self._run_id = session.run_id
        self._session_id = session.id
        self._algorithm_name = self._qualified_name(session.algorithm)
        self._started_at = started_at
        self._started_monotonic = time.perf_counter() - max(0.0, elapsed_seconds)
        self._completed_batches = completed_batches
        self._completed_evaluations = completed_evaluations
        self._artifact_cache = session.artifact_cache
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(run_manifest, Mapping):
            self._write_json(self.run_manifest_path, run_manifest)
        else:
            self._write_manifest(session)
        self._write_csv()
        if completed:
            if not isinstance(run_summary, Mapping):
                raise ValueError("Completed CSV checkpoint has no run summary.")
            self._write_json(self.run_summary_path, run_summary)
        else:
            self.run_summary_path.unlink(missing_ok=True)

    def _proposal_row(self, proposal: Proposal) -> dict[str, Any]:
        individual = proposal.individual
        algorithm_metadata = individual.metadata.get("algorithm", {})
        if not isinstance(algorithm_metadata, Mapping):
            algorithm_metadata = {}
        genotype_json = self._canonical_json(individual.genotype)
        pipeline = [
            {
                "slot": choice.slot,
                "stage": choice.stage,
                "parameters": choice.parameters,
                "wrapper_inputs": choice.wrapper_inputs,
            }
            for choice in individual.slots
        ]
        pipeline_json = self._canonical_json(pipeline)
        design_json = self._canonical_json(individual.design)
        fingerprint_payload = self._canonical_json(
            {"pipeline": pipeline, "design": individual.design}
        )
        genotype_key = genotype_json if individual.genotype is not None else fingerprint_payload
        duplicate_genotype = genotype_key in self._seen_genotypes
        self._seen_genotypes.add(genotype_key)

        row: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self._run_id,
            "session_id": self._session_id,
            "proposal_id": proposal.proposal_id,
            "proposal_sequence": proposal.proposal_sequence,
            "individual_id": individual.id,
            "scenario_id": individual.scenario,
            "batch_index": proposal.batch_index,
            "batch_position": proposal.batch_position,
            "generation": algorithm_metadata.get("generation"),
            "population_index": algorithm_metadata.get("population_index"),
            "proposal_origin": algorithm_metadata.get("proposal_origin"),
            "search_index": individual.search_index,
            "genotype_json": genotype_json,
            "genotype_hash": self._sha256(genotype_key),
            "individual_fingerprint": self._sha256(fingerprint_payload),
            "duplicate_genotype_seen": duplicate_genotype,
            "pipeline_json": pipeline_json,
            "design_json": design_json,
            "individual_metadata_json": self._canonical_json(individual.metadata),
            "algorithm_metadata_json": self._canonical_json(algorithm_metadata),
            "algorithm_name": self._algorithm_name,
            "evaluation_status": "not_evaluated",
            "partial_metrics": False,
            "cache_hit": False,
            "cache_json": self._canonical_json({}),
            "metrics_json": self._canonical_json({}),
            "error_type": "",
            "error_message": "",
        }
        for position, choice in enumerate(individual.slots):
            prefix = f"slot.{position:03d}"
            row[f"{prefix}.gene"] = (
                individual.genotype[position]
                if individual.genotype is not None and position < len(individual.genotype)
                else None
            )
            row[f"{prefix}.slot"] = choice.slot
            row[f"{prefix}.stage"] = choice.stage
            row[f"{prefix}.parameters_json"] = self._canonical_json(choice.parameters)
            row[f"{prefix}.wrapper_inputs_json"] = self._canonical_json(
                choice.wrapper_inputs
            )
        self._flatten_mapping("design", individual.design, row)
        self._flatten_mapping("algorithm", algorithm_metadata, row)
        return row

    def _write_csv(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = self._ordered_rows()
        dynamic_columns = sorted(
            {key for row in rows for key in row if key not in self._BASE_COLUMNS}
        )
        fieldnames = [*self._BASE_COLUMNS, *dynamic_columns]
        temporary_path = self.individuals_path.with_suffix(
            f"{self.individuals_path.suffix}.tmp"
        )
        with temporary_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: "" if value is None else value
                        for key, value in row.items()
                    }
                )
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(self.individuals_path)
        self._fsync_directory()

    def _metric_summary(self) -> dict[str, dict[str, float | int]]:
        values_by_metric: dict[str, list[float]] = {}
        for row in self._ordered_rows():
            for key, value in row.items():
                if key.startswith("metric.") and isinstance(value, (int, float)):
                    values_by_metric.setdefault(key.removeprefix("metric."), []).append(
                        float(value)
                    )
        return {
            metric: {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
            for metric, values in sorted(values_by_metric.items())
        }

    def _ordered_rows(self) -> list[dict[str, Any]]:
        return [self._rows[proposal_id] for proposal_id in self._proposal_order]

    @classmethod
    def _flatten_mapping(
        cls,
        prefix: str,
        values: Mapping[str, Any],
        target: dict[str, Any],
    ) -> None:
        for key, value in sorted(values.items()):
            column = f"{prefix}.{key}"
            if isinstance(value, Mapping):
                cls._flatten_mapping(column, value, target)
            elif value is None or isinstance(value, (str, int, float, bool)):
                target[column] = value
            else:
                target[column] = cls._canonical_json(value)

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _qualified_name(value: Any) -> str:
        cls = type(value)
        return f"{cls.__module__}.{cls.__qualname__}"

    @staticmethod
    def _isoformat(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    def _write_json(self, path: Path, value: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    value,
                    indent=2,
                    ensure_ascii=True,
                    sort_keys=True,
                    default=str,
                )
            )
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
        self._fsync_directory()

    @staticmethod
    def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, Mapping) else None

    def _fsync_directory(self) -> None:
        descriptor = os.open(self.output_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = ["CSVStatisticsCollector"]
