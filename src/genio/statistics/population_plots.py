"""Headless Matplotlib renderer for population-analysis snapshots."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any

from genio.core.result import ResultStatus
from genio.statistics.population_analysis import (
    best_individual_gene_matrix,
    best_score_evolution,
    failure_rate_by_batch,
    gene_entropy,
    gene_value_frequency,
    mean_pairwise_hamming_distance,
    objective_summary_by_batch,
    project_population_pca,
    score_summary_by_batch,
    stage_frequency_by_slot,
    unique_genotype_ratio,
)
from genio.statistics.population_models import PopulationRecord, PopulationSnapshot


class PopulationPlotError(RuntimeError):
    """Raised when population plots cannot be rendered under strict policy."""


class PopulationPlotDependencyError(PopulationPlotError):
    """Raised when the optional Matplotlib dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class PopulationPlotConfig:
    """Configure population plot cadence, selection, format, and failures."""

    every_batches: int | None = None
    tracked_genes: tuple[int, ...] = ()
    image_format: str = "png"
    final_plots: bool = True
    strict: bool = False

    def __post_init__(self) -> None:
        if self.every_batches is not None and (
            isinstance(self.every_batches, bool)
            or not isinstance(self.every_batches, Integral)
            or self.every_batches <= 0
        ):
            raise ValueError("every_batches must be a positive integer or None.")
        genes = tuple(self.tracked_genes)
        if any(
            isinstance(gene, bool) or not isinstance(gene, Integral) or gene < 0
            for gene in genes
        ):
            raise ValueError("tracked_genes must contain non-negative integers.")
        if len(set(genes)) != len(genes):
            raise ValueError("tracked_genes must not contain duplicates.")
        if self.image_format not in {"png", "svg"}:
            raise ValueError("image_format must be 'png' or 'svg'.")
        if not isinstance(self.final_plots, bool) or not isinstance(self.strict, bool):
            raise TypeError("final_plots and strict must be booleans.")
        object.__setattr__(self, "every_batches", (
            int(self.every_batches) if self.every_batches is not None else None
        ))
        object.__setattr__(self, "tracked_genes", tuple(int(gene) for gene in genes))


@dataclass(frozen=True, slots=True)
class PopulationPlotResult:
    """Describe generated files, inapplicable plots, and recoverable failures."""

    generated: tuple[str, ...] = ()
    skipped: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "generated", tuple(self.generated))
        object.__setattr__(self, "skipped", MappingProxyType(dict(self.skipped)))
        object.__setattr__(self, "warnings", tuple(self.warnings))


class PopulationPlotRenderer:
    """Render deterministic population plots from immutable captured data."""

    def __init__(self, config: PopulationPlotConfig | None = None) -> None:
        self.config = config or PopulationPlotConfig()

    def render(
        self,
        snapshots: Sequence[PopulationSnapshot],
        *,
        best_individual_ids: Sequence[str] = (),
        target_dir: str | Path,
    ) -> PopulationPlotResult:
        """Render every applicable MVP plot into ``target_dir``."""

        normalized = tuple(snapshots)
        if any(not isinstance(snapshot, PopulationSnapshot) for snapshot in normalized):
            raise TypeError("snapshots must contain PopulationSnapshot instances.")
        if any(
            current.batch_index >= following.batch_index
            for current, following in zip(normalized, normalized[1:])
        ):
            raise ValueError("snapshots must be ordered by increasing batch_index.")
        output_dir = Path(target_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        records = tuple(record for snapshot in normalized for record in snapshot.records)
        plotters: tuple[tuple[str, Callable[[], Any | str]], ...] = (
            ("population_size", lambda: self._plot_population_size(normalized)),
            ("stage_distribution", lambda: self._plot_stage_distribution(normalized)),
            ("gene_distribution", lambda: self._plot_gene_distribution(normalized)),
            ("gene_entropy", lambda: self._plot_gene_entropy(normalized)),
            ("unique_genotypes", lambda: self._plot_unique_genotypes(normalized)),
            ("genotype_distance", lambda: self._plot_genotype_distance(normalized)),
            (
                "population_projection",
                lambda: self._plot_population_projection(records, best_individual_ids),
            ),
            ("objective_evolution", lambda: self._plot_objective_evolution(normalized)),
            (
                "aggregate_score_evolution",
                lambda: self._plot_score_evolution(normalized),
            ),
            (
                "best_individuals_genes",
                lambda: self._plot_best_individuals(records, best_individual_ids),
            ),
            ("failure_rate", lambda: self._plot_failure_rate(normalized)),
            (
                "objective_scatter",
                lambda: self._plot_objective_scatter(records, best_individual_ids),
            ),
        )
        generated: list[str] = []
        skipped: dict[str, str] = {}
        warnings: list[str] = []
        for name, plotter in plotters:
            figure: Any | None = None
            try:
                outcome = plotter()
                if isinstance(outcome, str):
                    skipped[name] = outcome
                    continue
                figure = outcome
                filename = f"{name}.{self.config.image_format}"
                figure.savefig(
                    output_dir / filename,
                    format=self.config.image_format,
                    bbox_inches="tight",
                    dpi=140,
                )
                generated.append(filename)
            except Exception as exc:
                message = f"{name}: {type(exc).__name__}: {exc}"
                if self.config.strict:
                    raise PopulationPlotError(message) from exc
                warnings.append(message)
            finally:
                if figure is not None:
                    figure.clear()
        return PopulationPlotResult(tuple(generated), skipped, tuple(warnings))

    @staticmethod
    def _figure(title: str, *, width: float = 8.0, height: float = 4.5):
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError as exc:
            raise PopulationPlotDependencyError(
                "Population plots require the optional 'matplotlib' dependency."
            ) from exc
        figure = Figure(figsize=(width, height), constrained_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.set_title(title)
        return figure, axis

    def _plot_population_size(self, snapshots: Sequence[PopulationSnapshot]):
        if not snapshots:
            return "No population snapshots are available."
        figure, axis = self._figure("Population size by batch")
        batches = [snapshot.batch_index for snapshot in snapshots]
        successful = [
            sum(record.evaluation_status == ResultStatus.SUCCESS.value for record in snapshot)
            for snapshot in snapshots
        ]
        failed = [len(snapshot) - count for snapshot, count in zip(snapshots, successful)]
        axis.bar(batches, successful, label="successful")
        axis.bar(batches, failed, bottom=successful, label="failed")
        axis.set(xlabel="Batch", ylabel="Individuals")
        axis.legend()
        return figure

    def _plot_stage_distribution(self, snapshots: Sequence[PopulationSnapshot]):
        if not snapshots or not snapshots[-1].records:
            return "No final population is available."
        frequencies = stage_frequency_by_slot(snapshots[-1])
        labels = [
            f"slot {slot}: {stage}"
            for slot, stages in frequencies.items()
            for stage in stages
        ]
        values = [count for stages in frequencies.values() for count in stages.values()]
        if not labels:
            return "Population records contain no stages."
        figure, axis = self._figure("Stage distribution in final population", width=10)
        positions = list(range(len(labels)))
        axis.bar(positions, values)
        axis.set_xticks(positions, labels, rotation=45, ha="right")
        axis.set_ylabel("Individuals")
        return figure

    def _plot_gene_distribution(self, snapshots: Sequence[PopulationSnapshot]):
        if not snapshots:
            return "No population snapshots are available."
        frequencies = gene_value_frequency(snapshots[-1])
        labels = [
            f"g{gene}={value}"
            for gene, values in frequencies.items()
            for value in values
        ]
        counts = [count for values in frequencies.values() for count in values.values()]
        if not labels:
            return "Final population contains no genotypes."
        figure, axis = self._figure("Gene values in final population", width=10)
        positions = list(range(len(labels)))
        axis.bar(positions, counts)
        axis.set_xticks(positions, labels, rotation=45, ha="right")
        axis.set_ylabel("Individuals")
        return figure

    def _plot_gene_entropy(self, snapshots: Sequence[PopulationSnapshot]):
        if not snapshots:
            return "No population snapshots are available."
        entropies = tuple(gene_entropy(snapshot) for snapshot in snapshots)
        if not entropies or not entropies[0]:
            return "Population snapshots contain no genotypes."
        width = len(entropies[0])
        if any(len(values) != width for values in entropies):
            raise ValueError("Gene count changes between population snapshots.")
        genes = self.config.tracked_genes or tuple(range(width))
        if any(gene >= width for gene in genes):
            raise ValueError("tracked_genes contains an unavailable gene index.")
        figure, axis = self._figure("Gene entropy evolution")
        batches = [snapshot.batch_index for snapshot in snapshots]
        for gene in genes:
            axis.plot(batches, [values[gene] for values in entropies], marker="o", label=f"g{gene}")
        axis.set(xlabel="Batch", ylabel="Entropy (bits)")
        axis.legend()
        return figure

    def _plot_unique_genotypes(self, snapshots: Sequence[PopulationSnapshot]):
        values = [unique_genotype_ratio(snapshot) for snapshot in snapshots]
        if not snapshots or any(value is None for value in values):
            return "Population snapshots contain no complete genotypes."
        figure, axis = self._figure("Unique genotype ratio")
        axis.plot([snapshot.batch_index for snapshot in snapshots], values, marker="o")
        axis.set(xlabel="Batch", ylabel="Unique ratio", ylim=(0.0, 1.05))
        return figure

    def _plot_genotype_distance(self, snapshots: Sequence[PopulationSnapshot]):
        values = [mean_pairwise_hamming_distance(snapshot) for snapshot in snapshots]
        if not snapshots or all(value is None for value in values):
            return "At least two genotypes per population are required."
        figure, axis = self._figure("Mean pairwise genotype distance")
        points = [
            (snapshot.batch_index, value)
            for snapshot, value in zip(snapshots, values)
            if value is not None
        ]
        axis.plot([point[0] for point in points], [point[1] for point in points], marker="o")
        axis.set(xlabel="Batch", ylabel="Normalized Hamming distance", ylim=(0.0, 1.05))
        return figure

    def _plot_population_projection(
        self,
        records: Sequence[PopulationRecord],
        best_individual_ids: Sequence[str],
    ):
        if len(records) < 2:
            return "At least two population records are required."
        coordinates = project_population_pca(records)
        figure, axis = self._figure("Population projection (one-hot PCA)")
        batches = [record.batch_index for record in records]
        scatter = axis.scatter(
            [point[0] for point in coordinates],
            [point[1] for point in coordinates],
            c=batches,
            cmap="viridis",
            alpha=0.75,
        )
        best_ids = set(best_individual_ids)
        best_points = [
            point
            for point, record in zip(coordinates, records)
            if record.individual_id in best_ids
        ]
        if best_points:
            axis.scatter(
                [point[0] for point in best_points],
                [point[1] for point in best_points],
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
                label="algorithm best",
            )
            axis.legend()
        axis.set(xlabel="PC1", ylabel="PC2")
        figure.colorbar(scatter, ax=axis, label="Batch")
        return figure

    def _plot_objective_evolution(self, snapshots: Sequence[PopulationSnapshot]):
        summaries = objective_summary_by_batch(snapshots)
        names = tuple(sorted({name for values in summaries.values() for name in values}))
        if not names:
            return "No valid objective values are available."
        figure, axis = self._figure("Raw objective evolution")
        for name in names:
            points = [
                (batch, values[name])
                for batch, values in summaries.items()
                if name in values
            ]
            x = [point[0] for point in points]
            means = [point[1].mean for point in points]
            lower = [point[1].first_quartile for point in points]
            upper = [point[1].third_quartile for point in points]
            axis.plot(x, means, marker="o", label=name)
            axis.fill_between(x, lower, upper, alpha=0.15)
        axis.set(xlabel="Batch", ylabel="Raw objective value")
        axis.legend()
        return figure

    def _plot_score_evolution(self, snapshots: Sequence[PopulationSnapshot]):
        summaries = score_summary_by_batch(snapshots)
        available = [(batch, value) for batch, value in summaries.items() if value is not None]
        if not available:
            return "No aggregate scores are available."
        best = best_score_evolution(snapshots)
        figure, axis = self._figure("Aggregate score evolution")
        x = [batch for batch, _ in available]
        axis.plot(x, [summary.mean for _, summary in available], marker="o", label="mean")
        axis.plot(x, [best[batch] for batch in x], marker="o", label="best")
        axis.set(xlabel="Batch", ylabel="Aggregate score")
        axis.legend()
        return figure

    def _plot_best_individuals(
        self,
        records: Sequence[PopulationRecord],
        best_individual_ids: Sequence[str],
    ):
        matrix = best_individual_gene_matrix(records, best_individual_ids)
        if not matrix:
            return "The algorithm selected no individuals with genotypes."
        import numpy as np

        figure, axis = self._figure("Algorithm-selected genotypes")
        image = axis.imshow(np.asarray(matrix, dtype=float), aspect="auto", cmap="viridis")
        axis.set(xlabel="Gene", ylabel="Selected occurrence")
        figure.colorbar(image, ax=axis, label="Gene value")
        return figure

    def _plot_failure_rate(self, snapshots: Sequence[PopulationSnapshot]):
        values = failure_rate_by_batch(snapshots)
        if not values:
            return "No population snapshots are available."
        figure, axis = self._figure("Evaluation failure rate")
        axis.plot(list(values), list(values.values()), marker="o")
        axis.set(xlabel="Batch", ylabel="Failure rate", ylim=(0.0, 1.05))
        return figure

    def _plot_objective_scatter(
        self,
        records: Sequence[PopulationRecord],
        best_individual_ids: Sequence[str],
    ):
        valid = tuple(record for record in records if record.objectives_valid)
        names = tuple(sorted({name for record in valid for name in record.objective_values}))
        if len(names) != 2:
            return "Exactly two raw objectives are required."
        complete = tuple(
            record for record in valid if all(name in record.objective_values for name in names)
        )
        if not complete:
            return "No complete two-objective records are available."
        figure, axis = self._figure("Raw objective dispersion")
        axis.scatter(
            [record.objective_values[names[0]] for record in complete],
            [record.objective_values[names[1]] for record in complete],
            c=[record.batch_index for record in complete],
            cmap="viridis",
            alpha=0.75,
        )
        best_ids = set(best_individual_ids)
        selected = [record for record in complete if record.individual_id in best_ids]
        if selected:
            axis.scatter(
                [record.objective_values[names[0]] for record in selected],
                [record.objective_values[names[1]] for record in selected],
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
                label="algorithm best",
            )
            axis.legend()
        axis.set(xlabel=names[0], ylabel=names[1])
        return figure


__all__ = [
    "PopulationPlotConfig",
    "PopulationPlotDependencyError",
    "PopulationPlotError",
    "PopulationPlotRenderer",
    "PopulationPlotResult",
]
