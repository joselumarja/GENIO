"""Headless Matplotlib renderer for population-analysis snapshots."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
import re
from statistics import fmean
from types import MappingProxyType
from typing import Any

from genio.core.result import ResultStatus
from genio.objective.base import ObjectiveSchema, OptimizationDirection
from genio.search_space.space import SearchSpace
from genio.statistics.population_analysis import (
    best_score_evolution,
    failure_rate_by_batch,
    gene_entropy,
    mean_pairwise_hamming_distance,
    objective_summary_by_batch,
    project_population_pca,
    score_summary_by_batch,
    unique_genotype_ratio,
)
from genio.statistics.population_models import PopulationRecord, PopulationSnapshot

_MISSING = object()


class PopulationPlotError(RuntimeError):
    """Raised when population plots cannot be rendered under strict policy."""


class PopulationPlotDependencyError(PopulationPlotError):
    """Raised when the optional Matplotlib dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class PopulationPlotConfig:
    """Configure cadence, decision-gene selection, format, and failure policy.

    An empty ``tracked_genes`` selects all decisions. A pipeline gene includes
    its stage choice and all conditional parameters; a grouped design gene
    includes every parameter in that group. Low-cardinality decisions use
    composition counts; numeric decisions with over 20 values use box plots.
    """

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


@dataclass(frozen=True, slots=True)
class _DecisionParameter:
    label: str
    gene: int
    alternatives: tuple[Any, ...]
    domain: str = ""
    name: str = ""
    slot: int | None = None
    stage: str | None = None

    def value(self, record: PopulationRecord) -> Any:
        if self.slot is None:
            return record.design.get(self.domain, {}).get(self.name, _MISSING)
        if self.slot >= len(record.stages):
            return _MISSING
        if self.stage is None:
            return record.stages[self.slot]
        if record.stages[self.slot] != self.stage:
            return _MISSING
        return record.stage_parameters[self.slot].get(self.name, _MISSING)


class PopulationPlotRenderer:
    """Render deterministic population plots from immutable captured data."""

    def __init__(self, config: PopulationPlotConfig | None = None) -> None:
        self.config = config or PopulationPlotConfig()
        self._parameters: tuple[_DecisionParameter, ...] = ()
        self._objective_schema: ObjectiveSchema | None = None

    def configure(self, search_space: SearchSpace, objectives: ObjectiveSchema | None) -> None:
        """Resolve real decision values, including conditional stage parameters."""

        self._objective_schema = objectives
        parameters: list[_DecisionParameter] = []
        scenario = search_space.scenario
        domains = scenario.domain_order or (
            (("pipeline",) if scenario.slots else ()) + tuple(scenario.design_spaces)
        )
        gene = 0
        for domain in domains:
            if domain == "pipeline":
                for slot in scenario.slots:
                    parameters.append(_DecisionParameter(
                        f"pipeline.{slot.index}.stage", gene,
                        tuple(dict.fromkeys(choice.stage for choice in slot.alternatives)),
                        slot=slot.index,
                    ))
                    grouped: dict[tuple[str, str], list[Any]] = {}
                    for choice in slot.alternatives:
                        for name, value in choice.parameters.items():
                            grouped.setdefault((choice.stage, name), []).append(value)
                    for (stage, name), values in grouped.items():
                        parameters.append(_DecisionParameter(
                            f"pipeline.{slot.index}.{stage}.{name}", gene, tuple(values),
                            slot=slot.index, stage=stage, name=name,
                        ))
                    gene += 1
            else:
                for name, dimension in scenario.design_spaces[domain].items():
                    for parameter in dimension.parameter_names or (name,):
                        values = tuple(
                            value[parameter] if dimension.parameter_names else value
                            for value in dimension.alternatives
                        )
                        parameters.append(_DecisionParameter(
                            f"{domain}.{parameter}", gene, values,
                            domain=domain, name=parameter,
                        ))
                    gene += 1
        if any(gene >= len(search_space.genotype_lengths) for gene in self.config.tracked_genes):
            raise ValueError("tracked_genes contains an unavailable gene index.")
        self._parameters = tuple(
            parameter for parameter in parameters
            if not self.config.tracked_genes or parameter.gene in self.config.tracked_genes
        )

    def render(
        self,
        snapshots: Sequence[PopulationSnapshot],
        *,
        best_individual_ids: Sequence[str] = (),
        target_dir: str | Path,
    ) -> PopulationPlotResult:
        """Render population diagnostics and one evolution plot per decision parameter."""

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
            ("failure_rate", lambda: self._plot_failure_rate(normalized)),
            (
                "objective_scatter",
                lambda: self._plot_objective_scatter(records, best_individual_ids),
            ),
        )
        # Ordinals keep filenames safe and unique even for arbitrary parameter names.
        plotters += tuple(
            (f"parameter_{index:03d}_{re.sub(r'[^a-zA-Z0-9_-]+', '_', parameter.label)[:100]}_evolution",
             lambda parameter=parameter: self._plot_parameter_evolution(normalized, parameter))
            for index, parameter in enumerate(self._parameters)
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

    def _plot_parameter_evolution(
        self, snapshots: Sequence[PopulationSnapshot], parameter: _DecisionParameter,
    ):
        values = [
            [value for record in snapshot if (value := parameter.value(record)) is not _MISSING]
            for snapshot in snapshots
        ]
        if not any(values):
            return f"No active values for {parameter.label}."
        batches = [snapshot.batch_index for snapshot in snapshots]
        figure, axis = self._figure(parameter.label, width=10)
        numeric = all(
            isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
            for value in parameter.alternatives
        )
        categories = tuple(dict.fromkeys(_category_label(value) for value in parameter.alternatives))
        if numeric and len(categories) > 20:
            available = [(batch, items) for batch, items in zip(batches, values) if items]
            axis.boxplot(
                [items for _, items in available],
                positions=[batch for batch, _ in available], manage_ticks=False,
            )
            axis.plot(
                [batch for batch, _ in available],
                [fmean(items) for _, items in available],
                marker="o", label="mean",
            )
            axis.set_ylabel(parameter.label)
            axis.legend()
        else:
            counts = [Counter(_category_label(value) for value in items) for items in values]
            series = [[count[category] for count in counts] for category in categories]
            if len(batches) == 1:
                bottom = 0
                for category, items in zip(categories, series):
                    axis.bar(batches, items, bottom=bottom, label=category)
                    bottom += items[0]
            else:
                axis.stackplot(batches, *series, labels=categories)
            axis.set_ylabel("Individuals (proposals)")
            axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        axis.set_xlabel("Batch")
        axis.set_xticks(batches)
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
            label = next((parameter.label for parameter in self._parameters
                          if parameter.gene == gene), f"g{gene}")
            axis.plot(batches, [values[gene] for values in entropies], marker="o", label=label)
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
        names = self._objective_names(snapshots)
        if not names or not any(summaries.values()):
            return "No valid objective values are available."
        figure, initial_axis = self._figure("Raw objective evolution", height=3.5 * len(names))
        figure.delaxes(initial_axis)
        axes = figure.subplots(len(names), 1, squeeze=False)[:, 0]
        directions = (
            dict(zip(self._objective_schema.names, self._objective_schema.directions))
            if self._objective_schema is not None else {}
        )
        for name, axis in zip(names, axes):
            points = [
                (batch, values[name])
                for batch, values in summaries.items()
                if name in values
            ]
            x = [point[0] for point in points]
            medians = [point[1].median for point in points]
            lower = [point[1].first_quartile for point in points]
            upper = [point[1].third_quartile for point in points]
            axis.plot(x, medians, marker="o", label="median")
            axis.fill_between(x, lower, upper, alpha=0.15, label="interquartile range")
            if name in directions:
                direction = directions[name]
                minimize = direction == OptimizationDirection.MINIMIZE
                best = [point[1].minimum if minimize else point[1].maximum for point in points]
                axis.plot(x, best, marker=".", label=f"best ({direction.value})")
            axis.set(xlabel="Batch", ylabel=name, title=name)
            axis.legend()
        return figure

    def _objective_names(self, snapshots: Sequence[PopulationSnapshot]) -> tuple[str, ...]:
        if self._objective_schema is not None:
            return self._objective_schema.names
        return tuple(dict.fromkeys(
            name for snapshot in snapshots for record in snapshot.valid_objective_records
            for name in record.objective_values
        ))

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
        names = self._objective_schema.names if self._objective_schema is not None else tuple(
            dict.fromkeys(name for record in valid for name in record.objective_values)
        )
        if len(names) < 2 or not valid:
            return "At least two objectives with valid values are required."
        pairs = tuple(combinations(names, 2))
        columns = min(3, len(pairs))
        rows = (len(pairs) + columns - 1) // columns
        figure, initial_axis = self._figure(
            "Raw objective dispersion", width=6 * columns, height=4 * rows,
        )
        figure.delaxes(initial_axis)
        axes = figure.subplots(rows, columns, squeeze=False).ravel()
        for axis in axes[len(pairs):]:
            figure.delaxes(axis)
        best_ids = set(best_individual_ids)
        for (left, right), axis in zip(pairs, axes):
            complete = [record for record in valid if left in record.objective_values
                        and right in record.objective_values]
            scatter = axis.scatter(
                [record.objective_values[left] for record in complete],
                [record.objective_values[right] for record in complete],
                c=[record.batch_index for record in complete], cmap="viridis", alpha=0.75,
            )
            selected = [record for record in complete if record.individual_id in best_ids]
            if selected:
                axis.scatter(
                    [record.objective_values[left] for record in selected],
                    [record.objective_values[right] for record in selected],
                    facecolors="none", edgecolors="red", linewidths=1.5,
                    label="algorithm best",
                )
                axis.legend()
            axis.set(xlabel=left, ylabel=right)
            figure.colorbar(scatter, ax=axis, label="Batch")
        return figure


def _category_label(value: Any) -> str:
    if isinstance(value, Mapping):
        return "{" + ", ".join(
            f"{key!r}: {_category_label(value[key])}" for key in sorted(value)
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_category_label(item) for item in value) + "]"
    return repr(value)


__all__ = [
    "PopulationPlotConfig",
    "PopulationPlotDependencyError",
    "PopulationPlotError",
    "PopulationPlotRenderer",
    "PopulationPlotResult",
]
