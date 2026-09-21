"""Pure numerical analysis for captured population snapshots."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log2
from statistics import fmean, median
from typing import Any

from genio.core.result import ResultStatus
from genio.statistics.population_models import PopulationRecord, PopulationSnapshot


@dataclass(frozen=True, slots=True)
class NumericSummary:
    """Describe the finite distribution of one value in one population."""

    count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    first_quartile: float
    third_quartile: float


def stage_frequency_by_slot(
    snapshot: PopulationSnapshot,
) -> dict[int, dict[str, int]]:
    """Count selected stage names independently for each pipeline slot."""

    frequencies: dict[int, Counter[str]] = {}
    for record in snapshot:
        for slot, stage in enumerate(record.stages):
            frequencies.setdefault(slot, Counter())[stage] += 1
    return {
        slot: dict(sorted(counts.items()))
        for slot, counts in sorted(frequencies.items())
    }


def gene_value_frequency(
    snapshot: PopulationSnapshot,
) -> dict[int, dict[int, int]]:
    """Count mixed-radix gene values by position."""

    genotypes = _validated_genotypes(snapshot.records)
    frequencies: dict[int, Counter[int]] = {
        gene: Counter() for gene in range(len(genotypes[0]))
    } if genotypes else {}
    for genotype in genotypes:
        for gene, value in enumerate(genotype):
            frequencies[gene][value] += 1
    return {
        gene: dict(sorted(counts.items()))
        for gene, counts in frequencies.items()
    }


def parameter_distribution(
    snapshot: PopulationSnapshot,
) -> dict[str, tuple[Any, ...]]:
    """Collect stage-parameter values using stable ``slot.parameter`` keys."""

    values: dict[str, list[Any]] = {}
    for record in snapshot:
        for slot, parameters in enumerate(record.stage_parameters):
            for name, value in sorted(parameters.items()):
                values.setdefault(f"slot.{slot:03d}.{name}", []).append(value)
    return {name: tuple(items) for name, items in sorted(values.items())}


def unique_genotype_ratio(snapshot: PopulationSnapshot) -> float | None:
    """Return the fraction of available genotypes that are unique."""

    genotypes = tuple(
        record.genotype for record in snapshot if record.genotype is not None
    )
    return len(set(genotypes)) / len(genotypes) if genotypes else None


def duplicate_genotype_ratio(snapshot: PopulationSnapshot) -> float | None:
    """Return the fraction of available genotypes that duplicate another row."""

    unique_ratio = unique_genotype_ratio(snapshot)
    return None if unique_ratio is None else 1.0 - unique_ratio


def gene_entropy(snapshot: PopulationSnapshot) -> tuple[float, ...]:
    """Return Shannon entropy in bits for every gene position."""

    genotypes = _validated_genotypes(snapshot.records)
    if not genotypes:
        return ()
    population_size = len(genotypes)
    return tuple(
        -sum(
            (count / population_size) * log2(count / population_size)
            for count in Counter(genotype[gene] for genotype in genotypes).values()
        )
        for gene in range(len(genotypes[0]))
    )


def mean_pairwise_hamming_distance(snapshot: PopulationSnapshot) -> float | None:
    """Return mean normalized Hamming distance between genotype pairs."""

    genotypes = _validated_genotypes(snapshot.records)
    if len(genotypes) < 2:
        return None
    width = len(genotypes[0])
    if width == 0:
        return 0.0
    distances = [
        sum(left_gene != right_gene for left_gene, right_gene in zip(left, right, strict=True))
        / width
        for left_index, left in enumerate(genotypes)
        for right in genotypes[left_index + 1 :]
    ]
    return fmean(distances)


def objective_summary_by_batch(
    snapshots: Sequence[PopulationSnapshot],
) -> dict[int, dict[str, NumericSummary]]:
    """Summarize each raw objective independently for every batch."""

    result: dict[int, dict[str, NumericSummary]] = {}
    for snapshot in snapshots:
        values: dict[str, list[float]] = {}
        for record in snapshot.valid_objective_records:
            for name, value in record.objective_values.items():
                values.setdefault(name, []).append(value)
        result[snapshot.batch_index] = {
            name: _numeric_summary(items)
            for name, items in sorted(values.items())
        }
    return result


def score_summary_by_batch(
    snapshots: Sequence[PopulationSnapshot],
) -> dict[int, NumericSummary | None]:
    """Summarize aggregate scores for every batch where they are available."""

    return {
        snapshot.batch_index: (
            _numeric_summary(scores) if scores else None
        )
        for snapshot in snapshots
        for scores in [
            [
                record.aggregate_score
                for record in snapshot.valid_objective_records
                if record.aggregate_score is not None
            ]
        ]
    }


def best_score_evolution(
    snapshots: Sequence[PopulationSnapshot],
) -> dict[int, float | None]:
    """Return the highest aggregate score observed in each batch."""

    return {
        batch_index: (summary.maximum if summary is not None else None)
        for batch_index, summary in score_summary_by_batch(snapshots).items()
    }


def failure_rate_by_batch(
    snapshots: Sequence[PopulationSnapshot],
) -> dict[int, float | None]:
    """Return execution-failure fraction for every completed population."""

    return {
        snapshot.batch_index: (
            sum(
                record.evaluation_status == ResultStatus.FAILED.value
                for record in snapshot.records
            )
            / len(snapshot)
            if snapshot
            else None
        )
        for snapshot in snapshots
    }


def encode_genotypes_one_hot(
    records: Sequence[PopulationRecord],
) -> tuple[tuple[float, ...], ...]:
    """Encode categorical genes without assigning distances to category indexes."""

    genotypes = _validated_genotypes(records)
    if not genotypes:
        return ()
    categories = tuple(
        tuple(sorted({genotype[gene] for genotype in genotypes}))
        for gene in range(len(genotypes[0]))
    )
    return tuple(
        tuple(
            1.0 if genotype[gene] == category else 0.0
            for gene, gene_categories in enumerate(categories)
            for category in gene_categories
        )
        for genotype in genotypes
    )


def project_population_pca(
    records: Sequence[PopulationRecord],
) -> tuple[tuple[float, float], ...]:
    """Project one-hot genotypes onto two deterministic principal components."""

    encoded = encode_genotypes_one_hot(records)
    if not encoded:
        return ()
    if len(encoded) == 1:
        return ((0.0, 0.0),)

    import numpy as np

    matrix = np.asarray(encoded, dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    component_count = min(2, singular_values.size)
    coordinates = left[:, :component_count] * singular_values[:component_count]
    for component in range(component_count):
        loading = right[component]
        anchor = int(np.argmax(np.abs(loading)))
        if loading[anchor] < 0.0:
            coordinates[:, component] *= -1.0
    if component_count < 2:
        coordinates = np.column_stack((coordinates, np.zeros(len(encoded))))
    return tuple((float(row[0]), float(row[1])) for row in coordinates)


def best_individual_gene_matrix(
    records: Sequence[PopulationRecord],
    best_individual_ids: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    """Return genotypes of algorithm-selected individuals in history order."""

    identifiers = set(best_individual_ids)
    return tuple(
        record.genotype
        for record in records
        if record.individual_id in identifiers and record.genotype is not None
    )


def best_individual_objective_matrix(
    records: Sequence[PopulationRecord],
    best_individual_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    """Return a stable raw-objective matrix for selected individuals."""

    identifiers = set(best_individual_ids)
    selected = tuple(
        record
        for record in records
        if record.individual_id in identifiers and record.objectives_valid
    )
    if not selected:
        return (), ()
    names = tuple(selected[0].objective_values)
    if any(tuple(record.objective_values) != names for record in selected[1:]):
        raise ValueError("Selected individuals must expose the same objective names.")
    return names, tuple(
        tuple(record.objective_values[name] for name in names)
        for record in selected
    )


def _validated_genotypes(
    records: Sequence[PopulationRecord],
) -> tuple[tuple[int, ...], ...]:
    genotypes = tuple(record.genotype for record in records if record.genotype is not None)
    if len(genotypes) != len(records):
        raise ValueError("Every population record must provide a genotype.")
    if genotypes and any(len(genotype) != len(genotypes[0]) for genotype in genotypes):
        raise ValueError("Population genotypes must have equal lengths.")
    return genotypes


def _numeric_summary(values: Sequence[float]) -> NumericSummary:
    ordered = tuple(sorted(float(value) for value in values))
    if not ordered:
        raise ValueError("Cannot summarize an empty value sequence.")
    return NumericSummary(
        count=len(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=fmean(ordered),
        median=median(ordered),
        first_quartile=_percentile(ordered, 0.25),
        third_quartile=_percentile(ordered, 0.75),
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


__all__ = [
    "NumericSummary",
    "best_individual_gene_matrix",
    "best_individual_objective_matrix",
    "best_score_evolution",
    "duplicate_genotype_ratio",
    "encode_genotypes_one_hot",
    "failure_rate_by_batch",
    "gene_entropy",
    "gene_value_frequency",
    "mean_pairwise_hamming_distance",
    "objective_summary_by_batch",
    "parameter_distribution",
    "project_population_pca",
    "score_summary_by_batch",
    "stage_frequency_by_slot",
    "unique_genotype_ratio",
]
