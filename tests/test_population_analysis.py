from __future__ import annotations

import math

import pytest

from genio.statistics import (
    PopulationRecord,
    PopulationSnapshot,
    best_individual_gene_matrix,
    best_individual_objective_matrix,
    best_score_evolution,
    duplicate_genotype_ratio,
    encode_genotypes_one_hot,
    failure_rate_by_batch,
    gene_entropy,
    gene_value_frequency,
    mean_pairwise_hamming_distance,
    objective_summary_by_batch,
    parameter_distribution,
    project_population_pca,
    score_summary_by_batch,
    stage_frequency_by_slot,
    unique_genotype_ratio,
)


def record(
    identifier: str,
    position: int,
    genotype: tuple[int, ...] | None,
    *,
    batch_index: int = 0,
    score: float | None = None,
    quality: float | None = None,
    failed: bool = False,
) -> PopulationRecord:
    return PopulationRecord(
        proposal_id=f"run:{batch_index:03d}:{position:03d}",
        individual_id=identifier,
        batch_index=batch_index,
        batch_position=position,
        genotype=genotype,
        search_index=position,
        stages=("threshold" if genotype and genotype[0] == 0 else "blur", "output"),
        stage_parameters=(
            {"value": genotype[0] if genotype else 0},
            {"mode": "gray"},
        ),
        evaluation_status="failed" if failed else "success",
        objective_status="evaluation_failed" if failed else "valid",
        objective_values={} if failed else {"quality": quality if quality is not None else 0.0},
        aggregate_score=None if failed else score,
        error="failed" if failed else None,
    )


def population() -> PopulationSnapshot:
    return PopulationSnapshot(
        0,
        (
            record("a", 0, (0, 0), quality=0.2, score=0.2),
            record("b", 1, (0, 1), quality=0.8, score=0.8),
            record("c", 2, (1, 0), quality=0.6, score=0.6),
            record("d", 3, (0, 0), failed=True),
        ),
    )


def test_population_composition_summaries() -> None:
    snapshot = population()

    assert stage_frequency_by_slot(snapshot) == {
        0: {"blur": 1, "threshold": 3},
        1: {"output": 4},
    }
    assert gene_value_frequency(snapshot) == {
        0: {0: 3, 1: 1},
        1: {0: 3, 1: 1},
    }
    assert parameter_distribution(snapshot) == {
        "slot.000.value": (0, 0, 1, 0),
        "slot.001.mode": ("gray", "gray", "gray", "gray"),
    }


def test_population_diversity_statistics() -> None:
    snapshot = population()

    assert unique_genotype_ratio(snapshot) == 0.75
    assert duplicate_genotype_ratio(snapshot) == 0.25
    expected_entropy = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert gene_entropy(snapshot) == pytest.approx((expected_entropy, expected_entropy))
    assert mean_pairwise_hamming_distance(snapshot) == pytest.approx(0.5)


def test_population_objective_score_and_failure_evolution() -> None:
    first = population()
    second = PopulationSnapshot(
        1,
        (
            record("e", 0, (1, 1), batch_index=1, quality=0.9, score=0.9),
            record("f", 1, (1, 0), batch_index=1, quality=0.7, score=0.7),
        ),
    )

    objective = objective_summary_by_batch((first, second))
    scores = score_summary_by_batch((first, second))

    assert objective[0]["quality"].count == 3
    assert objective[0]["quality"].mean == pytest.approx(1.6 / 3.0)
    assert objective[1]["quality"].maximum == 0.9
    assert scores[0] is not None and scores[0].median == 0.6
    assert best_score_evolution((first, second)) == {0: 0.8, 1: 0.9}
    assert failure_rate_by_batch((first, second)) == {0: 0.25, 1: 0.0}


def test_population_one_hot_and_pca_are_deterministic() -> None:
    records = population().records
    encoded = encode_genotypes_one_hot(records)

    assert encoded[0] == (1.0, 0.0, 1.0, 0.0)
    assert all(len(row) == 4 for row in encoded)
    first_projection = project_population_pca(records)
    second_projection = project_population_pca(records)
    assert first_projection == second_projection
    assert len(first_projection) == 4
    assert sum(point[0] for point in first_projection) == pytest.approx(0.0)
    assert sum(point[1] for point in first_projection) == pytest.approx(0.0)


def test_population_best_individual_matrices_follow_history_order() -> None:
    records = population().records

    assert best_individual_gene_matrix(records, ("c", "b")) == ((0, 1), (1, 0))
    names, values = best_individual_objective_matrix(records, ("c", "b"))
    assert names == ("quality",)
    assert values == ((0.8,), (0.6,))


def test_population_genotype_analysis_rejects_missing_or_ragged_genotypes() -> None:
    missing = (record("missing", 0, None, quality=0.1),)
    ragged = (
        record("first", 0, (0,), quality=0.1),
        record("second", 1, (0, 1), quality=0.2),
    )

    with pytest.raises(ValueError, match="must provide a genotype"):
        encode_genotypes_one_hot(missing)
    with pytest.raises(ValueError, match="equal lengths"):
        gene_entropy(PopulationSnapshot(0, ragged))
