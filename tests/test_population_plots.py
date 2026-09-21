from __future__ import annotations

from pathlib import Path

import pytest

from genio.statistics import (
    PopulationPlotConfig,
    PopulationPlotError,
    PopulationPlotRenderer,
    PopulationRecord,
    PopulationSnapshot,
)


def record(
    identifier: str,
    batch: int,
    position: int,
    genotype: tuple[int, int],
    quality: float,
    latency: float,
    *,
    score: float | None,
    failed: bool = False,
) -> PopulationRecord:
    return PopulationRecord(
        proposal_id=f"run:{batch:03d}:{position:03d}",
        individual_id=identifier,
        batch_index=batch,
        batch_position=position,
        genotype=genotype,
        search_index=batch * 10 + position,
        stages=("threshold" if genotype[0] == 0 else "blur", "output"),
        stage_parameters=({"value": genotype[0]}, {"mode": "gray"}),
        evaluation_status="failed" if failed else "success",
        objective_status="evaluation_failed" if failed else "valid",
        objective_values=(
            {} if failed else {"quality": quality, "latency": latency}
        ),
        aggregate_score=None if failed else score,
        error="failed" if failed else None,
    )


def snapshots() -> tuple[PopulationSnapshot, ...]:
    return (
        PopulationSnapshot(
            0,
            (
                record("a", 0, 0, (0, 0), 0.5, 20.0, score=0.4),
                record("b", 0, 1, (0, 1), 0.7, 18.0, score=0.6),
                record("c", 0, 2, (1, 0), 0.0, 0.0, score=None, failed=True),
            ),
        ),
        PopulationSnapshot(
            1,
            (
                record("d", 1, 0, (1, 1), 0.8, 17.0, score=0.7),
                record("e", 1, 1, (0, 1), 0.9, 16.0, score=0.8),
                record("f", 1, 2, (1, 0), 0.75, 15.0, score=0.65),
            ),
        ),
    )


def assert_non_empty_files(target: Path, filenames: tuple[str, ...]) -> None:
    for filename in filenames:
        path = target / filename
        assert path.is_file()
        assert path.stat().st_size > 0


def test_population_plot_config_validates_public_options() -> None:
    assert PopulationPlotConfig(every_batches=2, tracked_genes=(0, 2)).every_batches == 2
    with pytest.raises(ValueError, match="positive integer"):
        PopulationPlotConfig(every_batches=0)
    with pytest.raises(ValueError, match="duplicates"):
        PopulationPlotConfig(tracked_genes=(1, 1))
    with pytest.raises(ValueError, match="png.*svg"):
        PopulationPlotConfig(image_format="jpg")


def test_population_plot_renderer_generates_all_mvp_plots(tmp_path) -> None:
    renderer = PopulationPlotRenderer(PopulationPlotConfig(image_format="png"))

    result = renderer.render(
        snapshots(),
        best_individual_ids=("e", "f"),
        target_dir=tmp_path,
    )

    assert result.skipped == {}
    assert result.warnings == ()
    assert set(result.generated) == {
        "population_size.png",
        "stage_distribution.png",
        "gene_distribution.png",
        "gene_entropy.png",
        "unique_genotypes.png",
        "genotype_distance.png",
        "population_projection.png",
        "objective_evolution.png",
        "aggregate_score_evolution.png",
        "best_individuals_genes.png",
        "failure_rate.png",
        "objective_scatter.png",
    }
    assert_non_empty_files(tmp_path, result.generated)


def test_population_plot_renderer_skips_inapplicable_plots(tmp_path) -> None:
    sparse = PopulationSnapshot(
        0,
        (
            PopulationRecord(
                proposal_id="run:0",
                individual_id="only",
                batch_index=0,
                batch_position=0,
                genotype=(0,),
                search_index=0,
                stages=("nop",),
                stage_parameters=({},),
                evaluation_status="success",
                objective_status="not_configured",
            ),
        ),
    )

    result = PopulationPlotRenderer().render((sparse,), target_dir=tmp_path)

    assert "objective_evolution" in result.skipped
    assert "aggregate_score_evolution" in result.skipped
    assert "best_individuals_genes" in result.skipped
    assert "objective_scatter" in result.skipped
    assert "population_projection" in result.skipped
    assert result.warnings == ()
    assert_non_empty_files(tmp_path, result.generated)


def test_population_plot_renderer_reports_or_raises_render_failures(
    tmp_path,
    monkeypatch,
) -> None:
    def fail(_self, _snapshots):
        raise RuntimeError("planned renderer failure")

    monkeypatch.setattr(PopulationPlotRenderer, "_plot_population_size", fail)
    result = PopulationPlotRenderer().render(snapshots(), target_dir=tmp_path / "lenient")
    assert result.warnings == (
        "population_size: RuntimeError: planned renderer failure",
    )

    strict = PopulationPlotRenderer(PopulationPlotConfig(strict=True))
    with pytest.raises(PopulationPlotError, match="planned renderer failure"):
        strict.render(snapshots(), target_dir=tmp_path / "strict")
