"""Population statistics regression tests using synthetic evaluations only."""

import csv
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from genio.core.evaluation import Evaluation
from genio.core.individual import StageChoice
from genio.core.proposal import Proposal
from genio.core.result import Result
from genio.core.search_result import SearchResult
from genio.objective.base import (
    MetricObjective,
    ObjectiveDescriptor,
    ObjectiveSchema,
    ObjectiveSet,
    OptimizationDirection,
)
from genio.objective.runtime import (
    EvaluatedBatch,
    EvaluatedIndividual,
    ObjectiveEvaluationStatus,
    ObjectiveValues,
)
from genio.search_space.space import SearchSpace
from genio.search_space.spec import DesignDimensionSpec, SearchScenarioSpec, SlotSpec
from genio.statistics.population import PopulationAnalysisCollector
from genio.statistics.population_models import PopulationRecord, PopulationSnapshot
from genio.statistics.population_plots import (
    PopulationPlotConfig,
    PopulationPlotError,
    PopulationPlotRenderer,
)
from genio.statistics.population_statistics import PopulationStatisticsCollector


PARAMETER_NAMES = (
    "model.width",
    "model.height",
    "pipeline.0.stage",
    "pipeline.0.blur.radius",
    "pipeline.0.edge.radius",
    "runtime.payload",
    "runtime.learning_rate",
)


@pytest.fixture
def search_space():
    # Storage order deliberately differs from domain_order. A grouped dimension
    # contributes two real parameters but only one genotype gene.
    return SearchSpace.from_scenario(SearchScenarioSpec(
        id="population-tests",
        domain_order=("model", "pipeline", "runtime"),
        slots=(SlotSpec(0, (
            StageChoice(0, "blur", {"radius": 3}),
            StageChoice(0, "blur", {"radius": 7}),
            StageChoice(0, "edge", {"radius": 100}),
            StageChoice(0, "edge", {"radius": 200}),
        )),),
        design_spaces={
            "runtime": {
                "payload": DesignDimensionSpec((None, False, True, [1, 2], {"mode": "fast"})),
                "learning_rate": DesignDimensionSpec(tuple(index / 100 for index in range(21))),
            },
            "model": {
                "shape": DesignDimensionSpec(
                    ({"width": 16, "height": 32}, {"width": 64, "height": 128}),
                    parameter_names=("width", "height"),
                ),
            },
        },
    ))


@pytest.fixture
def schema():
    # Neither alphabetical order nor the order used in objective_values below.
    return ObjectiveSchema((
        ObjectiveDescriptor("quality", OptimizationDirection.MAXIMIZE),
        ObjectiveDescriptor("cost", OptimizationDirection.MINIMIZE),
        ObjectiveDescriptor("speed", OptimizationDirection.MAXIMIZE),
    ))


def _batch(search_space, schema, batch_index=0):
    genotypes = ((0, 0, 0, 0), (1, 2, 1, 10), (0, 1, 3, 19), (1, 3, 4, 5))
    proposals = []
    items = []
    for position, genotype in enumerate(genotypes):
        sequence = batch_index * len(genotypes) + position
        individual = search_space.from_genotype(genotype, id=f"individual-{sequence}")
        proposal = Proposal(
            proposal_id=f"proposal-{sequence}", proposal_sequence=sequence,
            batch_index=batch_index, batch_position=position, individual=individual,
        )
        raw = (float(position + 1 + batch_index), float(100 - 10 * position), float(20 + position))
        evaluation = Evaluation(
            individual, Result.success(individual.id, dict(zip(schema.names, raw))),
            metadata={
                "proposal_id": proposal.proposal_id, "proposal_sequence": sequence,
                "batch_index": batch_index, "batch_position": position,
            },
        )
        values = ObjectiveValues(
            schema.names, raw, (-raw[0], raw[1], -raw[2]),
            (raw[0], -raw[1], raw[2]), aggregate_score=float(position),
        )
        proposals.append(proposal)
        items.append(EvaluatedIndividual(evaluation, values, ObjectiveEvaluationStatus.VALID))
    return tuple(proposals), EvaluatedBatch(tuple(items), schema.names, batch_index=batch_index)


def _snapshot(proposals, batch):
    return PopulationSnapshot(batch.batch_index, tuple(
        PopulationRecord.from_proposal(proposal).with_evaluation(item)
        for proposal, item in zip(proposals, batch)
    ))


def _capture(collector, proposals, batch):
    collector.on_batch_started(batch.batch_index, tuple(p.individual for p in proposals))
    collector.on_proposals_generated(proposals)
    for evaluation in batch.evaluations:
        collector.on_evaluation_completed(evaluation)
    # Correlation is by proposal ID, not arrival order of objective results.
    collector.on_evaluated_batch(replace(batch, items=tuple(reversed(batch.items))))
    collector.on_batch_completed(batch.batch_index, batch.evaluations)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_configure_discovers_real_parameters_in_domain_order(search_space, schema):
    renderer = PopulationPlotRenderer()
    renderer.configure(search_space, schema)
    assert search_space.genotype_lengths == (2, 4, 5, 21)
    assert tuple(parameter.label for parameter in renderer._parameters) == PARAMETER_NAMES
    assert tuple(parameter.gene for parameter in renderer._parameters) == (0, 0, 1, 1, 1, 2, 3)
    assert renderer._parameters[0].alternatives == (16, 64)
    assert renderer._parameters[1].alternatives == (32, 128)
    assert renderer._parameters[2].alternatives == ("blur", "edge")
    assert renderer._parameters[3].alternatives == (3, 7)
    assert renderer._parameters[4].alternatives == (100, 200)
    assert renderer._parameters[5].alternatives == (None, False, True, [1, 2], {"mode": "fast"})


def test_tracked_genes_filters_without_reordering_real_parameters(search_space, schema):
    renderer = PopulationPlotRenderer(PopulationPlotConfig(tracked_genes=(2, 0, 1)))
    renderer.configure(search_space, schema)
    assert tuple(parameter.label for parameter in renderer._parameters) == PARAMETER_NAMES[:-1]
    assert tuple(parameter.gene for parameter in renderer._parameters) == (0, 0, 1, 1, 1, 2)
    invalid = PopulationPlotRenderer(PopulationPlotConfig(tracked_genes=(4,)))
    with pytest.raises(ValueError, match="unavailable gene"):
        invalid.configure(search_space, schema)


@pytest.mark.parametrize("domain_order", [(), ("runtime", "pipeline", "model")])
def test_configure_default_and_explicit_pipeline_order(search_space, domain_order):
    space = SearchSpace.from_scenario(replace(search_space.scenario, domain_order=domain_order))
    renderer = PopulationPlotRenderer()
    renderer.configure(space, None)
    expected = (
        PARAMETER_NAMES[2:5] + PARAMETER_NAMES[5:] + PARAMETER_NAMES[:2]
        if not domain_order else PARAMETER_NAMES[5:] + PARAMETER_NAMES[2:5] + PARAMETER_NAMES[:2]
    )
    assert tuple(parameter.label for parameter in renderer._parameters) == expected


@pytest.mark.parametrize(
    ("label", "categories", "counts"),
    [
        ("model.width", ("16", "64"), (2, 2)),
        ("model.height", ("32", "128"), (2, 2)),
        ("pipeline.0.stage", ("'blur'", "'edge'"), (2, 2)),
        ("pipeline.0.blur.radius", ("3", "7"), (1, 1)),
        ("pipeline.0.edge.radius", ("100", "200"), (1, 1)),
        ("runtime.payload", ("None", "False", "True", "[1, 2]", "{'mode': 'fast'}"), (1, 1, 0, 1, 1)),
    ],
)
def test_low_cardinality_single_batch_is_stacked_counts(
    search_space, schema, label, categories, counts,
):
    renderer = PopulationPlotRenderer()
    renderer.configure(search_space, schema)
    parameter = next(item for item in renderer._parameters if item.label == label)
    snapshot = _snapshot(*_batch(search_space, schema))
    figure = renderer._plot_parameter_evolution((snapshot,), parameter)
    try:
        axis = figure.axes[0]
        assert axis.get_title() == label
        assert axis.get_ylabel() == "Individuals (proposals)"
        assert tuple(axis.get_legend_handles_labels()[1]) == categories
        assert [patch.get_height() for patch in axis.patches] == list(counts)
        assert [patch.get_y() for patch in axis.patches] == list(np.cumsum((0,) + counts[:-1]))
        assert not axis.lines  # Numeric categories must not become index/mean curves.
    finally:
        figure.clear()


def test_multiple_batches_stack_actual_counts_including_unobserved_values(
    search_space, schema, monkeypatch,
):
    renderer = PopulationPlotRenderer()
    renderer.configure(search_space, schema)
    snapshots = tuple(_snapshot(*_batch(search_space, schema, index)) for index in (0, 2))
    parameter = next(item for item in renderer._parameters if item.label == "runtime.payload")
    original = Axes.stackplot
    calls = []

    def stackplot(axis, x, *series, **kwargs):
        calls.append((x, series, kwargs))
        return original(axis, x, *series, **kwargs)

    monkeypatch.setattr(Axes, "stackplot", stackplot)
    figure = renderer._plot_parameter_evolution(snapshots, parameter)
    try:
        assert len(calls) == 1
        x, series, kwargs = calls[0]
        assert x == [0, 2]
        assert series == ([1, 1], [1, 1], [0, 0], [1, 1], [1, 1])
        assert kwargs["labels"] == ("None", "False", "True", "[1, 2]", "{'mode': 'fast'}")
        assert len(figure.axes[0].collections) == 5
        assert not figure.axes[0].lines
    finally:
        figure.clear()


def test_mapping_categories_escape_keys_and_ignore_insertion_order():
    space = SearchSpace.from_scenario(SearchScenarioSpec(
        id="mapping-categories",
        design_spaces={"config": {"payload": DesignDimensionSpec((
            {"a": 1, "b": 2}, {"a: 1, b": 2}, {"b": 2, "a": 1},
        ))}},
    ))
    renderer = PopulationPlotRenderer()
    renderer.configure(space, None)
    records = tuple(PopulationRecord.from_proposal(Proposal(
        proposal_id=f"mapping-{position}", proposal_sequence=position,
        batch_index=0, batch_position=position,
        individual=space.from_genotype((position,)),
    )) for position in range(3))
    figure = renderer._plot_parameter_evolution((PopulationSnapshot(0, records),), renderer._parameters[0])
    try:
        axis = figure.axes[0]
        assert axis.get_legend_handles_labels()[1] == ["{'a': 1, 'b': 2}", "{'a: 1, b': 2}"]
        assert [patch.get_height() for patch in axis.patches] == [2, 1]
        assert [patch.get_y() for patch in axis.patches] == [0, 2]
    finally:
        figure.clear()


@pytest.mark.parametrize("cardinality", [20, 21])
def test_numeric_cardinality_boundary_uses_boxplot_and_real_mean(
    search_space, schema, monkeypatch, cardinality,
):
    renderer = PopulationPlotRenderer()
    renderer.configure(search_space, schema)
    parameter = replace(renderer._parameters[-1], alternatives=tuple(index / 100 for index in range(cardinality)))
    snapshots = tuple(_snapshot(*_batch(search_space, schema, index)) for index in (0, 2))
    # Keep the real Matplotlib method bound to its axis while recording inputs.
    original = Axes.boxplot
    calls = []

    def capture_boxplot(axis, values, **kwargs):
        calls.append((values, kwargs))
        return original(axis, values, **kwargs)

    monkeypatch.setattr(Axes, "boxplot", capture_boxplot)
    figure = renderer._plot_parameter_evolution(snapshots, parameter)
    try:
        axis = figure.axes[0]
        if cardinality == 20:
            assert not calls
            assert not axis.lines
            assert len(axis.collections) == 20
        else:
            assert len(calls) == 1
            values, kwargs = calls[0]
            assert values == [[0.0, 0.1, 0.19, 0.05]] * 2
            assert kwargs["positions"] == [0, 2]
            mean = next(line for line in axis.lines if line.get_label() == "mean")
            np.testing.assert_allclose(mean.get_xdata(), [0, 2])
            np.testing.assert_allclose(mean.get_ydata(), [0.085, 0.085])
    finally:
        figure.clear()


def test_objective_axes_follow_schema_with_median_iqr_and_directional_best(search_space, schema):
    renderer = PopulationPlotRenderer()
    renderer.configure(search_space, schema)
    snapshots = []
    for batch_index in (0, 2):
        snapshot = _snapshot(*_batch(search_space, schema, batch_index))
        # Objective mapping insertion order must not control subplot ordering.
        records = tuple(replace(record, objective_values={
            name: record.objective_values[name] for name in reversed(schema.names)
        }) for record in snapshot)
        snapshots.append(PopulationSnapshot(batch_index, records))
    figure = renderer._plot_objective_evolution(snapshots)
    try:
        assert [axis.get_ylabel() for axis in figure.axes] == list(schema.names)
        expected = (
            ([2.5, 4.5], [1.75, 3.75], [3.25, 5.25], [4, 6], "maximize"),
            ([85, 85], [77.5, 77.5], [92.5, 92.5], [70, 70], "minimize"),
            ([21.5, 21.5], [20.75, 20.75], [22.25, 22.25], [23, 23], "maximize"),
        )
        for axis, (medians, lower, upper, best, direction) in zip(figure.axes, expected):
            lines = {line.get_label(): line for line in axis.lines}
            np.testing.assert_allclose(lines["median"].get_xdata(), [0, 2])
            np.testing.assert_allclose(lines["median"].get_ydata(), medians)
            np.testing.assert_allclose(lines[f"best ({direction})"].get_ydata(), best)
            band = axis.collections[0]
            assert band.get_label() == "interquartile range"
            vertices = np.concatenate([path.vertices for path in band.get_paths()])
            for index, batch_index in enumerate((0, 2)):
                y = vertices[vertices[:, 0] == batch_index, 1]
                assert y.min() == pytest.approx(lower[index])
                assert y.max() == pytest.approx(upper[index])
        assert not figure.axes[0].get_shared_y_axes().joined(figure.axes[0], figure.axes[1])
    finally:
        figure.clear()


def test_three_objectives_produce_all_pairs_in_schema_order(search_space, schema):
    renderer = PopulationPlotRenderer()
    renderer.configure(search_space, schema)
    snapshot = _snapshot(*_batch(search_space, schema))
    best = snapshot[1]
    figure = renderer._plot_objective_scatter(snapshot.records, (best.individual_id,))
    try:
        axes = [axis for axis in figure.axes if axis.get_xlabel() in schema.names]
        pairs = [("quality", "cost"), ("quality", "speed"), ("cost", "speed")]
        assert [(axis.get_xlabel(), axis.get_ylabel()) for axis in axes] == pairs
        for axis, (left, right) in zip(axes, pairs):
            np.testing.assert_allclose(axis.collections[0].get_offsets(), [
                [record.objective_values[left], record.objective_values[right]]
                for record in snapshot
            ])
            np.testing.assert_allclose(axis.collections[1].get_offsets(), [[
                best.objective_values[left], best.objective_values[right],
            ]])
            assert axis.collections[1].get_label() == "algorithm best"
    finally:
        figure.clear()


@pytest.mark.parametrize("image_format", ["png", "svg"])
def test_renderer_writes_real_images_with_real_parameter_names(
    search_space, schema, tmp_path, image_format,
):
    renderer = PopulationPlotRenderer(PopulationPlotConfig(image_format=image_format, strict=True))
    renderer.configure(search_space, schema)
    result = renderer.render((_snapshot(*_batch(search_space, schema)),), target_dir=tmp_path)
    assert not result.warnings
    assert not result.skipped
    expected_parameters = tuple(
        f"parameter_{index:03d}_{name.replace('.', '_')}_evolution.{image_format}"
        for index, name in enumerate(PARAMETER_NAMES)
    )
    assert tuple(name for name in result.generated if name.startswith("parameter_")) == expected_parameters
    assert len(result.generated) == 9 + len(PARAMETER_NAMES)
    assert set(result.generated) == {path.name for path in tmp_path.iterdir()}
    for name in result.generated:
        content = (tmp_path / name).read_bytes()
        if image_format == "png":
            assert content.startswith(b"\x89PNG\r\n\x1a\n")
        else:
            assert b"<svg" in content
            assert b"</svg>" in content


@pytest.mark.parametrize("strict", [False, True])
def test_render_failure_policy_preserves_cause_and_clears_figure(tmp_path, monkeypatch, strict):
    renderer = PopulationPlotRenderer(PopulationPlotConfig(strict=strict))
    figure, _ = renderer._figure("synthetic save failure")
    monkeypatch.setattr(renderer, "_plot_population_size", lambda snapshots: figure)
    failure = OSError("synthetic disk failure")
    monkeypatch.setattr(Figure, "savefig", Mock(side_effect=failure))
    if strict:
        with pytest.raises(PopulationPlotError, match="population_size: OSError: synthetic disk failure") as caught:
            renderer.render((), target_dir=tmp_path)
        assert caught.value.__cause__ is failure
    else:
        result = renderer.render((), target_dir=tmp_path)
        assert result.warnings == ("population_size: OSError: synthetic disk failure",)
        assert not result.generated
        assert "objective_evolution" in result.skipped
    assert not figure.axes


@pytest.mark.parametrize("state", ["empty", "pending", "evaluated"])
def test_no_objectives_skip_objective_plots_without_errors(search_space, schema, tmp_path, state):
    renderer = PopulationPlotRenderer(PopulationPlotConfig(strict=True))
    renderer.configure(search_space, None)
    snapshots = ()
    if state != "empty":
        proposals, batch = _batch(search_space, schema)
        records = tuple(PopulationRecord.from_proposal(p) for p in proposals)
        if state == "evaluated":
            batch = EvaluatedBatch.from_evaluations(batch.evaluations, batch_index=0)
            records = tuple(record.with_evaluation(item) for record, item in zip(records, batch))
        snapshots = (PopulationSnapshot(0, records),)
    result = renderer.render(snapshots, target_dir=tmp_path)
    assert not result.warnings
    assert {"objective_evolution", "objective_scatter", "aggregate_score_evolution"} <= result.skipped.keys()
    if state == "empty":
        assert not result.generated


@pytest.mark.parametrize("with_proposals", [False, True])
def test_configured_objectives_without_evaluations_are_skipped(
    search_space, schema, tmp_path, with_proposals,
):
    renderer = PopulationPlotRenderer(PopulationPlotConfig(strict=True))
    renderer.configure(search_space, schema)
    snapshots = ()
    if with_proposals:
        proposals, _ = _batch(search_space, schema)
        snapshots = (PopulationSnapshot(0, tuple(PopulationRecord.from_proposal(p) for p in proposals)),)
    result = renderer.render(snapshots, target_dir=tmp_path)
    assert not result.warnings
    if not with_proposals:
        assert not result.generated
    assert {"objective_evolution", "objective_scatter", "aggregate_score_evolution"} <= result.skipped.keys()


def test_analysis_lifecycle_and_checkpoint_reconfigure_same_names_and_content(
    search_space, schema, tmp_path,
):
    config = PopulationPlotConfig(every_batches=2, strict=True)
    session = SimpleNamespace(search_space=search_space, objective_set=SimpleNamespace(schema=schema))
    collector = PopulationAnalysisCollector(tmp_path / "original", plots=config)
    collector.on_session_started(session)
    batches = []
    for batch_index in (0, 2):
        proposals, batch = _batch(search_space, schema, batch_index)
        _capture(collector, proposals, batch)
        batches.append(batch)
        if batch_index == 0:
            assert not collector.manifest_path.exists()
            assert not collector.summary_path.exists()
        else:
            assert _json(collector.summary_path)["records"] == 8
            cadence_manifest = _json(collector.manifest_path)
            assert not cadence_manifest["warnings"]
            assert all(name.startswith("plots/batch_000002/") for name in cadence_manifest["generated"])
            assert len(cadence_manifest["generated"]) == 9 + len(PARAMETER_NAMES)
    evaluations = tuple(item for batch in batches for item in batch.evaluations)
    best = batches[-1][1].individual
    collector.on_session_completed(SearchResult("synthetic", evaluations, (best,)))
    assert collector.best_individual_ids == (best.id,)
    assert [snapshot.batch_index for snapshot in collector.snapshots] == [0, 2]
    assert [record.proposal_id for record in collector.records] == [f"proposal-{i}" for i in (*range(4), *range(8, 12))]
    assert collector.snapshot()["evaluated_records"] == 8
    assert collector.snapshot()["valid_objective_records"] == 8
    manifest = _json(collector.manifest_path)
    assert not manifest["warnings"]
    assert {name.split("/")[1] for name in manifest["generated"]} == {"batch_000002", "final"}
    summary = _json(collector.summary_path)
    assert summary["objectives_by_batch"]["0"]["quality"]["median"] == 2.5
    assert summary["best_individual_ids"] == [best.id]

    restored = PopulationAnalysisCollector(tmp_path / "restored", plots=config)
    # Stale renderer configuration must be replaced from the restored session.
    stale_space = SearchSpace.from_scenario(SearchScenarioSpec(
        id="stale", design_spaces={"old": {"obsolete": DesignDimensionSpec((1, 2))}},
    ))
    restored.on_session_started(SimpleNamespace(search_space=stale_space, objective_set=None))
    restored.restore_checkpoint_state(
        collector.checkpoint_state(), session=session, evaluations=evaluations,
        evaluated_batches=tuple(batches), completed=True,
    )
    assert restored.snapshots == collector.snapshots
    assert restored.best_individual_ids == collector.best_individual_ids
    assert tuple(p.label for p in restored._renderer._parameters) == PARAMETER_NAMES
    assert restored._renderer._objective_schema == schema
    assert _json(restored.summary_path) == summary
    assert _json(restored.manifest_path) == manifest
    for name in manifest["generated"]:
        assert (restored.output_dir / name).read_bytes() == (collector.output_dir / name).read_bytes()


def test_session_restart_resets_history_and_reconfigures_renderer(search_space, schema, tmp_path):
    collector = PopulationAnalysisCollector(tmp_path, plots=PopulationPlotConfig(final_plots=False))
    collector.on_session_started(SimpleNamespace(
        search_space=search_space, objective_set=SimpleNamespace(schema=schema),
    ))
    proposals, batch = _batch(search_space, schema)
    _capture(collector, proposals, batch)
    collector.on_session_completed(SearchResult("first", batch.evaluations, (batch[0].individual,)))
    assert collector.summary_path.exists()
    assert collector.best_individual_ids
    replacement = SearchSpace.from_scenario(SearchScenarioSpec(
        id="replacement", design_spaces={"new": {"actual_name": DesignDimensionSpec((None, True))}},
    ))
    collector.on_session_started(SimpleNamespace(search_space=replacement, objective_set=None))
    assert collector.records == ()
    assert collector.snapshots == ()
    assert collector.best_individual_ids == ()
    assert not collector.summary_path.exists()
    assert not collector.manifest_path.exists()
    assert tuple(p.label for p in collector._renderer._parameters) == ("new.actual_name",)
    assert collector._renderer._objective_schema is None


def test_facade_integrates_csv_and_population_analysis(search_space, tmp_path):
    objectives = ObjectiveSet(tuple(
        MetricObjective(name, direction)
        for name, direction in (("quality", "maximize"), ("cost", "minimize"), ("speed", "maximize"))
    ))
    session = SimpleNamespace(
        search_space=search_space, objective_set=objectives, run_id="synthetic-run", id="synthetic-session",
        algorithm=SimpleNamespace(), max_workers=1, cache_capacity={}, work_dir=tmp_path,
        execution_metadata={}, metadata={},
        evaluation_workflow=SimpleNamespace(execution_order=lambda: ()),
    )
    collector = PopulationStatisticsCollector(
        tmp_path, individuals_filename="proposals.csv",
        plots=PopulationPlotConfig(final_plots=False, strict=True),
    )
    assert collector.supports_checkpointing
    collector.on_session_started(session)
    proposals, batch = _batch(search_space, objectives.schema)
    _capture(collector, proposals, batch)
    collector.on_session_completed(SearchResult(session.id, batch.evaluations, (batch[0].individual,)))
    with (tmp_path / "proposals.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["proposal_id"] for row in rows] == [p.proposal_id for p in proposals]
    assert {row["evaluation_status"] for row in rows} == {"success"}
    assert {row["objective_status"] for row in rows} == {"valid"}
    assert [float(row["objective.quality.raw"]) for row in rows] == [1, 2, 3, 4]
    assert json.loads(rows[0]["design_json"])["model"] == {"width": 16, "height": 32}
    assert _json(tmp_path / "run_summary.json")["evaluated_individuals"] == 4
    assert _json(tmp_path / "analysis" / "analysis_summary.json")["records"] == 4
    assert set(collector.snapshot()) == {"csv", "analysis"}
    assert collector.snapshot()["analysis"]["best_individual_ids"] == [batch[0].individual.id]
