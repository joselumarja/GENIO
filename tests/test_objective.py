import pytest

from genio import (
    Evaluation,
    Individual,
    MetricObjective,
    ObjectiveError,
    ObjectiveSet,
    OptimizationDirection,
    Result,
    StageChoice,
    dominates,
)


def make_evaluation(metrics: dict[str, float]) -> Evaluation:
    individual = Individual.from_slots(
        id="individual_001",
        scenario="objective_space",
        slots=[StageChoice(slot=0, stage="nop")],
    )
    return Evaluation(
        individual=individual,
        result=Result.success(individual.id, metrics=metrics),
    )


def test_metric_objective_reads_metric_value():
    evaluation = make_evaluation({"quality.f1": 0.9})
    objective = MetricObjective(
        metric="quality.f1",
        optimization_direction=OptimizationDirection.MAXIMIZE,
    )

    assert objective.name == "quality.f1"
    assert objective.direction is OptimizationDirection.MAXIMIZE
    assert objective.value(evaluation) == 0.9
    assert objective.score(evaluation) == 0.9


def test_metric_objective_minimize_score_is_negated():
    evaluation = make_evaluation({"hls.latency": 12.0})
    objective = MetricObjective(
        metric="hls.latency",
        optimization_direction=OptimizationDirection.MINIMIZE,
    )

    assert objective.value(evaluation) == 12.0
    assert objective.score(evaluation) == -12.0


def test_metric_objective_can_use_custom_name():
    objective = MetricObjective(
        metric="hls.latency",
        optimization_direction=OptimizationDirection.MINIMIZE,
        id="latency",
    )

    assert objective.name == "latency"


def test_metric_objective_rejects_missing_metric():
    evaluation = make_evaluation({})
    objective = MetricObjective(
        metric="quality.f1",
        optimization_direction=OptimizationDirection.MAXIMIZE,
    )

    with pytest.raises(ObjectiveError, match="Metric 'quality.f1' is not available"):
        objective.value(evaluation)


def test_metric_objective_rejects_non_numeric_metric():
    evaluation = make_evaluation({"quality.f1": "bad"})
    objective = MetricObjective(
        metric="quality.f1",
        optimization_direction=OptimizationDirection.MAXIMIZE,
    )

    with pytest.raises(ObjectiveError, match="Metric 'quality.f1' must be numeric"):
        objective.value(evaluation)


def test_metric_objective_rejects_boolean_metric():
    evaluation = make_evaluation({"quality.ok": True})
    objective = MetricObjective(
        metric="quality.ok",
        optimization_direction=OptimizationDirection.MAXIMIZE,
    )

    with pytest.raises(ObjectiveError, match="Metric 'quality.ok' must be numeric"):
        objective.value(evaluation)


def test_objective_set_returns_values_and_scores():
    evaluation = make_evaluation({"quality.f1": 0.9, "hls.latency": 12.0})
    objectives = ObjectiveSet(
        (
            MetricObjective("quality.f1", OptimizationDirection.MAXIMIZE, id="f1"),
            MetricObjective("hls.latency", OptimizationDirection.MINIMIZE, id="latency"),
        )
    )

    assert objectives.values(evaluation) == {"f1": 0.9, "latency": 12.0}
    assert objectives.scores(evaluation) == {"f1": 0.9, "latency": -12.0}


def test_objective_set_rejects_empty_objectives():
    with pytest.raises(ObjectiveError, match="requires at least one objective"):
        ObjectiveSet(())


def test_objective_set_rejects_duplicate_names():
    with pytest.raises(ObjectiveError, match="Duplicate objective names"):
        ObjectiveSet(
            (
                MetricObjective("quality.f1", OptimizationDirection.MAXIMIZE, id="score"),
                MetricObjective("quality.accuracy", OptimizationDirection.MAXIMIZE, id="score"),
            )
        )


def test_dominates_uses_objective_directions():
    better = make_evaluation({"quality.f1": 0.9, "hls.latency": 80.0})
    worse = make_evaluation({"quality.f1": 0.8, "hls.latency": 100.0})
    objectives = ObjectiveSet(
        (
            MetricObjective("quality.f1", OptimizationDirection.MAXIMIZE),
            MetricObjective("hls.latency", OptimizationDirection.MINIMIZE),
        )
    )

    assert dominates(better, worse, objectives)
    assert not dominates(worse, better, objectives)


def test_dominates_returns_false_for_tradeoff():
    left = make_evaluation({"quality.f1": 0.9, "hls.latency": 100.0})
    right = make_evaluation({"quality.f1": 0.8, "hls.latency": 80.0})
    objectives = ObjectiveSet(
        (
            MetricObjective("quality.f1", OptimizationDirection.MAXIMIZE),
            MetricObjective("hls.latency", OptimizationDirection.MINIMIZE),
        )
    )

    assert not dominates(left, right, objectives)
    assert not dominates(right, left, objectives)


def test_dominates_returns_false_for_equal_values():
    left = make_evaluation({"quality.f1": 0.9, "hls.latency": 80.0})
    right = make_evaluation({"quality.f1": 0.9, "hls.latency": 80.0})
    objectives = ObjectiveSet(
        (
            MetricObjective("quality.f1", OptimizationDirection.MAXIMIZE),
            MetricObjective("hls.latency", OptimizationDirection.MINIMIZE),
        )
    )

    assert not dominates(left, right, objectives)
