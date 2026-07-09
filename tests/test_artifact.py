from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pytest

from genio import Artifact, ArtifactError, MetricArtifact


@dataclass(frozen=True, slots=True)
class TextArtifact(Artifact):
    content: str

    def load(self) -> Sequence[Any]:
        return [self.content]


@dataclass(frozen=True, slots=True)
class ScoreArtifact(MetricArtifact):
    values: Mapping[str, float]

    def load(self) -> Sequence[Any]:
        return [dict(self.values)]

    def metrics(self) -> Mapping[str, float]:
        return self.values


def test_artifact_base_keeps_common_metadata():
    artifact = TextArtifact(
        name="report",
        producer="TestEvaluator",
        individual_id="individual_001",
        objective="functional",
        metadata={"metric": "accuracy"},
        content="result=ok\n",
    )

    loaded = artifact.load()

    assert artifact.producer == "TestEvaluator"
    assert artifact.name == "report"
    assert artifact.individual_id == "individual_001"
    assert artifact.objective == "functional"
    assert artifact.metadata == {"metric": "accuracy"}
    assert loaded == ["result=ok\n"]


def test_artifact_base_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Artifact(name="report", producer="TestEvaluator", individual_id="individual_001")


def test_metric_artifact_exposes_numeric_metrics():
    artifact = ScoreArtifact(
        name="scores",
        producer="ScoreTask",
        individual_id="individual_001",
        objective="quality",
        values={"f1": 0.9, "latency": 12.0},
    )

    assert artifact.load() == [{"f1": 0.9, "latency": 12.0}]
    assert artifact.metrics() == {"f1": 0.9, "latency": 12.0}


def test_metric_artifact_requires_metrics_method():
    @dataclass(frozen=True, slots=True)
    class IncompleteMetricArtifact(MetricArtifact):
        def load(self) -> Sequence[Any]:
            return []

    with pytest.raises(TypeError):
        IncompleteMetricArtifact(
            name="scores",
            producer="ScoreTask",
            individual_id="individual_001",
        )


def test_artifact_error_is_available_for_custom_artifacts():
    with pytest.raises(ArtifactError, match="custom failure"):
        raise ArtifactError("custom failure")
