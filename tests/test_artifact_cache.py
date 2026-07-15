from threading import Lock

from genio import (
    EvaluationExecutor,
    EvaluationStep,
    EvaluationTask,
    EvaluationWorkflow,
    ExecutionContext,
    HLSImagePipelineSynthesisTask,
    ImageFunctionalMetricsArtifact,
    Individual,
    LFUArtifactCache,
    ParallelLocalBackend,
    PythonImageFunctionalTask,
    ResultStatus,
    StageChoice,
)


class CallCounter:
    def __init__(self) -> None:
        self.calls = 0
        self.lock = Lock()

    def increment(self) -> int:
        with self.lock:
            self.calls += 1
            return self.calls


class CacheableMetricTask(EvaluationTask):
    def cache_inputs(self):
        return {"semantic": self.metadata["semantic"]}

    def run(self, context: ExecutionContext):
        call = self.metadata["counter"].increment()
        if self.metadata.get("fail_first") and call == 1:
            raise RuntimeError("transient cache failure")
        return [
            ImageFunctionalMetricsArtifact(
                name="metrics",
                producer="cache_test",
                individual_id=self.individual.id,
                values={"score": float(self.metadata.get("score", 1.0))},
            )
        ]


class CacheableEmptyTask(EvaluationTask):
    def cache_inputs(self):
        return {"semantic": self.metadata["semantic"]}

    def run(self, context: ExecutionContext):
        self.metadata["counter"].increment()
        return []


class CacheableMetricStep(EvaluationStep):
    id = "cacheable"
    task_type = CacheableMetricTask

    def __init__(self, counter: CallCounter, *, fail_first: bool = False) -> None:
        self.counter = counter
        self.fail_first = fail_first

    def create_task(self, individual, artifacts):
        return CacheableMetricTask(
            individual=individual,
            step_id=self.id,
            metadata={
                "semantic": individual.metadata["semantic"],
                "score": individual.metadata.get("score", 1.0),
                "counter": self.counter,
                "fail_first": self.fail_first,
            },
        )


class CacheableEmptyStep(EvaluationStep):
    id = "empty"
    task_type = CacheableEmptyTask

    def __init__(self, counter: CallCounter) -> None:
        self.counter = counter

    def create_task(self, individual, artifacts):
        return CacheableEmptyTask(
            individual=individual,
            step_id=self.id,
            metadata={"semantic": "same", "counter": self.counter},
        )


def make_individual(
    identifier: str,
    *,
    semantic: str = "same",
    hls: dict | None = None,
    system: dict | None = None,
) -> Individual:
    return Individual.from_slots(
        id=identifier,
        scenario="cache_test",
        slots=[
            StageChoice(
                slot=0,
                stage="threshold",
                parameters={"threshold": 10, "maxval": 255},
            )
        ],
        design={"hls": hls or {}, "system": system or {}},
        metadata={"semantic": semantic},
    )


def metric_artifact(identifier: str, value: float = 1.0) -> ImageFunctionalMetricsArtifact:
    return ImageFunctionalMetricsArtifact(
        name="metrics",
        producer="cache_test",
        individual_id=identifier,
        values={"score": value},
        metadata={"nested": {"value": 1}},
    )


def test_lfu_cache_evicts_least_frequent_entry() -> None:
    cache = LFUArtifactCache({"step": 2})
    cache.put("step", "a", [metric_artifact("a")], source_individual_id="a")
    cache.put("step", "b", [metric_artifact("b")], source_individual_id="b")
    assert cache.get("step", "a", reads=2) is not None

    cache.put("step", "c", [metric_artifact("c")], source_individual_id="c")

    assert cache.get("step", "b") is None
    assert cache.get("step", "a") is not None
    assert cache.get("step", "c") is not None
    snapshot = cache.snapshot()["namespaces"]["step"]
    assert snapshot["entries"] == 2
    assert snapshot["evictions"] == 1


def test_lfu_cache_uses_lru_to_break_frequency_ties() -> None:
    cache = LFUArtifactCache({"step": 2})
    cache.put("step", "older", [metric_artifact("older")], source_individual_id="older")
    cache.put("step", "newer", [metric_artifact("newer")], source_individual_id="newer")

    cache.put("step", "third", [metric_artifact("third")], source_individual_id="third")

    assert cache.get("step", "older") is None
    assert cache.get("step", "newer") is not None


def test_cached_artifacts_are_rebound_and_deeply_copied() -> None:
    cache = LFUArtifactCache({"step": 1})
    cache.put(
        "step",
        "key",
        [metric_artifact("source")],
        source_individual_id="source",
    )
    entry = cache.get("step", "key")
    assert entry is not None

    first = entry.artifacts_for("first")[0]
    second = entry.artifacts_for("second")[0]
    first.metadata["nested"]["value"] = 99

    assert first.individual_id == "first"
    assert second.individual_id == "second"
    assert second.metadata["nested"]["value"] == 1
    assert second.metadata["cache_source_individual_id"] == "source"


def test_executor_coalesces_same_batch_cache_misses(tmp_path) -> None:
    counter = CallCounter()
    cache = LFUArtifactCache({"cacheable": 4})
    workflow = EvaluationWorkflow((CacheableMetricStep(counter),))
    individuals = tuple(make_individual(identifier) for identifier in ("a", "b", "c"))

    with ParallelLocalBackend(max_workers=3, base_work_dir=tmp_path) as backend:
        results = EvaluationExecutor(workflow, backend, cache).evaluate_many(individuals)

    assert counter.calls == 1
    assert all(result.status is ResultStatus.SUCCESS for result in results)
    assert [result.metadata["cache"]["cacheable"]["status"] for result in results] == [
        "miss",
        "coalesced",
        "coalesced",
    ]
    assert cache.snapshot()["namespaces"]["cacheable"]["coalesced"] == 2


def test_executor_reuses_cache_across_batches(tmp_path) -> None:
    counter = CallCounter()
    cache = LFUArtifactCache({"cacheable": 2})
    workflow = EvaluationWorkflow((CacheableMetricStep(counter),))

    with ParallelLocalBackend(max_workers=2, base_work_dir=tmp_path) as backend:
        executor = EvaluationExecutor(workflow, backend, cache)
        first = executor.evaluate(make_individual("first"))
        second = executor.evaluate(make_individual("second"))

    assert counter.calls == 1
    assert first.metadata["cache"]["cacheable"]["status"] == "miss"
    assert second.metadata["cache"]["cacheable"]["status"] == "hit"
    assert second.metrics == {"cacheable.score": 1.0}


def test_executor_does_not_cache_failures(tmp_path) -> None:
    counter = CallCounter()
    cache = LFUArtifactCache({"cacheable": 2})
    workflow = EvaluationWorkflow((CacheableMetricStep(counter, fail_first=True),))

    with ParallelLocalBackend(max_workers=1, base_work_dir=tmp_path) as backend:
        executor = EvaluationExecutor(workflow, backend, cache)
        failed_group = executor.evaluate_many(
            (make_individual("first"), make_individual("second"))
        )
        succeeded = executor.evaluate(make_individual("third"))

    assert all(result.status is ResultStatus.FAILED for result in failed_group)
    assert succeeded.status is ResultStatus.SUCCESS
    assert counter.calls == 2
    assert cache.snapshot()["namespaces"]["cacheable"]["stores"] == 1


def test_executor_caches_empty_artifact_bundles(tmp_path) -> None:
    counter = CallCounter()
    cache = LFUArtifactCache({"empty": 1})
    workflow = EvaluationWorkflow((CacheableEmptyStep(counter),))

    with ParallelLocalBackend(max_workers=1, base_work_dir=tmp_path) as backend:
        executor = EvaluationExecutor(workflow, backend, cache)
        first = executor.evaluate(make_individual("first"))
        second = executor.evaluate(make_individual("second"))

    assert first.status is ResultStatus.SUCCESS
    assert second.status is ResultStatus.SUCCESS
    assert second.metadata["cache"]["empty"]["status"] == "hit"
    assert counter.calls == 1


def test_executor_bypasses_cache_when_step_capacity_is_zero(tmp_path) -> None:
    counter = CallCounter()
    cache = LFUArtifactCache(default_capacity=0)
    workflow = EvaluationWorkflow((CacheableMetricStep(counter),))
    individuals = (make_individual("first"), make_individual("second"))

    with ParallelLocalBackend(max_workers=2, base_work_dir=tmp_path) as backend:
        results = EvaluationExecutor(workflow, backend, cache).evaluate_many(individuals)

    assert counter.calls == 2
    assert [result.metadata["cache"]["cacheable"]["status"] for result in results] == [
        "bypass",
        "bypass",
    ]
    assert cache.snapshot()["namespaces"]["cacheable"]["bypasses"] == 2


def test_lfu_cache_keeps_step_partitions_independent() -> None:
    cache = LFUArtifactCache({"python": 1, "hls": 1})
    cache.put("python", "p1", [metric_artifact("p1")], source_individual_id="p1")
    cache.put("hls", "h1", [metric_artifact("h1")], source_individual_id="h1")
    cache.put("python", "p2", [metric_artifact("p2")], source_individual_id="p2")

    assert cache.get("python", "p1") is None
    assert cache.get("python", "p2") is not None
    assert cache.get("hls", "h1") is not None


def test_production_tasks_select_only_relevant_cache_inputs() -> None:
    base = make_individual(
        "base",
        hls={"clock": 5, "npc": "XF_NPPC1"},
        system={"mcu": "small"},
    )
    system_variant = make_individual(
        "system",
        hls={"clock": 5, "npc": "XF_NPPC1"},
        system={"mcu": "large"},
    )
    hls_variant = make_individual(
        "hls",
        hls={"clock": 10, "npc": "XF_NPPC1"},
        system={"mcu": "small"},
    )

    python_base = PythonImageFunctionalTask(individual=base).cache_inputs()
    python_system = PythonImageFunctionalTask(individual=system_variant).cache_inputs()
    python_hls = PythonImageFunctionalTask(individual=hls_variant).cache_inputs()
    hls_base = HLSImagePipelineSynthesisTask(individual=base).cache_inputs()
    hls_system = HLSImagePipelineSynthesisTask(individual=system_variant).cache_inputs()
    hls_variant_inputs = HLSImagePipelineSynthesisTask(individual=hls_variant).cache_inputs()

    assert python_base == python_system == python_hls
    assert hls_base == hls_system
    assert hls_base != hls_variant_inputs
