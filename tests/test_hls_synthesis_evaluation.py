from pathlib import Path

from genio import ExecutionContext
from genio import HLSSynthesisEvaluationStep
from genio import HLSSynthesisTask
from genio import SearchScenarioSpec
from genio import SearchSpace
from genio import SlotSpec
from genio import StageChoice

import pytest


def make_individual():
    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="hls_synthesis_test",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(StageChoice(slot=0, stage="nop"),),
                ),
            ),
        )
    )
    return search_space.from_index(0)


def test_hls_synthesis_step_creates_task() -> None:
    individual = make_individual()
    step = HLSSynthesisEvaluationStep(
        hls_config=Path("config/hls_config.cfg"),
        work_dir_name="hls_work",
        top_function="pipeline_top",
        clock_period=5.0,
        part="xczu9eg-ffvb1156-2-e",
        config_defaults={"hls.clock": "5", "hls.flow_target": "vivado"},
        config_overrides={"hls.flow_target": "vivado"},
    )

    task = step.create_task(individual, artifacts={})

    assert isinstance(task, HLSSynthesisTask)
    assert task.step_id == step.id
    assert task.hls_tool == "v++"
    assert task.hls_config == Path("config/hls_config.cfg")
    assert task.work_dir_name == "hls_work"
    assert task.top_function == "pipeline_top"
    assert task.clock_period == 5.0
    assert task.part == "xczu9eg-ffvb1156-2-e"
    assert task.config_defaults == {"hls.clock": "5", "hls.flow_target": "vivado"}
    assert task.config_overrides == {"hls.flow_target": "vivado"}


def test_hls_synthesis_task_is_not_implemented(tmp_path) -> None:
    task = HLSSynthesisTask(individual=make_individual())

    with pytest.raises(NotImplementedError, match="HLS synthesis evaluation"):
        task.run(ExecutionContext(base_work_dir=tmp_path))
