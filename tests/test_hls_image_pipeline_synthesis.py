from pathlib import Path

from genio import ExecutionContext
from genio import HLSExecutionPackage
from genio import HLSImagePipelineComposer
from genio import HLSReportArtifact
from genio import HLSRTLArtifact
from genio import HLSImagePipelineSynthesisConfigurationError
from genio import HLSImagePipelineSynthesisEvaluationStep
from genio import HLSImagePipelineSynthesisTask
from genio import SearchScenarioSpec
from genio import SearchSpace
from genio import SlotSpec
from genio import StageChoice

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS_PATH = ROOT / "search_space/stages/definitions"
HLS_TEMPLATES_PATH = ROOT / "hls_templates/vitis_vision_image_pipeline"
VITIS_LIBRARIES_PATH = ROOT / "Vitis_Libraries"


class DummyComposer:
    def compose(self, individual):
        raise AssertionError("compose should not be reached by validation tests")


class PackageComposer:
    def compose(self, individual):
        return HLSExecutionPackage(
            files={
                "hls_config.cfg": "part=original-part\n\n[hls]\nclock=10\nflow_target=vivado\nsyn.file=old.cpp\nsyn.top=old_top\n",
                "src/pipeline.cpp": "void top() {}\n",
            },
            metadata={"source_files": ["src/pipeline.cpp"], "top_function": "top"},
        )


class WrongPackageComposer:
    def compose(self, individual):
        return {"not": "a package"}


def make_individual():
    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="hls_image_pipeline_synthesis_test",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(StageChoice(slot=0, stage="nop"),),
                ),
            ),
        )
    )
    return search_space.from_index(0)


def hls_report_artifact(artifacts):
    return next(artifact for artifact in artifacts if isinstance(artifact, HLSReportArtifact))


def hls_rtl_artifact(artifacts):
    return next(artifact for artifact in artifacts if isinstance(artifact, HLSRTLArtifact))


def test_hls_image_pipeline_synthesis_step_creates_task() -> None:
    individual = make_individual()
    step = HLSImagePipelineSynthesisEvaluationStep(
        hls_config=Path("config/hls_config.cfg"),
        work_dir_name="hls_work",
        top_function="pipeline_top",
        clock_period=5.0,
        part="xa7a100tcsg324-1I",
        config_defaults={"hls.clock": "5", "hls.flow_target": "vivado"},
        config_overrides={"hls.flow_target": "vivado"},
    )

    task = step.create_task(individual, artifacts={})

    assert step.id == "hls_image_pipeline_synthesis"
    assert isinstance(task, HLSImagePipelineSynthesisTask)
    assert task.step_id == step.id
    assert task.hls_tool == "v++"
    assert task.hls_config == Path("config/hls_config.cfg")
    assert task.work_dir_name == "hls_work"
    assert task.top_function == "pipeline_top"
    assert task.clock_period == 5.0
    assert task.part == "xa7a100tcsg324-1I"
    assert task.config_defaults == {"hls.clock": "5", "hls.flow_target": "vivado"}
    assert task.config_overrides == {"hls.flow_target": "vivado"}


def test_hls_image_pipeline_synthesis_task_requires_composer(tmp_path) -> None:
    task = HLSImagePipelineSynthesisTask(individual=make_individual())

    with pytest.raises(
        HLSImagePipelineSynthesisConfigurationError,
        match="requires a composer",
    ):
        task.run(ExecutionContext(base_work_dir=tmp_path))


def test_hls_image_pipeline_synthesis_task_validates_hls_config_path(tmp_path) -> None:
    task = HLSImagePipelineSynthesisTask(
        individual=make_individual(),
        composer=DummyComposer(),
        hls_config=Path("missing.cfg"),
    )

    with pytest.raises(HLSImagePipelineSynthesisConfigurationError, match="does not exist"):
        task.run(ExecutionContext(base_work_dir=tmp_path))


def test_hls_image_pipeline_synthesis_task_parses_report_from_config(tmp_path) -> None:
    config = tmp_path / "config.cfg"
    config.write_text("part=xa7a100tcsg324-1I\n\n[hls]\nclock=5\nflow_target=vivado\n", encoding="utf-8")
    task = HLSImagePipelineSynthesisTask(
        individual=make_individual(),
        composer=PackageComposer(),
        hls_config=config,
    )

    artifacts = task.run(ExecutionContext(base_work_dir=tmp_path))
    report = hls_report_artifact(artifacts)
    rtl = hls_rtl_artifact(artifacts)

    assert len(artifacts) == 2
    assert report.origin == "hls_synthesis"
    assert report.metadata["top_function"] == "top"
    assert report.metrics()["hls_synthesis.lut"] >= 0.0
    assert report.metrics()["hls_synthesis.target_clock_period_ns"] == 5.0
    assert rtl.origin == "hls_synthesis"
    assert rtl.top_function == "top"
    assert rtl.rtl_paths
    assert all(path.exists() for path in rtl.load())


def test_hls_image_pipeline_synthesis_task_materializes_package(tmp_path) -> None:
    task = HLSImagePipelineSynthesisTask(
        individual=make_individual(),
        step_id="hls_image_pipeline_synthesis",
        composer=PackageComposer(),
        part="xa7a100tcsg324-1I",
    )

    artifacts = task.run(ExecutionContext(base_work_dir=tmp_path))
    report = hls_report_artifact(artifacts)
    rtl = hls_rtl_artifact(artifacts)

    package_dir = tmp_path / task.individual.id / "hls_image_pipeline_synthesis" / "package"
    assert "[hls]" in (package_dir / "hls_config.cfg").read_text(encoding="utf-8")
    assert (package_dir / "src/pipeline.cpp").read_text(encoding="utf-8") == "void top() {}\n"
    metadata = (package_dir / "composition_metadata.json").read_text(encoding="utf-8")
    assert '"entrypoint": "hls_config.cfg"' in metadata
    assert '"top_function": "top"' in metadata
    logs_dir = tmp_path / task.individual.id / "hls_image_pipeline_synthesis" / "logs"
    stdout = (logs_dir / "hls_stdout.log").read_text(encoding="utf-8")
    stderr = (logs_dir / "hls_stderr.log").read_text(encoding="utf-8")
    assert "v++" in stdout
    assert "HLS Build" in stdout
    assert stderr == ""
    assert (package_dir / "work").is_dir()
    run_metadata = (
        tmp_path
        / task.individual.id
        / "hls_image_pipeline_synthesis"
        / "artifacts/hls_run_metadata.json"
    ).read_text(encoding="utf-8")
    assert '"v++"' in run_metadata
    assert '"returncode": 0' in run_metadata
    assert "hls_compile.log" in run_metadata
    assert report.origin == "hls_synthesis"
    assert "top_csynth.xml" in {path.name for path in report.report_paths}
    assert report.metrics()["hls_synthesis.ff"] >= 0.0
    assert "top.v" in {path.name for path in rtl.verilog_paths}
    assert "top.vhd" in {path.name for path in rtl.vhdl_paths}


def test_hls_image_pipeline_synthesis_task_preserves_logs_on_failure(tmp_path) -> None:
    task = HLSImagePipelineSynthesisTask(
        individual=make_individual(),
        step_id="hls_image_pipeline_synthesis",
        composer=PackageComposer(),
    )

    with pytest.raises(RuntimeError, match="hls_compile.log"):
        task.run(ExecutionContext(base_work_dir=tmp_path))

    task_dir = tmp_path / task.individual.id / "hls_image_pipeline_synthesis"
    assert "HLS Build" in (task_dir / "logs/hls_stdout.log").read_text(encoding="utf-8")
    metadata = (task_dir / "artifacts/hls_run_metadata.json").read_text(encoding="utf-8")
    assert '"v++"' in metadata
    assert '"returncode": 1' in metadata
    assert "hls_compile.log" in metadata


def test_hls_image_pipeline_synthesis_task_requires_hls_package(tmp_path) -> None:
    task = HLSImagePipelineSynthesisTask(
        individual=make_individual(),
        composer=WrongPackageComposer(),
    )

    with pytest.raises(TypeError, match="HLSExecutionPackage"):
        task.run(ExecutionContext(base_work_dir=tmp_path))


def test_hls_image_pipeline_synthesis_task_prepares_final_config(tmp_path) -> None:
    search_space = SearchSpace.from_scenario(
        SearchScenarioSpec(
            id="hls_image_pipeline_synthesis_test",
            slots=(
                SlotSpec(
                    index=0,
                    alternatives=(StageChoice(slot=0, stage="nop"),),
                ),
            ),
            design_spaces={
                "hls": {
                    "clock": (5,),
                    "flow_target": ("vivado",),
                    "npc": ("XF_NPPC2",),
                    "pipeline_ii": (1,),
                    "use_uram": (True,),
                }
            },
        )
    )
    individual = search_space.from_index(0)
    task = HLSImagePipelineSynthesisTask(
        individual=individual,
        step_id="hls_image_pipeline_synthesis",
        composer=PackageComposer(),
        part="xa7a100tcsg324-1I",
        clock_period=7.5,
        config_defaults={"hls.flow_target": "vitis"},
        config_overrides={"hls.package.output.syn": "false"},
    )

    artifacts = task.run(ExecutionContext(base_work_dir=tmp_path))
    report = hls_report_artifact(artifacts)

    config_path = (
        tmp_path
        / individual.id
        / "hls_image_pipeline_synthesis"
        / "package"
        / "hls_config.cfg"
    )
    config = config_path.read_text(encoding="utf-8")

    assert "part=xa7a100tcsg324-1I" in config
    assert "clock=7.5" in config
    assert "flow_target=vivado" in config
    assert "syn.file=src/pipeline.cpp" in config
    assert "syn.top=top" in config
    assert "package.output.syn=false" in config
    assert "npc=" not in config
    assert "pipeline_ii=" not in config
    assert "use_uram=" not in config
    assert (config_path.parent / "hls_config_metadata.json").exists()
    assert report.metrics()["hls_synthesis.target_clock_period_ns"] == 7.5
    assert report.metrics()["hls_synthesis.estimated_clock_period_ns"] >= 0.0


def test_hls_image_pipeline_synthesis_requires_vitis_libraries(tmp_path) -> None:
    search_space = SearchSpace(
        ROOT / "search_space/tests/insect_segmentation_pipeline.json",
        DEFINITIONS_PATH,
    )
    individual = search_space.from_index(0)
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=HLS_TEMPLATES_PATH,
        rows=2160,
        cols=3840,
    )
    task = HLSImagePipelineSynthesisTask(
        individual=individual,
        step_id="hls_image_pipeline_synthesis",
        composer=composer,
    )

    with pytest.raises(
        HLSImagePipelineSynthesisConfigurationError,
        match="vitis_libraries_path",
    ):
        task.run(ExecutionContext(base_work_dir=tmp_path))


def test_hls_image_pipeline_synthesis_with_insect_pipeline(tmp_path) -> None:
    search_space = SearchSpace(
        ROOT / "search_space/tests/insect_segmentation_pipeline.json",
        DEFINITIONS_PATH,
    )
    individual = search_space.from_index(0)
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=HLS_TEMPLATES_PATH,
        rows=2160,
        cols=3840,
    )
    task = HLSImagePipelineSynthesisTask(
        individual=individual,
        step_id="hls_image_pipeline_synthesis",
        composer=composer,
        part="xa7a100tcsg324-1I",
    )

    context = ExecutionContext(
        base_work_dir=tmp_path,
        metadata={"vitis_libraries_path": str(VITIS_LIBRARIES_PATH)},
    )

    artifacts = task.run(context)
    report = hls_report_artifact(artifacts)
    rtl = hls_rtl_artifact(artifacts)

    task_dir = tmp_path / individual.id / "hls_image_pipeline_synthesis"
    package_dir = task_dir / "package"
    source = (package_dir / "src/pipeline.cpp").read_text(encoding="utf-8")
    config = (package_dir / "hls_config.cfg").read_text(encoding="utf-8")
    stdout = (task_dir / "logs" / "hls_stdout.log").read_text(encoding="utf-8")
    run_metadata = (task_dir / "artifacts" / "hls_run_metadata.json").read_text(
        encoding="utf-8"
    )
    report_metadata = (task_dir / "artifacts" / "hls_report_hls_synthesis.json").read_text(
        encoding="utf-8"
    )
    rtl_metadata = (task_dir / "artifacts" / "hls_rtl_hls_synthesis.json").read_text(
        encoding="utf-8"
    )

    assert "xf::cv::resize" in source
    assert "xf::cv::bgr2gray" in source
    assert "xf::cv::equalizeHist" in source
    assert "xf::cv::Threshold" in source
    assert "XF_INTERPOLATION_AREA" in source
    assert "XF_NPPC1" in source
    assert "@" not in source
    assert "part=xa7a100tcsg324-1I" in config
    assert "clock=5" in config
    assert "flow_target=vivado" in config
    assert "syn.file=src/pipeline.cpp" in config
    assert "syn.top=top" in config
    assert f"syn.cflags=-Iinclude -I{VITIS_LIBRARIES_PATH / 'vision/L1/include'}" in config
    assert "pipeline_ii=" not in config
    assert "use_uram=" not in config
    assert (package_dir / "composition_metadata.json").exists()
    assert (package_dir / "hls_config_metadata.json").exists()
    assert "HLS Build" in stdout
    assert '"v++"' in run_metadata
    assert '"returncode": 0' in run_metadata
    assert "hls_compile.log" in run_metadata
    assert report.origin == "hls_synthesis"
    assert report.metrics()["hls_synthesis.lut"] > 0.0
    assert report.metrics()["hls_synthesis.ff"] > 0.0
    assert report.metrics()["hls_synthesis.latency_max_cycles"] > 0.0
    assert report.metadata["flow_target"] == "vivado"
    assert '"origin": "hls_synthesis"' in report_metadata
    assert "top_csynth.xml" in report_metadata
    assert len(artifacts) == 2
    assert rtl.origin == "hls_synthesis"
    assert rtl.top_function == "top"
    assert "top.v" in {path.name for path in rtl.verilog_paths}
    assert "top.vhd" in {path.name for path in rtl.vhdl_paths}
    assert len(rtl.verilog_paths) > 1
    assert len(rtl.vhdl_paths) > 1
    assert all(path.exists() for path in rtl.rtl_paths)
    assert '"top_function": "top"' in rtl_metadata
    assert "top.v" in rtl_metadata
    assert "top.vhd" in rtl_metadata
