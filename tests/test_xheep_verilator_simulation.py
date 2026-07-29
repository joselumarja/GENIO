from pathlib import Path

import pytest

from genio import (
    EvaluationWorkflow,
    EvaluationWorkflowError,
    GRHeepConfigurationComposer,
    HLSImagePipelineSynthesisEvaluationStep,
    HLSRTLArtifact,
    Individual,
    StageChoice,
    XHeepVerilatorSimulationEvaluationStep,
    XHeepVerilatorSimulationResultError,
    XHeepVerilatorSimulationTask,
)
from genio.evaluation.task import ExecutionContext


ROOT = Path(__file__).resolve().parents[1]


def make_conda_proxy(tmp_path: Path) -> Path:
    proxy = tmp_path / "conda-proxy"
    proxy.write_text(
        "#!/bin/sh\n"
        "shift 4\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    proxy.chmod(0o755)
    return proxy


def make_artifact(tmp_path: Path) -> HLSRTLArtifact:
    rtl = tmp_path / "safa_accelerator.v"
    rtl.write_text("module safa_accelerator; endmodule\n", encoding="utf-8")
    return HLSRTLArtifact(
        name="rtl_hls_synthesis",
        producer="hls_image_pipeline_synthesis",
        individual_id="individual",
        origin="hls_synthesis",
        top_function="safa_accelerator",
        verilog_paths=(rtl,),
        metadata={
            "interface": "safa_fifo",
            "input_rows": 48,
            "input_cols": 48,
            "output_rows": 48,
            "output_cols": 48,
            "input_words": 1728,
            "output_words": 576,
        },
    )


def make_composer() -> GRHeepConfigurationComposer:
    return GRHeepConfigurationComposer(
        ROOT / "search_space/stages/definitions",
        templates_path=ROOT / "gr_heep_templates",
    )


def test_xheep_step_requires_hls_rtl_artifact() -> None:
    step = XHeepVerilatorSimulationEvaluationStep()

    assert tuple(step.required_artifacts) == (
        "hls_image_pipeline_synthesis.rtl_hls_synthesis",
    )


def test_xheep_workflow_requires_hls_step_dependency() -> None:
    with pytest.raises(EvaluationWorkflowError, match="not a declared dependency"):
        EvaluationWorkflow((XHeepVerilatorSimulationEvaluationStep(),))


def test_xheep_workflow_orders_hls_before_simulation() -> None:
    hls_step = HLSImagePipelineSynthesisEvaluationStep()
    xheep_step = XHeepVerilatorSimulationEvaluationStep(
        depends_on=(hls_step.id,)
    )

    workflow = EvaluationWorkflow((xheep_step, hls_step))

    assert tuple(step.id for step in workflow.execution_order()) == (
        "hls_image_pipeline_synthesis",
        "xheep_verilator_simulation",
    )


def test_xheep_commands_run_in_core_v_mini_mcu_conda_environment(tmp_path) -> None:
    task = XHeepVerilatorSimulationTask(
        individual=Individual.from_slots(
            id="command",
            scenario="scenario",
            design={"system": {}},
            slots=[StageChoice(slot=0, stage="nop")],
        ),
        composer=make_composer(),
        hls_artifact=make_artifact(tmp_path),
    )

    assert task._command(("make", "mcu-gen")) == (
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "core-v-mini-mcu",
        "make",
        "mcu-gen",
    )


def test_xheep_task_injects_rtl_and_parses_simulation_metrics(tmp_path) -> None:
    source = tmp_path / "GEN-HEEP"
    source.mkdir()
    (source / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    context = ExecutionContext(base_work_dir=tmp_path / "work")
    individual = Individual.from_slots(
        id="individual",
        scenario="scenario",
        design={"system": {}},
        slots=[StageChoice(slot=0, stage="nop")],
    )
    task = XHeepVerilatorSimulationTask(
        individual=individual,
        step_id="xheep_verilator_simulation",
        composer=make_composer(),
        hls_artifact=make_artifact(tmp_path),
        gr_heep_path=source,
        conda_tool=str(make_conda_proxy(tmp_path)),
        commands=(
            ("true",),
            ("true",),
            (
                "sh",
                "-c",
                "printf 'GENIO_PERF:accelerator:123\\n"
                "GENIO_METRIC:safa_input_stall_cycles:17\\nGENIO_STATUS:0\\n"
                "Simulation finished after 456 clock cycles\\n'",
            ),
        ),
    )
    artifacts = task.run(context)

    checkout = context.task_dir(task, "xheep")
    assert (checkout / "hw/vendor/safa/hls_ip/safa_accelerator.v").is_file()
    core = (
        checkout / "hw/vendor/safa/hls_ip/hls_accelerator_component.core"
    ).read_text(encoding="utf-8")
    wrapper = (checkout / "hw/vendor/safa/rtl/safa_wrapper.sv").read_text(
        encoding="utf-8"
    )
    assert "- safa_accelerator.v" in core
    assert "toplevel: safa_accelerator" in core
    assert "safa_accelerator i_hls_top" in wrapper
    assert "parameter int unsigned FIFO_DEPTH = 16" in wrapper
    assert (checkout / "config/mcu-gen-config.py").is_file()
    assert artifacts[0].metrics() == {
        "xheep_verilator.accelerator_cycles": 123.0,
        "xheep_verilator.safa_input_stall_cycles": 17.0,
        "xheep_verilator.status": 0.0,
        "xheep_verilator.simulation_cycles": 456.0,
    }


def test_xheep_step_propagates_input_image_path(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"sample")
    artifact = make_artifact(tmp_path)
    step = XHeepVerilatorSimulationEvaluationStep(
        depends_on=("hls_image_pipeline_synthesis",),
        composer=make_composer(),
        input_image_path=image_path,
    )
    individual = Individual.from_slots(
        id="individual",
        scenario="scenario",
        design={"system": {}},
        slots=[StageChoice(slot=0, stage="nop")],
    )

    task = step.create_task(
        individual,
        {"hls_image_pipeline_synthesis.rtl_hls_synthesis": artifact},
    )

    assert task.input_image_path == image_path
    assert step.checkpoint_signature()["input_image_path"] == image_path


@pytest.mark.parametrize(
    ("output", "message"),
    (
        ("Simulation finished after 10 clock cycles\n", "did not emit"),
        ("GENIO_STATUS:1\n", "reported GENIO_STATUS:1"),
        (
            "GENIO_STATUS:0\nProgram Finished with value 1\n",
            "finished with value 1",
        ),
    ),
)
def test_xheep_task_rejects_unsuccessful_firmware_status(
    tmp_path,
    output,
    message,
) -> None:
    source = tmp_path / "GEN-HEEP"
    source.mkdir()
    (source / "Makefile").write_text("all:\n\t@true\n", encoding="utf-8")
    task = XHeepVerilatorSimulationTask(
        individual=Individual.from_slots(
            id="invalid_status",
            scenario="scenario",
            design={"system": {}},
            slots=[StageChoice(slot=0, stage="nop")],
        ),
        step_id="xheep_verilator_simulation",
        composer=make_composer(),
        hls_artifact=make_artifact(tmp_path),
        gr_heep_path=source,
        conda_tool=str(make_conda_proxy(tmp_path)),
        commands=(("sh", "-c", f"printf '{output}'"),),
    )

    with pytest.raises(XHeepVerilatorSimulationResultError, match=message):
        task.run(ExecutionContext(base_work_dir=tmp_path / "work"))
