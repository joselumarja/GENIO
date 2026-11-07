"""X-HEEP execution backends, logs, and optional real-toolchain smoke test."""

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from genio.core.individual import Individual
from genio.evaluation.executor import EvaluationExecutor
from genio.evaluation.step import EvaluationStep
from genio.evaluation.task import EvaluationOutput, TaskContext, _TaskTimeoutInterrupted
from genio.evaluation.workflow import EvaluationWorkflow
from genio_use_cases.image_pipeline.hls_synthesis import HLSRTLArtifact
from genio_use_cases.xheep.gr_heep import GRHeepConfigurationComposer
from genio_use_cases.xheep.xheep_verilator_simulation import (
    XHeepVerilatorSimulationConfigurationError,
    XHeepVerilatorSimulationError,
    XHeepVerilatorSimulationTask,
)


IMAGE = "ghcr.io/x-heep/x-heep/x-heep-toolchain:v1.0.5"


class _DockerCommandsTask(XHeepVerilatorSimulationTask):
    requires = ()

    def run(self, context):
        result = self._run_commands(context, self.gr_heep_path)
        return EvaluationOutput(metrics=self._simulation_metrics(context, result))


@pytest.fixture
def task(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "Makefile").write_text("all:\n\ttrue\n")
    rtl = tmp_path / "pipeline.v"
    rtl.write_text("module pipeline; endmodule\n")
    return XHeepVerilatorSimulationTask(
        individual=Individual("candidate", "test"),
        composer=GRHeepConfigurationComposer(tmp_path, application_name="genio_test"),
        gr_heep_path=source,
        artifacts={
            "hls_image_pipeline_synthesis.rtl_hls_synthesis": HLSRTLArtifact(
                data={"interface": "safa_fifo", "top_function": "pipeline"},
                files={"verilog/pipeline.v": rtl},
            ),
        },
    )


def test_local_conda_command_unchanged(task):
    assert task._command(("make", "app", "PROJECT={application_name}")) == (
        "conda", "run", "--no-capture-output", "-n", "core-v-mini-mcu",
        "make", "app", "PROJECT=genio_test",
    )


def test_copy_excludes_nested_kilo_worktrees(task, tmp_path):
    nested = task.gr_heep_path / ".kilo/worktrees/other/hw"
    nested.mkdir(parents=True)
    (nested / "unrelated.tpl").write_text("${undefined_variable}")
    vendor = task.gr_heep_path / "hw/vendor"
    vendor.mkdir(parents=True)
    (vendor / ".kilo").mkdir()
    (vendor / "module.tpl").write_text("module example; endmodule")
    (task.gr_heep_path / "sw").mkdir()
    (task.gr_heep_path / "sw/build").symlink_to("../hw/vendor", target_is_directory=True)
    checkout = task._copy_checkout(TaskContext(tmp_path / "evaluation"), task.gr_heep_path)
    assert not (checkout / ".kilo").exists()
    assert not (checkout / "hw/vendor/.kilo").exists()
    assert (checkout / "hw/vendor/module.tpl").is_file()
    assert (checkout / "sw/build").is_symlink()
    assert (checkout / "sw/build/module.tpl").is_file()
    assert (nested / "unrelated.tpl").is_file()


def test_docker_command_and_step_configuration(task, tmp_path):
    step = EvaluationStep(
        "simulation", XHeepVerilatorSimulationTask,
        config={"docker_image": IMAGE, "composer": task.composer, "make_tool": "gmake"},
    )
    configured = step.create_task(task.individual, task.artifacts)
    checkout = tmp_path / "checkout with spaces"
    command = configured._command(
        ("make", "app", "PROJECT={application_name}", "literal=$(false); a b"),
        checkout, "test-container",
    )
    assert command[:3] == ("docker", "run", "--rm")
    assert command[command.index("--volume") + 1] == f"{checkout}:/workspace/x-heep"
    assert command[command.index("--user") + 1] == "0:0"
    assert command[command.index("--workdir") + 1] == "/workspace/x-heep"
    assert "LC_ALL=C" in command
    assert not any(argument.startswith("HOME=") for argument in command)
    assert command[command.index(IMAGE) + 1:command.index(IMAGE) + 3] == ("/bin/bash", "-lc")
    script = command[command.index(IMAGE) + 3]
    assert "trap " in script
    assert "chown -hR" in script
    assert command[command.index(IMAGE) + 4:] == (
        "genio-xheep", f"{os.getuid()}:{os.getgid()}", "gmake", "app",
        "PROJECT=genio_test", "literal=$(false); a b",
    )


@pytest.mark.parametrize("image", ["", " ", "bad image", "--privileged", 123])
def test_invalid_docker_image(task, tmp_path, image):
    with pytest.raises(XHeepVerilatorSimulationConfigurationError, match="docker_image"):
        replace(task, docker_image=image)._validate_configuration(TaskContext(tmp_path))


def test_docker_does_not_require_host_conda(task, tmp_path):
    configured = replace(task, docker_image=IMAGE, conda_tool="", conda_environment="")
    assert configured._validate_configuration(TaskContext(tmp_path)) == task.gr_heep_path
    with pytest.raises(XHeepVerilatorSimulationConfigurationError, match="conda"):
        replace(configured, docker_image=None)._validate_configuration(TaskContext(tmp_path))


@pytest.mark.parametrize("returncode", [0, 7])
def test_docker_logs_and_metadata(task, tmp_path, monkeypatch, returncode):
    configured = replace(task, docker_image=IMAGE, commands=(("make", "app"),))

    def run(command, **kwargs):
        assert configured.runtime_state["container_name"] in command
        pending = json.loads((tmp_path / "artifacts/xheep_run_metadata.json").read_text())
        assert pending["status"] == "running"
        assert pending["command"] == list(command)
        assert pending["container_name"] == configured.runtime_state["container_name"]
        kwargs["stdout"].write("GENIO_STATUS:0\nGENIO_PERF:application:42\n")
        kwargs["stderr"].write("diagnostic\n")
        return subprocess.CompletedProcess(command, returncode)

    runner = Mock(side_effect=run)
    monkeypatch.setattr(subprocess, "run", runner)
    context = TaskContext(tmp_path)
    if returncode:
        with pytest.raises(XHeepVerilatorSimulationError) as error:
            configured._run_commands(context, task.gr_heep_path)
        assert error.value.returncode == returncode
        assert error.value.log_paths[-1].read_text() == "diagnostic\n"
    else:
        result = configured._run_commands(context, task.gr_heep_path)
        assert configured._simulation_metrics(context, result)[
            "xheep_verilator.application_cycles"
        ] == 42
    metadata = json.loads((tmp_path / "artifacts/xheep_run_metadata.json").read_text())
    recorded = metadata["command"] if returncode else metadata["commands"][0]
    assert recorded == list(runner.call_args.args[0])
    assert "container_name" not in configured.runtime_state


def test_timeout_removes_active_container(task, tmp_path, monkeypatch):
    configured = replace(task, docker_image=IMAGE, commands=(("make", "app"),))
    runner = Mock(side_effect=[
        _TaskTimeoutInterrupted(), subprocess.CompletedProcess([], 0),
        subprocess.CompletedProcess([], 0),
    ])
    monkeypatch.setattr(subprocess, "run", runner)
    monkeypatch.setattr(
        XHeepVerilatorSimulationTask, "run",
        lambda self, context: self._run_commands(context, task.gr_heep_path),
    )
    with pytest.raises(_TaskTimeoutInterrupted):
        configured._execute(TaskContext(tmp_path))
    container = runner.call_args_list[0].args[0]
    name = container[container.index("--name") + 1]
    assert runner.call_args_list[1].args[0] == ("docker", "rm", "--force", name)
    ownership = runner.call_args_list[2].args[0]
    assert ownership[-4:] == ("-hR", "--", f"{os.getuid()}:{os.getgid()}", "/workspace/x-heep")
    assert runner.call_args_list[2].kwargs["check"] is True
    assert "container_name" not in configured.runtime_state
    assert "checkout_dir" not in configured.runtime_state


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("GENIO_TEST_XHEEP_DOCKER_IMAGE"),
    reason="Set GENIO_TEST_XHEEP_DOCKER_IMAGE to run real Docker tools",
)
@pytest.mark.parametrize("supervised", [False, True])
def test_real_docker_toolchain(task, tmp_path, supervised):
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    (checkout / "Makefile").write_text(
        "build:\n"
        "\tverilator --version\n"
        "\tverible-verilog-format --version\n"
        "\ttest \"$$(id -u)\" = 0\n"
        "\tfusesoc --version\n"
        "\triscv32-unknown-elf-gcc -c main.c -o main.o\n"
        "\tpython -c 'import pathlib; pathlib.Path(\"built\").write_text(\"ok\")'\n"
        "\tmkdir -p generated/private && chmod 700 generated/private\n"
        "\ttouch generated/private/output.core\n"
        "check:\n"
        "\ttest -s main.o && test -f built\n"
        "\ttest '$(TOKEN)' = 'literal;$$(false) with spaces'\n"
        "\t@echo GENIO_STATUS:0\n"
        "\t@echo GENIO_PERF:application:42\n"
    )
    (checkout / "main.c").write_text("int main(void) { return 0; }\n")
    configured = replace(
        task, docker_image=os.environ["GENIO_TEST_XHEEP_DOCKER_IMAGE"],
        commands=(("make", "build"), ("make", "check", "TOKEN=literal;$$(false) with spaces")),
    )
    context = TaskContext(tmp_path)
    if supervised:
        step = EvaluationStep(
            "docker_probe", _DockerCommandsTask,
            config={
                "composer": task.composer,
                "gr_heep_path": checkout,
                "docker_image": configured.docker_image,
                "commands": configured.commands,
            },
            timeout_seconds=30,
        )
        executor = EvaluationExecutor(
            EvaluationWorkflow((step,)), work_dir=tmp_path / "executor", run_id="probe",
        )
        result = executor.evaluate(task.individual)
        assert result.metrics["docker_probe.xheep_verilator.application_cycles"] == 42, result
        assert (checkout / "main.o").stat().st_uid == os.getuid()
        shutil.rmtree(checkout)
        return
    result = configured._run_commands(context, checkout)
    assert configured._simulation_metrics(context, result)[
        "xheep_verilator.application_cycles"
    ] == 42
    assert (checkout / "main.o").stat().st_uid == os.getuid()
    assert (checkout / "generated/private/output.core").read_text() == ""
    shutil.rmtree(checkout)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("GENIO_TEST_XHEEP_DOCKER_IMAGE"),
    reason="Set GENIO_TEST_XHEEP_DOCKER_IMAGE to run real Docker tools",
)
def test_real_docker_failure_restores_ownership(task, tmp_path):
    checkout = tmp_path / "failed-checkout"
    checkout.mkdir()
    external = tmp_path / "outside"
    external.write_text("unchanged")
    (checkout / "outside-link").symlink_to(external)
    configured = replace(
        task, docker_image=os.environ["GENIO_TEST_XHEEP_DOCKER_IMAGE"],
        commands=(("/bin/bash", "-c", "mkdir -m 700 generated; touch generated/output.core; exit 7"),),
    )
    with pytest.raises(XHeepVerilatorSimulationError) as error:
        configured._run_commands(TaskContext(tmp_path), checkout)
    assert error.value.returncode == 7
    assert (checkout / "generated/output.core").stat().st_uid == os.getuid()
    assert external.read_text() == "unchanged"
    shutil.rmtree(checkout)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("GENIO_TEST_XHEEP_DOCKER_IMAGE"),
    reason="Set GENIO_TEST_XHEEP_DOCKER_IMAGE to run real Docker tools",
)
def test_real_docker_concurrent_workers(task, tmp_path):
    step = EvaluationStep(
        "docker_probe", _DockerCommandsTask,
        config={
            "composer": task.composer,
            "gr_heep_path": task.gr_heep_path,
            "docker_image": os.environ["GENIO_TEST_XHEEP_DOCKER_IMAGE"],
            "commands": (("printf", "GENIO_STATUS:0\\n"),),
        },
        timeout_seconds=60,
    )
    executor = EvaluationExecutor(
        EvaluationWorkflow((step,)), max_workers=16,
        work_dir=tmp_path / "concurrent", run_id="probe",
    )
    results = executor.evaluate_many([Individual(f"candidate-{i}", "test") for i in range(16)])
    assert all(result.metrics.get("docker_probe.xheep_verilator.status") == 0 for result in results), results


@pytest.mark.slow
@pytest.mark.skipif(
    not (os.environ.get("GENIO_TEST_XHEEP_DOCKER_IMAGE") and os.environ.get("GENIO_TEST_GR_HEEP_PATH")),
    reason="Set GENIO_TEST_XHEEP_DOCKER_IMAGE and GENIO_TEST_GR_HEEP_PATH for real MCU generation",
)
def test_real_docker_mcu_generation(task, tmp_path):
    configured = replace(
        task, docker_image=os.environ["GENIO_TEST_XHEEP_DOCKER_IMAGE"],
        commands=(("make", "mcu-gen"),),
    )
    context = TaskContext(tmp_path)
    checkout = configured._copy_checkout(context, Path(os.environ["GENIO_TEST_GR_HEEP_PATH"]))
    result = configured._run_commands(context, checkout)
    assert "X-HEEP configuration generated successfully" in result.output
    assert ".kilo/worktrees" not in result.output
