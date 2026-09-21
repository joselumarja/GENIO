from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from genio.artifacts import Artifact, HLSRTLArtifact, XHeepSimulationArtifact
from genio.composer import GRHeepConfigurationComposer, GRHeepConfigurationPackage
from genio.core.individual import Individual
from genio.evaluation.step import EvaluationStep
from genio.evaluation.task import EvaluationTask, ExecutionContext


class XHeepVerilatorSimulationConfigurationError(ValueError):
    """Indicate invalid X-HEEP paths, commands, composer, or RTL inputs."""


class XHeepVerilatorSimulationError(RuntimeError):
    """Report a nonzero exit from an X-HEEP build or simulation command.

    Attributes:
        command: Fully rendered command, including the conda wrapper.
        returncode: Exit status returned by the command.
        log_paths: Stdout and stderr logs accumulated through the failed command.
    """

    def __init__(
        self,
        command: tuple[str, ...],
        returncode: int,
        log_paths: tuple[Path, ...] = (),
    ) -> None:
        self.command = command
        self.returncode = returncode
        self.log_paths = log_paths
        hint = f" See {log_paths[-1]}." if log_paths else ""
        super().__init__(
            f"X-HEEP command {command!r} failed with return code {returncode}.{hint}"
        )


class XHeepVerilatorSimulationTimeoutError(TimeoutError):
    """Indicate that the aggregate X-HEEP command sequence exceeded its timeout."""


class XHeepVerilatorSimulationResultError(RuntimeError):
    """Indicate a missing or failing firmware-level GENIO result status."""


@dataclass(frozen=True, slots=True)
class _XHeepRunResult:
    """Collect the isolated checkout, command logs, and merged process output.

    Attributes:
        checkout_dir: Private GR-HEEP copy modified for this evaluation task.
        log_paths: Ordered stdout/stderr log paths for executed commands.
        output: Concatenated command streams parsed for firmware metrics.
    """

    checkout_dir: Path
    log_paths: tuple[Path, ...]
    output: str


@dataclass(frozen=True, slots=True)
class XHeepVerilatorSimulationTask(EvaluationTask):
    """Integrate SAFA-compatible HLS RTL into GR-HEEP and run Verilator.

    A :class:`GRHeepConfigurationComposer` renders the system configuration and
    overlays the upstream HLS RTL, optional image data, firmware, and integration
    files into an isolated copy of the GR-HEEP source tree. Each configured build
    command is wrapped with ``conda run --no-capture-output`` and shares one total
    task timeout.

    Firmware communicates results through textual records:
    ``GENIO_PERF:<region>:<cycles>``, ``GENIO_METRIC:<name>:<value>``, and the
    mandatory ``GENIO_STATUS:<value>``. Verilator and program-finish cycle/status
    messages are also recognized. A successful result requires zero GENIO status
    and, when emitted, zero program status.

    Attributes:
        composer: GR-HEEP configuration and overlay composer.
        hls_artifact: Upstream SAFA FIFO RTL artifact to integrate.
        gr_heep_path: Source GR-HEEP checkout copied for the task.
        input_image_path: Optional image embedded by the generated overlay.
        application_name: Read-only firmware project name obtained from ``composer``.
        conda_environment: Environment used for all build and simulation commands.
        conda_tool: Conda-compatible executable.
        make_tool: Replacement executable for commands beginning with ``make``.
        commands: Ordered command templates executed in the isolated checkout.
        metadata: Additional execution metadata and upstream artifact names.
    """

    composer: GRHeepConfigurationComposer | None = None
    hls_artifact: HLSRTLArtifact | None = None
    gr_heep_path: Path | None = None
    input_image_path: Path | None = None
    conda_environment: str = "core-v-mini-mcu"
    conda_tool: str = "conda"
    make_tool: str = "make"
    commands: tuple[tuple[str, ...], ...] = (
        ("make", "mcu-gen"),
        ("make", "verilator-build"),
        ("make", "app", "PROJECT={application_name}"),
        ("make", "verilator-run"),
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def application_name(self) -> str:
        """Return the firmware project selected by the GR-HEEP composer."""

        return self._require_gr_heep_composer().application_name

    def cache_inputs(self) -> Mapping[str, Any]:
        """Cache by pipeline, X-HEEP design and RTL artifact identity."""

        assert self.hls_artifact is not None
        return {
            "pipeline": self._pipeline_cache_inputs(),
            "hls_design": self.individual.design.get("hls", {}),
            "system_design": self.individual.design.get("system", {}),
            "hls_rtl": {
                "origin": self.hls_artifact.origin,
                "top_function": self.hls_artifact.top_function,
                "verilog": tuple(
                    (path.name, self._sha256(path))
                    for path in self.hls_artifact.verilog_paths
                ),
                "vhdl": tuple(
                    (path.name, self._sha256(path))
                    for path in self.hls_artifact.vhdl_paths
                ),
            },
            "composer": self._require_gr_heep_composer().checkpoint_signature(),
            "input_image": (
                (self.input_image_path.name, self._sha256(self.input_image_path))
                if self.input_image_path is not None
                else None
            ),
            "application_name": self.application_name,
            "conda_environment": self.conda_environment,
            "conda_tool": self.conda_tool,
            "make_tool": self.make_tool,
            "commands": self.commands,
        }

    def run(self, context: ExecutionContext) -> list[Artifact]:
        """Build and simulate an isolated GR-HEEP overlay for the candidate.

        Args:
            context: Execution services for resource resolution, copying, command
                invocation, logging, and artifact persistence.

        Returns:
            A single :class:`XHeepSimulationArtifact` containing parsed metrics and
            links to every command log.

        Raises:
            XHeepVerilatorSimulationConfigurationError: If inputs, commands, or
                the GR-HEEP/composer contract are invalid.
            TypeError: If composition returns an unexpected package type.
            XHeepVerilatorSimulationTimeoutError: If the total command deadline
                is exhausted.
            XHeepVerilatorSimulationError: If any command exits unsuccessfully.
            XHeepVerilatorSimulationResultError: If firmware omits GENIO status
                or reports a failing GENIO/program status.
        """

        gr_heep_path = self._validate_configuration(context)
        package = self._complete_package()
        checkout_dir = self._copy_checkout(context, gr_heep_path)
        self._apply_overlay(context, package, checkout_dir)
        run_result = self._run_commands(context, checkout_dir)
        return [self._simulation_artifact(context, run_result)]

    def _validate_configuration(self, context: ExecutionContext) -> Path:
        """Validate the GR-HEEP, composer, RTL, image, and command contracts.

        The RTL artifact must expose at least one existing Verilog file and declare
        the ``safa_fifo`` interface required by the generated GR-HEEP wrapper.

        Returns:
            The resolved GR-HEEP source directory.
        """

        if self.hls_artifact is None:
            raise XHeepVerilatorSimulationConfigurationError(
                "XHeepVerilatorSimulationTask requires an HLS RTL artifact."
            )
        if self.composer is None:
            raise XHeepVerilatorSimulationConfigurationError(
                "XHeepVerilatorSimulationTask requires a GRHeepConfigurationComposer."
            )
        self._require_gr_heep_composer()
        if self.gr_heep_path is None:
            raise XHeepVerilatorSimulationConfigurationError(
                "XHeepVerilatorSimulationTask requires gr_heep_path."
            )
        gr_heep_path = context.resolve_resource_path(self.gr_heep_path)
        if not gr_heep_path.is_dir():
            raise XHeepVerilatorSimulationConfigurationError(
                f"gr_heep_path must be an existing directory: {gr_heep_path}."
            )
        if not (gr_heep_path / "Makefile").is_file():
            raise XHeepVerilatorSimulationConfigurationError(
                f"gr_heep_path does not contain a Makefile: {gr_heep_path}."
            )
        if self.input_image_path is not None and not context.resolve_resource_path(
            self.input_image_path
        ).is_file():
            raise XHeepVerilatorSimulationConfigurationError(
                f"input_image_path must be an existing image: {self.input_image_path}."
            )
        if self.hls_artifact.metadata.get("interface") != "safa_fifo":
            raise XHeepVerilatorSimulationConfigurationError(
                "X-HEEP simulation requires HLS RTL with interface 'safa_fifo'."
            )
        if not self.hls_artifact.verilog_paths:
            raise XHeepVerilatorSimulationConfigurationError(
                "X-HEEP simulation requires Verilog files in the HLS RTL artifact."
            )
        for rtl_path in self.hls_artifact.verilog_paths:
            if not rtl_path.is_file():
                raise XHeepVerilatorSimulationConfigurationError(
                    f"HLS RTL file does not exist: {rtl_path}."
                )
        if not self.commands:
            raise XHeepVerilatorSimulationConfigurationError(
                "X-HEEP simulation requires at least one build command."
            )
        if not self.conda_environment or not self.conda_tool:
            raise XHeepVerilatorSimulationConfigurationError(
                "X-HEEP simulation requires a conda tool and environment."
            )
        for command in self.commands:
            if not command or any(not isinstance(argument, str) or not argument for argument in command):
                raise XHeepVerilatorSimulationConfigurationError(
                    "X-HEEP commands must contain only non-empty string arguments."
                )
        return gr_heep_path

    def _complete_package(self) -> GRHeepConfigurationPackage:
        """Compose system files and merge the HLS RTL/application overlay.

        The returned package records the upstream RTL producer, artifact name, top
        function, and Verilog filenames in metadata for traceability.

        Raises:
            TypeError: If the composer does not return the required GR-HEEP package.
        """

        composer = self._require_gr_heep_composer()
        package = composer.compose(self.individual)
        if not isinstance(package, GRHeepConfigurationPackage):
            raise TypeError(
                "XHeepVerilatorSimulationTask requires a GRHeepConfigurationPackage."
            )
        assert self.hls_artifact is not None
        rtl_overlay = composer.render_hls_artifact_overlay(
            self.hls_artifact,
            image_path=self.input_image_path,
            configuration=package.metadata.get("rendered_configuration", {}),
        )
        return GRHeepConfigurationPackage(
            files={**dict(package.files), **dict(rtl_overlay)},
            metadata={
                **dict(package.metadata),
                "hls_artifact": {
                    "producer": self.hls_artifact.producer,
                    "name": self.hls_artifact.name,
                    "top_function": self.hls_artifact.top_function,
                    "verilog_files": tuple(
                        path.name for path in self.hls_artifact.verilog_paths
                    ),
                },
            },
        )

    def _copy_checkout(self, context: ExecutionContext, source: Path) -> Path:
        """Create a fresh isolated GR-HEEP tree for destructive overlay writes.

        Existing task-local copies are removed. The destination is rejected if it
        resolves inside the source tree, preventing recursive self-copying.
        """

        checkout_dir = context.task_dir(self, "xheep")
        try:
            checkout_dir.resolve().relative_to(source.resolve())
        except ValueError:
            pass
        else:
            raise XHeepVerilatorSimulationConfigurationError(
                "The isolated X-HEEP checkout cannot be created inside gr_heep_path."
            )
        if checkout_dir.exists():
            shutil.rmtree(checkout_dir)
        context.copy_tree(source, checkout_dir, dirs_exist_ok=False, symlinks=True)
        return checkout_dir

    def _apply_overlay(
        self,
        context: ExecutionContext,
        package: GRHeepConfigurationPackage,
        checkout_dir: Path,
    ) -> None:
        """Materialize generated files and persist their composition metadata."""

        package.materialize(checkout_dir)
        context.write_json(
            context.artifact_path(self, "xheep_overlay_metadata.json"),
            dict(package.metadata),
        )

    def _run_commands(
        self,
        context: ExecutionContext,
        checkout_dir: Path,
    ) -> _XHeepRunResult:
        """Execute the configured tool flow under one aggregate deadline.

        Each command receives only the remaining task timeout. Its stdout and
        stderr are persisted separately with an ordered command prefix. Run
        metadata is written for command-level timeout, nonzero exit, and complete
        success; an already exhausted deadline fails before the next command runs.

        Returns:
            The checkout, ordered log paths, and concatenated process output.

        Raises:
            XHeepVerilatorSimulationTimeoutError: If no deadline remains or a
                command times out.
            XHeepVerilatorSimulationError: If a command returns nonzero.
        """

        log_paths: list[Path] = []
        outputs: list[str] = []
        timeout = self.execution_timeout_seconds()
        deadline = time.monotonic() + timeout if timeout is not None else None
        for index, configured_command in enumerate(self.commands, start=1):
            command = self._command(configured_command)
            name = f"{index:02d}_{self._command_name(command)}"
            command_timeout = (
                max(0.0, deadline - time.monotonic())
                if deadline is not None
                else None
            )
            if command_timeout == 0.0:
                raise XHeepVerilatorSimulationTimeoutError(
                    f"X-HEEP tool flow exceeded timeout of {timeout:g} seconds."
                )
            try:
                result = context.run_command(
                    command,
                    cwd=checkout_dir,
                    env={"LC_ALL": "C"},
                    check=False,
                    timeout=command_timeout,
                )
            except subprocess.TimeoutExpired as exc:
                stdout = self._timeout_output(exc.stdout)
                stderr = self._timeout_output(exc.stderr)
                stdout_path = context.write_log(self, f"{name}_stdout.log", stdout)
                stderr_path = context.write_log(self, f"{name}_stderr.log", stderr)
                log_paths.extend((stdout_path, stderr_path))
                context.write_json(
                    context.artifact_path(self, "xheep_run_metadata.json"),
                    {"status": "timeout", "command": command, "log_paths": [str(path) for path in log_paths]},
                )
                raise XHeepVerilatorSimulationTimeoutError(
                    f"X-HEEP command {command!r} exceeded timeout of {timeout:g} seconds."
                ) from exc
            stdout_path = context.write_log(self, f"{name}_stdout.log", result.stdout)
            stderr_path = context.write_log(self, f"{name}_stderr.log", result.stderr)
            log_paths.extend((stdout_path, stderr_path))
            outputs.extend((result.stdout, result.stderr))
            if result.returncode != 0:
                context.write_json(
                    context.artifact_path(self, "xheep_run_metadata.json"),
                    {"status": "failed", "command": result.command, "returncode": result.returncode, "log_paths": [str(path) for path in log_paths]},
                )
                raise XHeepVerilatorSimulationError(
                    result.command, result.returncode, tuple(log_paths)
                )
        output = "\n".join(outputs)
        context.write_json(
            context.artifact_path(self, "xheep_run_metadata.json"),
            {"status": "success", "commands": [list(self._command(command)) for command in self.commands], "log_paths": [str(path) for path in log_paths]},
        )
        return _XHeepRunResult(checkout_dir, tuple(log_paths), output)

    def _simulation_artifact(
        self,
        context: ExecutionContext,
        run_result: _XHeepRunResult,
    ) -> XHeepSimulationArtifact:
        """Parse, persist, validate, and package the simulation measurements.

        ``xheep_simulation.json`` is written before firmware statuses are validated,
        preserving parsed metrics and log locations even for invalid results.
        """

        values = self._parse_metrics(run_result.output)
        metadata_path = context.write_json(
            context.artifact_path(self, "xheep_simulation.json"),
            {
                "checkout_dir": str(run_result.checkout_dir),
                "log_paths": [str(path) for path in run_result.log_paths],
                "metrics": values,
            },
        )
        self._validate_simulation_result(values, run_result.log_paths)
        return XHeepSimulationArtifact(
            name="xheep_verilator_simulation",
            producer=self.step_id or "xheep_verilator_simulation",
            individual_id=self.individual.id,
            log_paths=run_result.log_paths,
            values=values,
            metadata={"path": str(metadata_path), "checkout_dir": str(run_result.checkout_dir)},
        )

    @staticmethod
    def _parse_metrics(output: str) -> dict[str, float]:
        """Parse the GENIO firmware and Verilator textual result protocols.

        ``GENIO_PERF`` regions become ``<region>_cycles`` metrics and
        ``GENIO_METRIC`` names are preserved. The last ``Program Finished`` value
        wins, while status and total simulation cycles use their first regex match.

        Args:
            output: Concatenated stdout and stderr from all tool-flow commands.

        Returns:
            Parsed numeric values represented uniformly as floats.
        """

        values: dict[str, float] = {}
        for region, cycles in re.findall(r"GENIO_PERF:([^:\s]+):(\d+)", output):
            values[f"{region}_cycles"] = float(cycles)
        for metric, value in re.findall(
            r"GENIO_METRIC:([^:\s]+):(-?\d+(?:\.\d+)?)",
            output,
        ):
            values[metric] = float(value)
        status = re.search(r"GENIO_STATUS:(-?\d+)", output)
        if status is not None:
            values["status"] = float(status.group(1))
        total = re.search(r"Simulation finished after\s+(\d+)\s+clock cycles", output)
        if total is not None:
            values["simulation_cycles"] = float(total.group(1))
        program_statuses = re.findall(r"Program Finished with value\s+(-?\d+)", output)
        if program_statuses:
            values["program_status"] = float(program_statuses[-1])
        return values

    @staticmethod
    def _validate_simulation_result(
        values: Mapping[str, float],
        log_paths: tuple[Path, ...],
    ) -> None:
        """Require successful firmware and optional program completion statuses.

        Raises:
            XHeepVerilatorSimulationResultError: If ``GENIO_STATUS`` is absent or
                nonzero, or if an emitted program-finish status is nonzero.
        """

        if "status" not in values:
            hint = f" See {log_paths[-1]}." if log_paths else ""
            raise XHeepVerilatorSimulationResultError(
                "X-HEEP firmware did not emit GENIO_STATUS:<value>." + hint
            )
        if values["status"] != 0.0:
            hint = f" See {log_paths[-1]}." if log_paths else ""
            raise XHeepVerilatorSimulationResultError(
                f"X-HEEP firmware reported GENIO_STATUS:{values['status']:g}." + hint
            )
        if values.get("program_status", 0.0) != 0.0:
            hint = f" See {log_paths[-1]}." if log_paths else ""
            raise XHeepVerilatorSimulationResultError(
                f"X-HEEP program finished with value {values['program_status']:g}." + hint
            )

    def _require_gr_heep_composer(self) -> GRHeepConfigurationComposer:
        """Return the composer after enforcing the GR-HEEP-specific protocol."""

        if not isinstance(self.composer, GRHeepConfigurationComposer):
            raise XHeepVerilatorSimulationConfigurationError(
                "XHeepVerilatorSimulationTask requires GRHeepConfigurationComposer."
            )
        return self.composer

    def _command(self, command: Sequence[str]) -> tuple[str, ...]:
        """Render placeholders, substitute the make tool, and wrap with conda.

        The ``{application_name}`` token is replaced in every argument. Only a
        leading executable equal to ``make`` is replaced by :attr:`make_tool`.
        """

        if not command:
            raise XHeepVerilatorSimulationConfigurationError("Empty X-HEEP command.")
        normalized = tuple(
            argument.replace("{application_name}", self.application_name)
            for argument in command
        )
        if normalized[0] == "make":
            normalized = (self.make_tool, *normalized[1:])
        return (
            self.conda_tool,
            "run",
            "--no-capture-output",
            "-n",
            self.conda_environment,
            *normalized,
        )

    @staticmethod
    def _command_name(command: Sequence[str]) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(command[:2]))

    @staticmethod
    def _timeout_output(output: str | bytes | None) -> str:
        if output is None:
            return ""
        return output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output

    @staticmethod
    def _sha256(path: Path) -> str:
        """Hash an RTL file in bounded chunks for the task cache signature."""

        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class XHeepVerilatorSimulationEvaluationStep(EvaluationStep):
    """Configure GR-HEEP/Verilator tasks that consume synthesized HLS RTL.

    The step requires the artifact key
    ``hls_image_pipeline_synthesis.rtl_hls_synthesis`` and transfers that
    :class:`HLSRTLArtifact` into every simulation task.

    Attributes:
        id: Evaluation graph identifier and artifact producer name.
        depends_on: Step identifiers that must complete before simulation.
        composer: GR-HEEP composer propagated to simulation tasks.
        gr_heep_path: Source checkout used to create isolated task trees.
        input_image_path: Optional image embedded in generated firmware data.
        conda_environment: Environment used by all commands.
        conda_tool: Conda-compatible executable.
        make_tool: Executable substituted for leading ``make`` commands.
        commands: Ordered build and simulation command templates.
        metadata: Additional metadata propagated to tasks.
        required_artifacts: Required upstream artifact key and type mapping.
        task_type: Concrete simulation task class.
    """

    id: str = "xheep_verilator_simulation"
    produced_artifacts = {
        "xheep_verilator_simulation": XHeepSimulationArtifact,
    }
    depends_on: tuple[str, ...] = ()
    composer: GRHeepConfigurationComposer | None = None
    gr_heep_path: Path | None = None
    input_image_path: Path | None = None
    conda_environment: str = "core-v-mini-mcu"
    conda_tool: str = "conda"
    make_tool: str = "make"
    commands: tuple[tuple[str, ...], ...] = (
        ("make", "mcu-gen"),
        ("make", "verilator-build"),
        ("make", "app", "PROJECT={application_name}"),
        ("make", "verilator-run"),
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    required_artifacts: Mapping[str, type[Artifact]] = field(
        default_factory=lambda: MappingProxyType(
            {"hls_image_pipeline_synthesis.rtl_hls_synthesis": HLSRTLArtifact}
        )
    )

    task_type: type[EvaluationTask] = XHeepVerilatorSimulationTask

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return all configuration that determines X-HEEP step reuse.

        Returns:
            A checkpoint mapping containing the base step signature, composer,
            source/image paths, tool selection, command templates, and metadata.
        """

        return {
            **EvaluationStep.checkpoint_signature(self),
            "composer": (
                self.composer.checkpoint_signature()
                if isinstance(self.composer, GRHeepConfigurationComposer)
                else None
            ),
            "gr_heep_path": self.gr_heep_path,
            "input_image_path": self.input_image_path,
            "conda_environment": self.conda_environment,
            "conda_tool": self.conda_tool,
            "make_tool": self.make_tool,
            "commands": self.commands,
            "metadata": dict(self.metadata),
        }

    def create_task(
        self,
        individual: Individual,
        artifacts: Mapping[str, Artifact],
    ) -> EvaluationTask:
        """Create a simulation task from the required synthesized RTL artifact.

        Args:
            individual: Candidate whose system configuration will be simulated.
            artifacts: Upstream artifacts keyed by fully qualified artifact name.

        Returns:
            A configured :class:`XHeepVerilatorSimulationTask` carrying the HLS RTL
            artifact and the sorted upstream artifact names in metadata.

        Raises:
            KeyError: If the required HLS synthesis artifact is absent.
            AssertionError: If the required artifact is not an
                :class:`HLSRTLArtifact`.
        """

        artifact = artifacts["hls_image_pipeline_synthesis.rtl_hls_synthesis"]
        assert isinstance(artifact, HLSRTLArtifact)
        return XHeepVerilatorSimulationTask(
            individual=individual,
            step_id=self.id,
            composer=self.composer,
            hls_artifact=artifact,
            gr_heep_path=self.gr_heep_path,
            input_image_path=self.input_image_path,
            conda_environment=self.conda_environment,
            conda_tool=self.conda_tool,
            make_tool=self.make_tool,
            commands=self.commands,
            metadata={**dict(self.metadata), "input_artifacts": tuple(sorted(artifacts))},
        )


__all__ = [
    "XHeepVerilatorSimulationConfigurationError",
    "XHeepVerilatorSimulationError",
    "XHeepVerilatorSimulationEvaluationStep",
    "XHeepVerilatorSimulationResultError",
    "XHeepVerilatorSimulationTask",
    "XHeepVerilatorSimulationTimeoutError",
]
