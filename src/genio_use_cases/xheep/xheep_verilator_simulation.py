from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from genio.evaluation.task import ArtifactInput, EvaluationOutput, EvaluationTask, TaskContext
from genio_use_cases.image_pipeline.hls_synthesis import HLSRTLArtifact
from genio_use_cases.xheep.gr_heep import (
    GRHeepConfigurationComposer,
    GRHeepConfigurationPackage,
)


_HLS_ARTIFACT_KEY = "hls_image_pipeline_synthesis.rtl_hls_synthesis"


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
    command is wrapped with ``conda run --no-capture-output``.

    Firmware communicates results through textual records:
    ``GENIO_PERF:<region>:<cycles>``, ``GENIO_METRIC:<name>:<value>``, and the
    mandatory ``GENIO_STATUS:<value>``. Verilator and program-finish cycle/status
    messages are also recognized. A successful result requires zero GENIO status
    and, when emitted, zero program status.

    Attributes:
        composer: GR-HEEP configuration and overlay composer.
        artifacts: Upstream artifacts, including the SAFA FIFO RTL to integrate.
        gr_heep_path: Source GR-HEEP checkout copied for the task.
        input_image_path: Optional image embedded by the generated overlay.
        application_name: Read-only firmware project name obtained from ``composer``.
        conda_environment: Environment used for all build and simulation commands.
        conda_tool: Conda-compatible executable.
        make_tool: Replacement executable for commands beginning with ``make``.
        commands: Ordered command templates executed in the isolated checkout.
        metadata: Additional execution metadata and upstream artifact names.
    """

    uses_domains = ("system",)

    requires: ClassVar[tuple[ArtifactInput, ...]] = (
        ArtifactInput(_HLS_ARTIFACT_KEY, HLSRTLArtifact),
    )

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

    @property
    def application_name(self) -> str:
        """Return the firmware project selected by the GR-HEEP composer."""

        return self._require_gr_heep_composer().application_name

    @property
    def _hls_artifact(self) -> HLSRTLArtifact:
        try:
            artifact = self.artifacts[_HLS_ARTIFACT_KEY]
        except KeyError as exc:
            raise XHeepVerilatorSimulationConfigurationError(
                f"X-HEEP simulation requires HLS RTL artifact {_HLS_ARTIFACT_KEY!r}."
            ) from exc
        if not isinstance(artifact, HLSRTLArtifact):
            raise XHeepVerilatorSimulationConfigurationError(
                f"X-HEEP simulation requires {_HLS_ARTIFACT_KEY!r} to be an HLSRTLArtifact."
            )
        return artifact

    def run(self, context: TaskContext) -> EvaluationOutput:
        """Build and simulate an isolated GR-HEEP overlay for the candidate.

        Args:
            context: Task workspace and execution resources.

        Returns:
            Simulation metrics prefixed with ``xheep_verilator.`` and no artifacts.
            Command log paths remain in ``xheep_simulation.json``.

        Raises:
            XHeepVerilatorSimulationConfigurationError: If inputs, commands, or
                the GR-HEEP/composer contract are invalid.
            TypeError: If composition returns an unexpected package type.
            XHeepVerilatorSimulationError: If any command exits unsuccessfully.
            XHeepVerilatorSimulationResultError: If firmware omits GENIO status
                or reports a failing GENIO/program status.
        """

        gr_heep_path = self._validate_configuration(context)
        package = self._complete_package()
        checkout_dir = self._copy_checkout(context, gr_heep_path)
        self._apply_overlay(context, package, checkout_dir)
        run_result = self._run_commands(context, checkout_dir)
        metrics = self._simulation_metrics(context, run_result)
        return EvaluationOutput(metrics=metrics)

    def _validate_configuration(self, context: TaskContext) -> Path:
        """Validate the GR-HEEP, composer, RTL, image, and command contracts.

        The RTL artifact must expose at least one existing Verilog file and declare
        the ``safa_fifo`` interface required by the generated GR-HEEP wrapper.

        Returns:
            The resolved GR-HEEP source directory.
        """

        hls_artifact = self._hls_artifact
        if self.composer is None:
            raise XHeepVerilatorSimulationConfigurationError(
                "XHeepVerilatorSimulationTask requires a GRHeepConfigurationComposer."
            )
        self._require_gr_heep_composer()
        if self.gr_heep_path is None:
            raise XHeepVerilatorSimulationConfigurationError(
                "XHeepVerilatorSimulationTask requires gr_heep_path."
            )
        gr_heep_path = Path(self.gr_heep_path).expanduser().resolve()
        if not gr_heep_path.is_dir():
            raise XHeepVerilatorSimulationConfigurationError(
                f"gr_heep_path must be an existing directory: {gr_heep_path}."
            )
        if not (gr_heep_path / "Makefile").is_file():
            raise XHeepVerilatorSimulationConfigurationError(
                f"gr_heep_path does not contain a Makefile: {gr_heep_path}."
            )
        if self.input_image_path is not None and not Path(
            self.input_image_path
        ).expanduser().resolve().is_file():
            raise XHeepVerilatorSimulationConfigurationError(
                f"input_image_path must be an existing image: {self.input_image_path}."
            )
        if not isinstance(hls_artifact.data, Mapping):
            raise XHeepVerilatorSimulationConfigurationError(
                "HLS RTL artifact data must be a mapping of RTL metadata."
            )
        if hls_artifact.data.get("interface") != "safa_fifo":
            raise XHeepVerilatorSimulationConfigurationError(
                "X-HEEP simulation requires HLS RTL with interface 'safa_fifo'."
            )
        if not isinstance(hls_artifact.files, Mapping):
            raise XHeepVerilatorSimulationConfigurationError(
                "HLS RTL artifact files must be a mapping of names to paths."
            )
        verilog_files = [
            (key, path)
            for key, path in hls_artifact.files.items()
            if isinstance(key, str) and key.startswith("verilog/")
        ]
        if not verilog_files:
            raise XHeepVerilatorSimulationConfigurationError(
                "X-HEEP simulation requires Verilog files in the HLS RTL artifact."
            )
        for key, rtl_path in verilog_files:
            if not isinstance(rtl_path, Path):
                raise XHeepVerilatorSimulationConfigurationError(
                    f"HLS RTL file {key!r} must be a Path."
                )
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
        hls_artifact = self._hls_artifact
        if not isinstance(hls_artifact.data, Mapping):
            raise XHeepVerilatorSimulationConfigurationError(
                "HLS RTL artifact data must be a mapping of RTL metadata."
            )
        rtl_overlay = composer.render_hls_artifact_overlay(
            hls_artifact,
            image_path=(
                Path(self.input_image_path).expanduser().resolve()
                if self.input_image_path is not None
                else None
            ),
            configuration=package.metadata.get("rendered_configuration", {}),
        )
        return GRHeepConfigurationPackage(
            files={**dict(package.files), **dict(rtl_overlay)},
            metadata={
                **dict(package.metadata),
                "hls_artifact": {
                    "producer": _HLS_ARTIFACT_KEY.split(".", 1)[0],
                    "name": _HLS_ARTIFACT_KEY.split(".", 1)[1],
                    "top_function": hls_artifact.data["top_function"],
                    "verilog_files": tuple(
                        Path(key).name for key in hls_artifact.files
                        if isinstance(key, str) and key.startswith("verilog/")
                    ),
                },
            },
        )

    def _copy_checkout(self, context: TaskContext, source: Path) -> Path:
        """Create a fresh isolated GR-HEEP tree for destructive overlay writes.

        Existing task-local copies are removed. The destination is rejected if it
        resolves inside the source tree, preventing recursive self-copying.
        """

        checkout_dir = context.workspace / "xheep"
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
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, checkout_dir, dirs_exist_ok=False, symlinks=True)
        return checkout_dir

    def _apply_overlay(
        self,
        context: TaskContext,
        package: GRHeepConfigurationPackage,
        checkout_dir: Path,
    ) -> None:
        """Materialize generated files and persist their composition metadata."""

        package.materialize(checkout_dir)
        _write_json(
            context.workspace / "artifacts" / "xheep_overlay_metadata.json",
            dict(package.metadata),
        )

    def _run_commands(
        self,
        context: TaskContext,
        checkout_dir: Path,
    ) -> _XHeepRunResult:
        """Execute the configured tool flow with persistent per-command logs.

        Stdout and stderr are persisted separately with an ordered command prefix.
        Run metadata is written for nonzero exit and complete success.

        Returns:
            The checkout, ordered log paths, and concatenated process output.

        Raises:
            XHeepVerilatorSimulationError: If a command returns nonzero.
        """

        log_paths: list[Path] = []
        outputs: list[str] = []
        logs_dir = context.workspace / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        for index, configured_command in enumerate(self.commands, start=1):
            command = self._command(configured_command)
            name = f"{index:02d}_{self._command_name(command)}"
            stdout_path = logs_dir / f"{name}_stdout.log"
            stderr_path = logs_dir / f"{name}_stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr_file:
                result = subprocess.run(
                    command,
                    cwd=checkout_dir,
                    env={**os.environ, "LC_ALL": "C"},
                    stdout=stdout_file,
                    stderr=stderr_file,
                    check=False,
                )
            stdout = stdout_path.read_text(encoding="utf-8")
            stderr = stderr_path.read_text(encoding="utf-8")
            log_paths.extend((stdout_path, stderr_path))
            outputs.extend((stdout, stderr))
            if result.returncode != 0:
                _write_json(
                    context.workspace / "artifacts" / "xheep_run_metadata.json",
                    {
                        "status": "failed",
                        "command": command,
                        "returncode": result.returncode,
                        "log_paths": [str(path) for path in log_paths],
                    },
                )
                raise XHeepVerilatorSimulationError(
                    command, result.returncode, tuple(log_paths)
                )
        output = "\n".join(outputs)
        _write_json(
            context.workspace / "artifacts" / "xheep_run_metadata.json",
            {
                "status": "success",
                "commands": [list(self._command(command)) for command in self.commands],
                "log_paths": [str(path) for path in log_paths],
            },
        )
        return _XHeepRunResult(checkout_dir, tuple(log_paths), output)

    def _simulation_metrics(
        self,
        context: TaskContext,
        run_result: _XHeepRunResult,
    ) -> Mapping[str, float]:
        """Parse, persist, and validate the simulation measurements.

        ``xheep_simulation.json`` is written before firmware statuses are validated,
        preserving parsed metrics and log locations even for invalid results.
        """

        values = self._parse_metrics(run_result.output)
        _write_json(
            context.workspace / "artifacts" / "xheep_simulation.json",
            {
                "checkout_dir": str(run_result.checkout_dir),
                "log_paths": [str(path) for path in run_result.log_paths],
                "metrics": values,
            },
        )
        self._validate_simulation_result(values, run_result.log_paths)
        return {f"xheep_verilator.{key}": value for key, value in values.items()}

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


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


__all__ = [
    "XHeepVerilatorSimulationConfigurationError",
    "XHeepVerilatorSimulationError",
    "XHeepVerilatorSimulationResultError",
    "XHeepVerilatorSimulationTask",
]
