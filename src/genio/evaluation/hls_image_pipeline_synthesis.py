from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genio.artifacts import (
    Artifact,
    HLSReportArtifact,
    HLSRTLArtifact,
)
from genio.composer import Composer, HLSExecutionPackage
from genio.core.individual import Individual
from genio.evaluation.hls_reports import parse_hls_synthesis_report
from genio.evaluation.step import EvaluationStep
from genio.evaluation.task import EvaluationTask, ExecutionContext


class HLSImagePipelineSynthesisConfigurationError(ValueError):
    """Raised when an HLS image pipeline synthesis task is misconfigured."""


class HLSImagePipelineSynthesisError(RuntimeError):
    """Raised when the external HLS synthesis command fails."""

    def __init__(
        self,
        command: tuple[str, ...],
        returncode: int,
        log_paths: tuple[Path, ...] = (),
    ) -> None:
        self.command = command
        self.returncode = returncode
        self.log_paths = log_paths
        log_hint = f" See {log_paths[0]}." if log_paths else ""
        super().__init__(
            f"HLS command {command!r} failed with return code {returncode}.{log_hint}"
        )


class HLSImagePipelineSynthesisTimeoutError(TimeoutError):
    """Raised when the external HLS synthesis command exceeds its timeout."""

    def __init__(
        self,
        command: tuple[str, ...],
        timeout_seconds: float,
        log_paths: tuple[Path, ...] = (),
    ) -> None:
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.log_paths = log_paths
        log_hint = f" See {log_paths[0]}." if log_paths else ""
        super().__init__(
            f"HLS command {command!r} exceeded timeout of "
            f"{timeout_seconds:g} seconds.{log_hint}"
        )


@dataclass(frozen=True, slots=True)
class _MaterializedHLSPackage:
    package_dir: Path
    package_metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _HLSRunResult:
    work_dir: Path


@dataclass(frozen=True, slots=True)
class HLSImagePipelineSynthesisTask(EvaluationTask):
    """Synthesizes an image-processing pipeline into HLS metrics and RTL."""

    composer: Composer | None = None
    hls_tool: str = "v++"
    hls_config: Path | None = None
    work_dir_name: str = "work"
    top_function: str | None = None
    clock_period: float | None = None
    part: str | None = None
    config_defaults: Mapping[str, str] = field(default_factory=dict)
    config_overrides: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def cache_inputs(self) -> Mapping[str, Any]:
        """Cache synthesis by pipeline and the HLS design domain."""

        return {
            "pipeline": self._pipeline_cache_inputs(),
            "hls_design": self.individual.design.get("hls", {}),
        }

    def run(self, context: ExecutionContext) -> list[Artifact]:
        """Synthesize the composed HLS pipeline and return report and RTL artifacts."""

        self._validate_configuration(context)

        package = self._compose_and_materialize(context)
        hls_config_path = self._prepare_hls_config(context, package)
        run_result = self._run_hls_tool(context, hls_config_path)

        report_artifact = self._parse_hls_synthesis_report(
            context,
            package,
            run_result,
        )
        rtl_artifact = self._collect_hls_rtl(
            context,
            package,
            run_result,
        )

        return [report_artifact, rtl_artifact]

    def _collect_hls_rtl(
        self,
        context: ExecutionContext,
        package: _MaterializedHLSPackage,
        run_result: _HLSRunResult,
    ) -> HLSRTLArtifact:
        top_function = self.top_function or package.package_metadata.get("top_function")
        if top_function is None:
            raise RuntimeError("Cannot identify the top function for generated HLS RTL.")

        synthesis_dir = run_result.work_dir / "hls" / "syn"
        verilog_dir = synthesis_dir / "verilog"
        vhdl_dir = synthesis_dir / "vhdl"
        verilog_paths = self._rtl_paths(verilog_dir, ("*.v", "*.sv", "*.vh"))
        vhdl_paths = self._rtl_paths(vhdl_dir, ("*.vhd", "*.vhdl"))
        if not verilog_paths and not vhdl_paths:
            raise RuntimeError(
                f"HLS synthesis completed but generated no RTL files under {synthesis_dir}."
            )

        origin = "hls_synthesis"
        metadata_path = context.write_json(
            context.artifact_path(self, f"hls_rtl_{origin}.json"),
            {
                "origin": origin,
                "top_function": str(top_function),
                "verilog_paths": tuple(str(path) for path in verilog_paths),
                "vhdl_paths": tuple(str(path) for path in vhdl_paths),
            },
        )
        return HLSRTLArtifact(
            name=f"rtl_{origin}",
            producer=self.step_id or "hls_image_pipeline_synthesis",
            individual_id=self.individual.id,
            origin=origin,
            top_function=str(top_function),
            verilog_paths=verilog_paths,
            vhdl_paths=vhdl_paths,
            metadata={"path": str(metadata_path)},
        )

    @staticmethod
    def _rtl_paths(directory: Path, patterns: tuple[str, ...]) -> tuple[Path, ...]:
        paths: set[Path] = set()
        for pattern in patterns:
            paths.update(directory.rglob(pattern))
        return tuple(sorted(path for path in paths if path.is_file()))

    def _parse_hls_synthesis_report(
        self,
        context: ExecutionContext,
        package: _MaterializedHLSPackage,
        run_result: _HLSRunResult,
    ) -> HLSReportArtifact:
        top_function = self.top_function or package.package_metadata.get("top_function")
        parsed_report = parse_hls_synthesis_report(
            run_result.work_dir,
            top_function=str(top_function) if top_function is not None else None,
        )
        metadata_path = context.write_json(
            context.artifact_path(self, f"hls_report_{parsed_report.origin}.json"),
            {
                "origin": parsed_report.origin,
                "report_paths": tuple(str(path) for path in parsed_report.report_paths),
                "metrics": dict(parsed_report.metrics),
                "metadata": dict(parsed_report.metadata),
            },
        )
        return HLSReportArtifact(
            name=f"report_{parsed_report.origin}",
            producer=self.step_id or "hls_image_pipeline_synthesis",
            individual_id=self.individual.id,
            origin=parsed_report.origin,
            report_paths=parsed_report.report_paths,
            values=parsed_report.metrics,
            metadata={
                "path": str(metadata_path),
                **dict(parsed_report.metadata),
            },
        )

    def _run_hls_tool(
        self,
        context: ExecutionContext,
        hls_config_path: Path,
    ) -> _HLSRunResult:
        package_dir = context.package_dir(self)
        work_dir = context.ensure_dir(package_dir / self.work_dir_name)
        command = (
            self.hls_tool,
            "--compile",
            "--mode",
            "hls",
            "--config",
            hls_config_path.name,
            "--work_dir",
            self.work_dir_name,
        )
        # Keep control of failures so logs and command metadata are always persisted.
        timeout_seconds = self.execution_timeout_seconds()
        try:
            result = context.run_command(
                command,
                cwd=package_dir,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path = context.write_log(
                self,
                "hls_stdout.log",
                self._timeout_output(exc.stdout),
            )
            stderr_path = context.write_log(
                self,
                "hls_stderr.log",
                self._timeout_output(exc.stderr),
            )
            tool_log_paths = self._tool_log_paths(work_dir)
            context.write_json(
                context.artifact_path(self, "hls_run_metadata.json"),
                {
                    "command": command,
                    "status": "timeout",
                    "returncode": None,
                    "timeout_seconds": timeout_seconds,
                    "cwd": str(package_dir),
                    "work_dir": str(work_dir),
                    "hls_config": str(hls_config_path),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "tool_log_paths": tuple(str(path) for path in tool_log_paths),
                },
            )
            assert timeout_seconds is not None
            raise HLSImagePipelineSynthesisTimeoutError(
                command=command,
                timeout_seconds=timeout_seconds,
                log_paths=tool_log_paths,
            ) from exc
        stdout_path = context.write_log(self, "hls_stdout.log", result.stdout)
        stderr_path = context.write_log(self, "hls_stderr.log", result.stderr)
        tool_log_paths = self._tool_log_paths(work_dir)
        context.write_json(
            context.artifact_path(self, "hls_run_metadata.json"),
            {
                "command": result.command,
                "status": "success" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "timeout_seconds": timeout_seconds,
                "cwd": str(result.cwd) if result.cwd is not None else None,
                "work_dir": str(work_dir),
                "hls_config": str(hls_config_path),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "tool_log_paths": tuple(str(path) for path in tool_log_paths),
            },
        )
        if result.returncode != 0:
            raise HLSImagePipelineSynthesisError(
                command=result.command,
                returncode=result.returncode,
                log_paths=tool_log_paths,
            )
        return _HLSRunResult(work_dir=work_dir)

    @staticmethod
    def _timeout_output(output: str | bytes | None) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output

    @staticmethod
    def _tool_log_paths(work_dir: Path) -> tuple[Path, ...]:
        candidates = (
            work_dir / "logs" / "hls_compile.log",
            work_dir / "logs" / "work.steps.log",
            work_dir / "logs" / "xcd.log",
        )
        return tuple(path for path in candidates if path.exists())

    def _prepare_hls_config(
        self,
        context: ExecutionContext,
        package: _MaterializedHLSPackage,
    ) -> Path:
        package_dir = package.package_dir
        package_config_path = package_dir / "hls_config.cfg"

        # Precedence: base config < defaults < HLS design < package values
        # < explicit task fields < overrides; backend include flags are appended last.
        if self.hls_config is not None:
            config = self._parse_hls_config(context.read_text(self.hls_config))
        elif package_config_path.exists():
            config = self._parse_hls_config(context.read_text(package_config_path))
        else:
            config = {}

        self._apply_config_mapping(config, self.config_defaults)
        hls_design = self.individual.design.get("hls", {})
        if isinstance(hls_design, Mapping):
            config_keys = self._hls_design_config_keys(hls_design)
            self._apply_config_mapping(config, config_keys)

        package_metadata = package.package_metadata
        top_function = self.top_function or package_metadata.get("top_function")
        if top_function is not None:
            self._set_config_value(config, "hls.syn.top", str(top_function))
        source_files = package_metadata.get("source_files")
        if source_files:
            self._set_config_value(config, "hls.syn.file", str(tuple(source_files)[0]))

        if self.part is not None:
            self._set_config_value(config, "part", self.part)
        if self.clock_period is not None:
            self._set_config_value(config, "hls.clock", str(self.clock_period))

        self._apply_config_mapping(config, self.config_overrides)
        self._apply_package_backend_config(context, package_dir, package_metadata, config)

        if "hls" not in config or not isinstance(config["hls"], dict):
            config["hls"] = {}
        hls_section = config["hls"]
        for key in ("syn.file", "syn.top"):
            if key not in hls_section:
                raise HLSImagePipelineSynthesisConfigurationError(
                    f"Final hls_config.cfg is missing required [hls] key: {key}."
                )

        context.write_text(package_config_path, self._render_hls_config(config))
        context.write_json(
            package_dir / "hls_config_metadata.json",
            {
                "path": str(package_config_path),
                "config": config,
            },
        )
        return package_config_path

    def _apply_package_backend_config(
        self,
        context: ExecutionContext,
        package_dir: Path,
        package_metadata: Mapping[str, Any],
        config: dict[str, Any],
    ) -> None:
        include_flags = [
            self._include_cflag(include_dir)
            for include_dir in self._package_include_dirs(package_dir, package_metadata)
        ]

        required_resources = self._metadata_tuple(package_metadata, "required_backend_resources")
        if "vitis_libraries_path" in required_resources:
            vitis_libraries_path = context.metadata.get("vitis_libraries_path")
            if not vitis_libraries_path:
                raise HLSImagePipelineSynthesisConfigurationError(
                    "Backend context must provide metadata['vitis_libraries_path'] "
                    "for Vitis Vision HLS packages."
                )
            vision_include_path = (
                Path(str(vitis_libraries_path)).expanduser().resolve()
                / "vision"
                / "L1"
                / "include"
            )
            if not vision_include_path.exists():
                raise HLSImagePipelineSynthesisConfigurationError(
                    "Vitis Vision include path does not exist: "
                    f"{vision_include_path}."
                )
            include_flags.append(self._include_cflag(vision_include_path))

        self._append_hls_cflags(config, include_flags)

    @classmethod
    def _package_include_dirs(
        cls,
        package_dir: Path,
        package_metadata: Mapping[str, Any],
    ) -> tuple[str, ...]:
        include_dirs = cls._metadata_tuple(package_metadata, "include_dirs")
        return tuple(include_dir for include_dir in include_dirs if (package_dir / include_dir).exists())

    @staticmethod
    def _metadata_tuple(metadata: Mapping[str, Any], key: str) -> tuple[str, ...]:
        values = metadata.get(key, ())
        if isinstance(values, str):
            return (values,)
        return tuple(str(value) for value in values)

    @classmethod
    def _append_hls_cflags(cls, config: dict[str, Any], flags: list[str]) -> None:
        if not flags:
            return
        if "hls" not in config or not isinstance(config["hls"], dict):
            config["hls"] = {}
        hls_section = config["hls"]
        existing = str(hls_section.get("syn.cflags", "")).strip()
        cflags = existing.split() if existing else []
        for flag in flags:
            if flag not in cflags:
                cflags.append(flag)
        hls_section["syn.cflags"] = " ".join(cflags)

    @staticmethod
    def _include_cflag(path: str | Path) -> str:
        value = str(path)
        if any(character.isspace() for character in value):
            return f'-I"{value}"'
        return f"-I{value}"

    def _compose_and_materialize(self, context: ExecutionContext) -> _MaterializedHLSPackage:
        assert self.composer is not None
        package = self.composer.compose(self.individual)
        if not isinstance(package, HLSExecutionPackage):
            raise TypeError(
                "HLSImagePipelineSynthesisTask requires composer.compose() to return "
                "HLSExecutionPackage."
            )

        package_dir = context.materialize_package(self, package)
        metadata = {
            "entrypoint": package.entrypoint,
            "files": tuple(sorted(package.files)),
            "package_metadata": dict(package.metadata),
        }
        context.write_json(package_dir / "composition_metadata.json", metadata)

        return _MaterializedHLSPackage(
            package_dir=package_dir,
            package_metadata=dict(package.metadata),
        )

    @staticmethod
    def _parse_hls_config(content: str) -> dict[str, Any]:
        config: dict[str, Any] = {}
        current_section: str | None = None
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                config.setdefault(current_section, {})
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if current_section is None:
                config[key] = value
            else:
                section = config.setdefault(current_section, {})
                assert isinstance(section, dict)
                section[key] = value
        return config

    @classmethod
    def _apply_config_mapping(
        cls,
        config: dict[str, Any],
        values: Mapping[str, Any],
    ) -> None:
        for key, value in values.items():
            cls._set_config_value(config, key, str(value))

    @staticmethod
    def _set_config_value(config: dict[str, Any], key: str, value: str) -> None:
        if "." not in key:
            config[key] = value
            return
        section_name, option = key.split(".", 1)
        section = config.setdefault(section_name, {})
        if not isinstance(section, dict):
            raise HLSImagePipelineSynthesisConfigurationError(
                f"Cannot set {key!r}: {section_name!r} is not a config section."
            )
        section[option] = value

    @staticmethod
    def _render_hls_config(config: Mapping[str, Any]) -> str:
        lines: list[str] = []
        sections: list[tuple[str, Mapping[str, Any]]] = []
        for key, value in config.items():
            if isinstance(value, Mapping):
                sections.append((key, value))
            else:
                lines.append(f"{key}={value}")
        if lines and sections:
            lines.append("")
        for section_name, section in sections:
            lines.append(f"[{section_name}]")
            for key, value in section.items():
                lines.append(f"{key}={value}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _validate_configuration(self, context: ExecutionContext) -> None:
        if self.composer is None:
            raise HLSImagePipelineSynthesisConfigurationError(
                "HLSImagePipelineSynthesisTask requires a composer before synthesis can run."
            )

        if not self.hls_tool:
            raise HLSImagePipelineSynthesisConfigurationError("hls_tool cannot be empty.")

        try:
            self.execution_timeout_seconds()
        except ValueError as exc:
            raise HLSImagePipelineSynthesisConfigurationError(str(exc)) from exc

        if not self.work_dir_name:
            raise HLSImagePipelineSynthesisConfigurationError(
                "work_dir_name cannot be empty."
            )
        work_dir = Path(self.work_dir_name)
        if work_dir.is_absolute() or ".." in work_dir.parts:
            raise HLSImagePipelineSynthesisConfigurationError(
                "work_dir_name must be a relative path inside the task workspace."
            )

        hls_design = self.individual.design.get("hls", {})
        if hls_design and not isinstance(hls_design, Mapping):
            raise HLSImagePipelineSynthesisConfigurationError(
                "Individual design domain 'hls' must be a mapping."
            )

        if self.hls_config is not None:
            hls_config = context.resolve_path(self.hls_config)
            if hls_config.suffix != ".cfg":
                raise HLSImagePipelineSynthesisConfigurationError(
                    f"hls_config must point to a .cfg file, got {hls_config}."
                )
            if not hls_config.exists():
                raise HLSImagePipelineSynthesisConfigurationError(
                    f"hls_config file does not exist: {hls_config}."
                )

        design_config = (
            self._hls_design_config_keys(hls_design)
            if isinstance(hls_design, Mapping)
            else {}
        )
        available_config = {
            **dict(self.config_defaults),
            **design_config,
            **dict(self.config_overrides),
        }
        has_config_source = self.hls_config is not None or bool(available_config)
        if not has_config_source:
            # The composed package may provide hls_config.cfg, so defer this check.
            return

        required_if_task_generates_config = {
            "hls.clock",
            "hls.flow_target",
        }
        missing = sorted(
            key for key in required_if_task_generates_config if key not in available_config
        )
        if self.hls_config is None and missing:
            raise HLSImagePipelineSynthesisConfigurationError(
                "HLS config generation requires missing keys: " + ", ".join(missing)
            )

    @staticmethod
    def _hls_design_config_keys(hls_design: Mapping[str, Any]) -> dict[str, str]:
        supported_keys = {"clock", "flow_target"}
        return {
            f"hls.{key}": str(value)
            for key, value in hls_design.items()
            if key in supported_keys
        }


@dataclass(frozen=True, slots=True)
class HLSImagePipelineSynthesisEvaluationStep(EvaluationStep):
    """Creates synthesis tasks for HLS image-processing pipelines."""

    id: str = "hls_image_pipeline_synthesis"
    depends_on: tuple[str, ...] = ()
    composer: Composer | None = None
    hls_tool: str = "v++"
    hls_config: Path | None = None
    work_dir_name: str = "work"
    top_function: str | None = None
    clock_period: float | None = None
    part: str | None = None
    config_defaults: Mapping[str, str] = field(default_factory=dict)
    config_overrides: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    task_type: type[EvaluationTask] = HLSImagePipelineSynthesisTask

    def create_task(
        self,
        individual: Individual,
        artifacts: Mapping[str, Artifact],
    ) -> EvaluationTask:
        """Create an HLS image pipeline synthesis task for an individual."""

        return HLSImagePipelineSynthesisTask(
            individual=individual,
            step_id=self.id,
            composer=self.composer,
            hls_tool=self.hls_tool,
            hls_config=self.hls_config,
            work_dir_name=self.work_dir_name,
            top_function=self.top_function,
            clock_period=self.clock_period,
            part=self.part,
            config_defaults=self.config_defaults,
            config_overrides=self.config_overrides,
            metadata={
                **dict(self.metadata),
                "input_artifacts": tuple(sorted(artifacts)),
            },
        )


__all__ = [
    "HLSImagePipelineSynthesisConfigurationError",
    "HLSImagePipelineSynthesisError",
    "HLSImagePipelineSynthesisTimeoutError",
    "HLSImagePipelineSynthesisEvaluationStep",
    "HLSImagePipelineSynthesisTask",
]
