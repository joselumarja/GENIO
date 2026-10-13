from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genio.composer import Composer
from genio.evaluation.task import EvaluationOutput, EvaluationTask, TaskContext
from genio_use_cases.image_pipeline.hls import HLSExecutionPackage
from genio_use_cases.image_pipeline.hls_reports import parse_hls_synthesis_report
from genio_use_cases.image_pipeline.hls_synthesis import HLSRTLArtifact


HLSConfigValue = str | Sequence[str]


class HLSImagePipelineSynthesisConfigurationError(ValueError):
    """Indicate invalid HLS task, package, target, or include configuration."""


class HLSImagePipelineSynthesisError(RuntimeError):
    """Report a nonzero exit from the external HLS synthesis command.

    Attributes:
        command: Fully rendered command passed to the execution context.
        returncode: Exit status returned by the HLS process.
        log_paths: Existing tool logs that may explain the failure.
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
        log_hint = f" See {log_paths[0]}." if log_paths else ""
        super().__init__(
            f"HLS command {command!r} failed with return code {returncode}.{log_hint}"
        )


@dataclass(frozen=True, slots=True)
class _MaterializedHLSPackage:
    """Hold the materialized HLS package and its composer metadata.

    Attributes:
        package_dir: Directory containing the generated source package.
        package_metadata: Metadata emitted by the HLS composer.
    """

    package_dir: Path
    package_metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _HLSRunResult:
    """Identify the output of a successful external HLS run.

    Attributes:
        work_dir: Vitis work directory containing reports and generated RTL.
    """

    work_dir: Path


@dataclass(frozen=True, slots=True)
class HLSImagePipelineSynthesisTask(EvaluationTask):
    """Synthesize an image pipeline into Vitis HLS reports and RTL.

    The composer must produce an :class:`HLSExecutionPackage`. Configuration is
    assembled with deterministic precedence from a base file, task defaults, the
    candidate's ``hls`` design domain, package metadata, explicit task fields,
    and task overrides. Package and execution include flags are appended last.

    The external tool is invoked with the ``v++ --compile --mode hls`` protocol.
    For completed invocations, standard streams, discovered tool
    logs, command metadata, the final config, parsed synthesis metrics, and
    generated Verilog/VHDL paths are persisted as available.

    Attributes:
        composer: Composer that generates the HLS source package.
        hls_tool: Executable used for HLS compilation.
        hls_config: Optional base ``.cfg`` file.
        work_dir_name: Relative Vitis work directory inside the package.
        top_function: Optional override for the synthesized top function.
        clock_period: Optional clock period override in nanoseconds.
        part: Optional FPGA part override.
        config_defaults: Config keys applied before design and package values.
        config_overrides: Config keys applied after explicit task fields.
        metadata: Additional execution metadata and backend resources.
    """

    uses_domains = ("pipeline", "hls")

    composer: Composer | None = None
    hls_tool: str = "v++"
    hls_config: Path | None = None
    work_dir_name: str = "work"
    top_function: str | None = None
    clock_period: float | None = None
    part: str | None = None
    config_defaults: Mapping[str, HLSConfigValue] = field(default_factory=dict)
    config_overrides: Mapping[str, HLSConfigValue] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def run(self, context: TaskContext) -> EvaluationOutput:
        """Compose, configure, and synthesize the candidate image pipeline.

        Args:
            context: Task workspace and execution-host resources.

        Returns:
            Origin-prefixed synthesis metrics and the generated HDL artifact.

        Raises:
            HLSImagePipelineSynthesisConfigurationError: If task configuration,
                required config keys, or backend include resources are invalid.
            TypeError: If the composer returns a non-HLS package.
            HLSImagePipelineSynthesisError: If ``v++`` returns a nonzero status.
            HLSReportParseError: If no usable synthesis XML report is found.
            RuntimeError: If no top function or generated RTL can be identified.
        """

        self._validate_configuration(context)

        package = self._compose_and_materialize(context)
        hls_config_path = self._prepare_hls_config(context, package)
        run_result = self._run_hls_tool(context, hls_config_path)

        metrics = self._parse_hls_synthesis_report(
            context,
            package,
            run_result,
        )
        rtl_artifact = self._collect_hls_rtl(
            context,
            package,
            run_result,
        )

        return EvaluationOutput(metrics=metrics, artifacts={"rtl_hls_synthesis": rtl_artifact})

    def _collect_hls_rtl(
        self,
        context: TaskContext,
        package: _MaterializedHLSPackage,
        run_result: _HLSRunResult,
    ) -> HLSRTLArtifact:
        """Discover generated HDL and describe it as an RTL artifact.

        Verilog/SystemVerilog files are collected from ``hls/syn/verilog`` and
        VHDL files from ``hls/syn/vhdl``. Interface and image-stream dimensions
        from package metadata are propagated for downstream integration.

        Raises:
            RuntimeError: If the top function is unknown or no HDL files exist.
        """

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
        configuration_metadata = {
            key: package.package_metadata[key]
            for key in (
                "interface",
                "vitis_version",
                "input_type",
                "output_type",
                "input_rows",
                "input_cols",
                "input_npc",
                "output_rows",
                "output_cols",
                "output_npc",
                "bus_width",
                "input_words",
                "output_words",
            )
            if package.package_metadata.get(key) is not None
        }
        data = {
            "origin": origin,
            "top_function": str(top_function),
            **configuration_metadata,
        }
        self._write_json(
            context.workspace / "artifacts" / f"hls_rtl_{origin}.json",
            {
                **data,
                "verilog_paths": tuple(str(path) for path in verilog_paths),
                "vhdl_paths": tuple(str(path) for path in vhdl_paths),
            },
        )
        files: dict[str, Path] = {}
        for language, directory, paths in (
            ("verilog", verilog_dir, verilog_paths),
            ("vhdl", vhdl_dir, vhdl_paths),
        ):
            for path in paths:
                key = f"{language}/{path.relative_to(directory).as_posix()}"
                if key in files:
                    raise ValueError(f"Duplicate HLS RTL artifact file key: {key!r}.")
                files[key] = path
        return HLSRTLArtifact(data=data, files=files)

    @staticmethod
    def _rtl_paths(directory: Path, patterns: tuple[str, ...]) -> tuple[Path, ...]:
        paths: set[Path] = set()
        for pattern in patterns:
            paths.update(directory.rglob(pattern))
        return tuple(sorted(path for path in paths if path.is_file()))

    def _parse_hls_synthesis_report(
        self,
        context: TaskContext,
        package: _MaterializedHLSPackage,
        run_result: _HLSRunResult,
    ) -> Mapping[str, float]:
        """Parse Vitis timing, latency, throughput, and resource metrics.

        The complete parsed report and source report paths are mirrored to
        ``hls_report_<origin>.json``.
        """

        top_function = self.top_function or package.package_metadata.get("top_function")
        parsed_report = parse_hls_synthesis_report(
            run_result.work_dir,
            top_function=str(top_function) if top_function is not None else None,
        )
        self._write_json(
            context.workspace / "artifacts" / f"hls_report_{parsed_report.origin}.json",
            {
                "origin": parsed_report.origin,
                "report_paths": tuple(str(path) for path in parsed_report.report_paths),
                "metrics": dict(parsed_report.metrics),
                "metadata": dict(parsed_report.metadata),
            },
        )
        return {
            f"{parsed_report.origin}.{key}": value
            for key, value in parsed_report.metrics.items()
        }

    def _run_hls_tool(
        self,
        context: TaskContext,
        hls_config_path: Path,
    ) -> _HLSRunResult:
        """Run Vitis HLS while persisting handled command outcomes.

        Both stdout/stderr logs and ``hls_run_metadata.json`` are written on
        success and nonzero exit. Whole-task timeouts are handled by the supervisor.

        Args:
            context: Task workspace containing the package, logs, and artifacts.
            hls_config_path: Final config file located in the package directory.

        Returns:
            The successful run's Vitis work directory.

        Raises:
            HLSImagePipelineSynthesisError: If the process exits unsuccessfully.
        """

        package_dir = context.workspace / "package"
        work_dir = package_dir / self.work_dir_name
        work_dir.mkdir(parents=True, exist_ok=True)
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
        log_dir = context.workspace / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "hls_stdout.log"
        stderr_path = log_dir / "hls_stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            result = subprocess.run(
                command,
                cwd=package_dir,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
            )
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        tool_log_paths = self._tool_log_paths(work_dir)
        self._write_json(
            context.workspace / "artifacts" / "hls_run_metadata.json",
            {
                "command": command,
                "status": "success" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "cwd": str(package_dir),
                "work_dir": str(work_dir),
                "hls_config": str(hls_config_path),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "tool_log_paths": tuple(str(path) for path in tool_log_paths),
            },
        )
        if result.returncode != 0:
            error = HLSImagePipelineSynthesisError(
                command=command,
                returncode=result.returncode,
                log_paths=tool_log_paths,
            )
            if stderr or stdout:
                error.add_note(f"HLS output: {(stderr or stdout)[-2000:]}")
            raise error
        return _HLSRunResult(work_dir=work_dir)

    @staticmethod
    def _write_json(path: Path, data: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        return path

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
        context: TaskContext,
        package: _MaterializedHLSPackage,
    ) -> Path:
        """Merge all HLS config sources and persist the effective config.

        Precedence is base config, defaults, candidate ``hls`` design, package
        top/source metadata, explicit part and clock fields, then overrides.
        Execution resource include flags are appended after those sources. The resulting
        ``hls_config.cfg`` must define ``[hls] syn.file`` and ``syn.top`` and is
        accompanied by applied-value metadata and a SHA-256 content digest.

        Returns:
            Path to the effective config in the materialized package directory.

        Raises:
            HLSImagePipelineSynthesisConfigurationError: If required keys or
                backend include resources are missing or invalid.
        """

        package_dir = package.package_dir
        package_config_path = package_dir / "hls_config.cfg"

        # Precedence: base config < defaults < HLS design < package values
        # < explicit task fields < overrides; backend include flags are appended last.
        if self.hls_config is not None:
            config_path = Path(self.hls_config)
            if not config_path.is_absolute():
                config_path = context.workspace.parent.parent / config_path
            config = config_path.read_text(encoding="utf-8")
        elif package_config_path.exists():
            config = package_config_path.read_text(encoding="utf-8")
        else:
            config = ""

        applied_values: dict[str, Any] = {}
        config = self._apply_config_mapping(
            config,
            self.config_defaults,
            applied_values,
        )
        hls_design = self.individual.values.get("hls", {})
        if isinstance(hls_design, Mapping):
            config_keys = self._hls_design_config_keys(hls_design)
            config = self._apply_config_mapping(
                config,
                config_keys,
                applied_values,
            )

        package_metadata = package.package_metadata
        top_function = self.top_function or package_metadata.get("top_function")
        if top_function is not None:
            config = self._replace_config_value(
                config,
                "hls.syn.top",
                str(top_function),
            )
            applied_values["hls.syn.top"] = str(top_function)
        source_files = self._metadata_tuple(package_metadata, "source_files")
        if source_files:
            rendered_source_files = tuple(str(value) for value in source_files)
            config = self._replace_config_values(
                config,
                "hls.syn.file",
                rendered_source_files,
            )
            applied_values["hls.syn.file"] = rendered_source_files

        if self.part is not None:
            config = self._replace_config_value(config, "part", self.part)
            applied_values["part"] = self.part
        if self.clock_period is not None:
            config = self._replace_config_value(
                config,
                "hls.clock",
                str(self.clock_period),
            )
            applied_values["hls.clock"] = str(self.clock_period)

        config = self._apply_config_mapping(
            config,
            self.config_overrides,
            applied_values,
        )
        config = self._apply_package_backend_config(
            context,
            package_dir,
            package_metadata,
            config,
            applied_values,
        )

        for key in ("syn.file", "syn.top"):
            if not self._config_values(config, f"hls.{key}"):
                raise HLSImagePipelineSynthesisConfigurationError(
                    f"Final hls_config.cfg is missing required [hls] key: {key}."
                )

        package_config_path.write_text(config, encoding="utf-8")
        self._write_json(
            package_dir / "hls_config_metadata.json",
            {
                "path": str(package_config_path),
                "applied_values": applied_values,
                "content_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
            },
        )
        return package_config_path

    def _apply_package_backend_config(
        self,
        context: TaskContext,
        package_dir: Path,
        package_metadata: Mapping[str, Any],
        config: str,
        applied_values: dict[str, Any],
    ) -> str:
        """Append package and execution-host include paths as C flags.

        Packages declaring ``vitis_libraries_path`` as a required backend resource
        resolve the Vitis Vision ``vision/L1/include`` directory from execution
        resources. Additional paths use ``resources['hls_include_paths']`` and are
        validated as execution-host directories.
        """

        include_flags = [
            self._include_cflag(include_dir)
            for include_dir in self._package_include_dirs(package_dir, package_metadata)
        ]

        required_resources = self._metadata_tuple(package_metadata, "required_backend_resources")
        if "vitis_libraries_path" in required_resources:
            vitis_libraries_path = context.resources.get("vitis_libraries_path")
            if not vitis_libraries_path:
                raise HLSImagePipelineSynthesisConfigurationError(
                    "Task context must provide resources['vitis_libraries_path'] "
                    "for Vitis Vision HLS packages."
                )
            vision_include_path = (
                Path(str(vitis_libraries_path)).expanduser().resolve()
                / "vision" / "L1" / "include"
            )
            if not vision_include_path.exists():
                raise HLSImagePipelineSynthesisConfigurationError(
                    "Vitis Vision include path does not exist: "
                    f"{vision_include_path}."
                )
            if not vision_include_path.is_dir():
                raise HLSImagePipelineSynthesisConfigurationError(
                    "Vitis Vision include path is not a directory: "
                    f"{vision_include_path}."
                )
            include_flags.append(self._include_cflag(vision_include_path))

        for include_path in self._execution_include_paths(context.resources):
            resolved_include_path = Path(include_path).expanduser().resolve()
            if not resolved_include_path.exists():
                raise HLSImagePipelineSynthesisConfigurationError(
                    f"Additional HLS include path does not exist: {resolved_include_path}."
                )
            if not resolved_include_path.is_dir():
                raise HLSImagePipelineSynthesisConfigurationError(
                    "Additional HLS include path is not a directory: "
                    f"{resolved_include_path}."
                )
            include_flag = self._include_cflag(resolved_include_path)
            if include_flag not in include_flags:
                include_flags.append(include_flag)

        return self._append_hls_cflags(config, include_flags, applied_values)

    @staticmethod
    def _execution_include_paths(metadata: Mapping[str, Any]) -> tuple[str, ...]:
        """Normalize and deduplicate execution-level HLS include path values.

        Raises:
            HLSImagePipelineSynthesisConfigurationError: If the metadata value is
                not a path or sequence of nonempty paths.
        """

        values = metadata.get("hls_include_paths", ())
        if isinstance(values, (str, Path)):
            values = (values,)
        elif not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
            raise HLSImagePipelineSynthesisConfigurationError(
                "metadata['hls_include_paths'] must be a path or a sequence of paths."
            )

        paths: list[str] = []
        for value in values:
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise HLSImagePipelineSynthesisConfigurationError(
                    "metadata['hls_include_paths'] must contain only non-empty paths."
                )
            path = str(value)
            if path not in paths:
                paths.append(path)
        return tuple(paths)

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
    def _append_hls_cflags(
        cls,
        config: str,
        flags: list[str],
        applied_values: dict[str, Any],
    ) -> str:
        """Append unique include flags to the last effective ``hls.syn.cflags``.

        Existing flags are shell-split for comparison, while newly rendered flag
        strings retain their original quoting in the generated config.
        """

        if not flags:
            return config
        existing_values = cls._config_values(config, "hls.syn.cflags")
        existing = existing_values[-1].strip() if existing_values else ""
        normalized_cflags = shlex.split(existing, posix=True) if existing else []
        new_flags: list[str] = []
        for flag in flags:
            normalized_flag = shlex.split(flag, posix=True)
            if len(normalized_flag) != 1:
                raise HLSImagePipelineSynthesisConfigurationError(
                    f"Invalid HLS include flag: {flag!r}."
                )
            if normalized_flag[0] not in normalized_cflags:
                normalized_cflags.append(normalized_flag[0])
                new_flags.append(flag)
        rendered_cflags = " ".join(
            value for value in (existing, *new_flags) if value
        )
        applied_values["hls.syn.cflags"] = rendered_cflags
        return cls._replace_config_value(
            config,
            "hls.syn.cflags",
            rendered_cflags,
        )

    @staticmethod
    def _include_cflag(path: str | Path) -> str:
        value = str(path)
        if any(character.isspace() for character in value):
            return f'-I"{value}"'
        return f"-I{value}"

    def _compose_and_materialize(self, context: TaskContext) -> _MaterializedHLSPackage:
        """Compose HLS sources and persist composition metadata.

        Raises:
            TypeError: If the configured composer does not return an
                :class:`HLSExecutionPackage`.
        """

        assert self.composer is not None
        package = self.composer.compose(self.individual)
        if not isinstance(package, HLSExecutionPackage):
            raise TypeError(
                "HLSImagePipelineSynthesisTask requires composer.compose() to return "
                "HLSExecutionPackage."
            )

        package_dir = package.materialize(context.workspace / "package")
        metadata = {
            "entrypoint": package.entrypoint,
            "files": tuple(sorted(package.files)),
            "package_metadata": dict(package.metadata),
        }
        self._write_json(package_dir / "composition_metadata.json", metadata)

        return _MaterializedHLSPackage(
            package_dir=package_dir,
            package_metadata=dict(package.metadata),
        )

    @classmethod
    def _apply_config_mapping(
        cls,
        config: str,
        values: Mapping[str, Any],
        applied_values: dict[str, Any],
    ) -> str:
        """Apply scalar or repeated config values and record their rendered form."""

        for key, value in values.items():
            if isinstance(value, Sequence) and not isinstance(
                value,
                (str, bytes, bytearray),
            ):
                rendered_values = tuple(str(item) for item in value)
                config = cls._replace_config_values(config, key, rendered_values)
                applied_values[key] = rendered_values
            else:
                rendered_value = str(value)
                config = cls._replace_config_value(config, key, rendered_value)
                applied_values[key] = rendered_value
        return config

    @classmethod
    def _replace_config_value(cls, content: str, key: str, value: str) -> str:
        return cls._replace_config_values(content, key, (value,))

    @classmethod
    def _replace_config_values(
        cls,
        content: str,
        key: str,
        values: Sequence[str],
    ) -> str:
        """Replace all occurrences of a config key or insert it in its section.

        Qualified keys use ``section.option`` syntax. Existing option positions
        are reused in order, excess occurrences are removed, and excess values
        are inserted after the last occurrence. The rendered result always ends
        in exactly one newline.
        """

        section_name, option = cls._split_config_key(key)
        lines = content.splitlines()
        matching_indexes = cls._config_key_indexes(lines, section_name, option)
        rendered_lines = [f"{option}={value}" for value in values]

        if matching_indexes:
            matching_set = set(matching_indexes)
            last_index = matching_indexes[-1]
            rendered_index = 0
            replaced: list[str] = []
            for index, line in enumerate(lines):
                if index in matching_set:
                    if rendered_index < len(rendered_lines):
                        replaced.append(rendered_lines[rendered_index])
                        rendered_index += 1
                    if index == last_index:
                        replaced.extend(rendered_lines[rendered_index:])
                else:
                    replaced.append(line)
            lines = replaced
        elif rendered_lines:
            lines = cls._insert_config_lines(lines, section_name, rendered_lines)

        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def _config_values(cls, content: str, key: str) -> tuple[str, ...]:
        section_name, option = cls._split_config_key(key)
        lines = content.splitlines()
        return tuple(
            cls._config_line_value(lines[index])
            for index in cls._config_key_indexes(lines, section_name, option)
        )

    @staticmethod
    def _split_config_key(key: str) -> tuple[str | None, str]:
        """Split an optional ``section.option`` key and reject empty components."""

        if "." not in key:
            if not key:
                raise HLSImagePipelineSynthesisConfigurationError(
                    "HLS config key cannot be empty."
                )
            return None, key
        section_name, option = key.split(".", 1)
        if not section_name or not option:
            raise HLSImagePipelineSynthesisConfigurationError(
                f"Invalid qualified HLS config key: {key!r}."
            )
        return section_name, option

    @classmethod
    def _config_key_indexes(
        cls,
        lines: Sequence[str],
        section_name: str | None,
        option: str,
    ) -> list[int]:
        indexes: list[int] = []
        current_section: str | None = None
        for index, line in enumerate(lines):
            parsed_section = cls._config_section(line)
            if parsed_section is not None:
                current_section = parsed_section
                continue
            if current_section == section_name and cls._config_line_key(line) == option:
                indexes.append(index)
        return indexes

    @classmethod
    def _insert_config_lines(
        cls,
        lines: list[str],
        section_name: str | None,
        rendered_lines: list[str],
    ) -> list[str]:
        if section_name is None:
            section_end = next(
                (index for index, line in enumerate(lines) if cls._config_section(line)),
                len(lines),
            )
            option_indexes = [
                index
                for index in range(section_end)
                if cls._config_line_key(lines[index]) is not None
            ]
            insertion_index = option_indexes[-1] + 1 if option_indexes else 0
            return (
                lines[:insertion_index]
                + rendered_lines
                + lines[insertion_index:]
            )

        section_start = next(
            (
                index
                for index, line in enumerate(lines)
                if cls._config_section(line) == section_name
            ),
            None,
        )
        if section_start is None:
            appended = list(lines)
            if appended and appended[-1].strip():
                appended.append("")
            appended.extend((f"[{section_name}]", *rendered_lines))
            return appended

        insertion_index = next(
            (
                index
                for index in range(section_start + 1, len(lines))
                if cls._config_section(lines[index]) is not None
            ),
            len(lines),
        )
        option_indexes = [
            index
            for index in range(section_start + 1, insertion_index)
            if cls._config_line_key(lines[index]) is not None
        ]
        insertion_index = (
            option_indexes[-1] + 1 if option_indexes else section_start + 1
        )
        return lines[:insertion_index] + rendered_lines + lines[insertion_index:]

    @staticmethod
    def _config_section(line: str) -> str | None:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return stripped[1:-1].strip()
        return None

    @staticmethod
    def _config_line_key(line: str) -> str | None:
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith(("#", ";", "["))
            or "=" not in stripped
        ):
            return None
        return stripped.split("=", 1)[0].strip()

    @staticmethod
    def _config_line_value(line: str) -> str:
        return line.split("=", 1)[1].strip()

    def _validate_configuration(self, context: TaskContext) -> None:
        """Validate tools, paths, design shape, and config prerequisites.

        Required generated-config keys are checked before composition only when
        the task has a known config source. A package-provided ``hls_config.cfg``
        is intentionally validated later after materialization.
        """

        if self.composer is None:
            raise HLSImagePipelineSynthesisConfigurationError(
                "HLSImagePipelineSynthesisTask requires a composer before synthesis can run."
            )

        if not self.hls_tool:
            raise HLSImagePipelineSynthesisConfigurationError("hls_tool cannot be empty.")

        if not self.work_dir_name:
            raise HLSImagePipelineSynthesisConfigurationError(
                "work_dir_name cannot be empty."
            )
        work_dir = Path(self.work_dir_name)
        if work_dir.is_absolute() or ".." in work_dir.parts:
            raise HLSImagePipelineSynthesisConfigurationError(
                "work_dir_name must be a relative path inside the task workspace."
            )

        hls_design = self.individual.values.get("hls", {})
        if hls_design and not isinstance(hls_design, Mapping):
            raise HLSImagePipelineSynthesisConfigurationError(
                "Individual values domain 'hls' must be a mapping."
            )

        if self.hls_config is not None:
            hls_config = Path(self.hls_config)
            if not hls_config.is_absolute():
                hls_config = context.workspace.parent.parent / hls_config
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


__all__ = [
    "HLSImagePipelineSynthesisConfigurationError",
    "HLSImagePipelineSynthesisError",
    "HLSImagePipelineSynthesisTask",
]
