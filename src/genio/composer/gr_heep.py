from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from genio.artifacts import HLSRTLArtifact
from genio.composer.base import Composer, ComposerError, ExecutionPackage
from genio.core import Individual


_TOKEN_PATTERN = re.compile(r"@[A-Z][A-Z0-9_]*@")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class GRHeepConfigurationPackage(ExecutionPackage):
    """Portable overlay containing files owned by GENIO in a GR-HEEP checkout."""

    entrypoint: str = "config/mcu-gen-config.py"
    files: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _ALLOWED_PATHS = ("config/mcu-gen-config.py",)
    _ALLOWED_PREFIXES = (
        "hw/vendor/safa/",
        "sw/applications/",
    )

    def materialize(self, target_dir: str | Path) -> Path:
        """Write the validated overlay beneath target_dir."""

        package_dir = Path(target_dir)
        package_dir.mkdir(parents=True, exist_ok=True)
        package_root = package_dir.resolve()
        for relative_path, content in self.files.items():
            destination = self._destination(package_dir, relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                destination.parent.resolve().relative_to(package_root)
            except ValueError as exc:
                raise ComposerError(
                    f"GR-HEEP overlay path escapes through a symlink: {relative_path!r}."
                ) from exc
            if destination.is_symlink():
                raise ComposerError(
                    f"GR-HEEP overlay destination is a symlink: {relative_path!r}."
                )
            destination.write_text(content, encoding="utf-8")

        if self.metadata:
            metadata_path = package_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(dict(self.metadata), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return package_dir

    @staticmethod
    def _destination(package_dir: Path, relative_path: str) -> Path:
        path = PurePosixPath(relative_path)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ComposerError(f"Invalid GR-HEEP overlay path: {relative_path!r}.")
        normalized = path.as_posix()
        if normalized not in GRHeepConfigurationPackage._ALLOWED_PATHS and not any(
            normalized.startswith(prefix)
            for prefix in GRHeepConfigurationPackage._ALLOWED_PREFIXES
        ):
            raise ComposerError(
                f"GR-HEEP overlay path is not owned by GENIO: {relative_path!r}."
            )
        return package_dir.joinpath(*path.parts)


class GRHeepConfigurationComposer(Composer):
    """Render GR-HEEP configuration and application overlay files."""

    MEMORY_PLACEMENTS = {
        "shared_data": {
            "CODE_SECTION_END": "0x00018000",
            "DATA_SECTION_START": "0x00018000",
        },
        "separate_code_input_output": {
            "CODE_SECTION_END": "0x00010000",
            "DATA_SECTION_START": "0x00010000",
        },
        "input_output_interleaved": {
            "CODE_SECTION_END": "0x00010000",
            "DATA_SECTION_START": "0x00010000",
        },
    }
    DEFAULT_CONFIGURATION = {
        "BUS_TYPE": "NtoM",
        "CPU": "cv32e40px",
        "RAM_BANKS": "[32] * 6",
        "INTERLEAVED_BANK_COUNT": "4",
        "INTERLEAVED_BANK_SIZE": "16",
        "CODE_SECTION_END": "0x00018000",
        "DATA_SECTION_START": "0x00018000",
        "DMA_NUM_CHANNELS": "4",
        "DMA_NUM_MASTER_PORTS": "2",
        "DMA_NUM_CHANNELS_PER_MASTER_PORT": "2",
        "DMA_FIFO_DEPTH": "4",
        "DMA_ADDR_MODE": "yes",
        "DMA_SUBADDR_MODE": "yes",
        "DMA_ZERO_PADDING": "yes",
        "SAFA_OFFSET": "0x00000000",
        "SAFA_LENGTH": "0x00001000",
        "HW_FIFO_CHANNELS": "[0]",
        "ACCELERATOR_FIFO_DEPTH": "16",
        "ACCELERATOR_FIFO_ALMOST_FULL_MARGIN": "4",
    }
    DEFAULT_APPLICATION = {
        "INPUT_ROWS": "0",
        "INPUT_COLS": "0",
        "OUTPUT_ROWS": "0",
        "OUTPUT_COLS": "0",
        "INPUT_WORDS": "0",
        "OUTPUT_WORDS": "0",
        "DMA_CHANNEL": "0",
        "TIMEOUT_CYCLES": "0",
        "IMAGE_WORDS": "[0 ... GENIO_INPUT_WORDS - 1] = 0x6090c030u,",
    }

    def __init__(
        self,
        stages_definitions_path: str | Path,
        *,
        templates_path: str | Path,
        application_name: str = "genio_target",
        configuration_defaults: Mapping[str, Any] | None = None,
        application_defaults: Mapping[str, Any] | None = None,
        parameter_bindings: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(stages_definitions_path)
        if not _IDENTIFIER_PATTERN.fullmatch(application_name):
            raise ComposerError(
                f"Invalid GR-HEEP application name: {application_name!r}."
            )
        self.templates_path = Path(templates_path)
        self.application_name = application_name
        self.configuration_defaults = {
            **self.DEFAULT_CONFIGURATION,
            **dict(configuration_defaults or {}),
        }
        self.application_defaults = {
            **self.DEFAULT_APPLICATION,
            **dict(application_defaults or {}),
        }
        self.parameter_bindings = dict(parameter_bindings or {})
        unknown_binding_tokens = sorted(
            {
                self._token_name(token)
                for token in self.parameter_bindings.values()
            }.difference(self.configuration_defaults)
        )
        if unknown_binding_tokens:
            raise ComposerError(
                "GR-HEEP parameter bindings reference unknown tokens: "
                f"{unknown_binding_tokens}."
            )
        if str(self.configuration_defaults["HW_FIFO_CHANNELS"]).replace(" ", "") != "[0]":
            raise ComposerError("Current GR-HEEP integration requires HW_FIFO_CHANNELS=[0].")
        if str(self.application_defaults["DMA_CHANNEL"]) != "0":
            raise ComposerError("Current SAFA application wiring requires DMA_CHANNEL=0.")

    def compose(self, individual: Individual) -> GRHeepConfigurationPackage:
        """Render the individual-dependent GR-HEEP overlay."""

        system_design = individual.design.get("system", {})
        if not isinstance(system_design, Mapping):
            raise ComposerError("Individual design domain 'system' must be a mapping.")

        configuration = dict(self.configuration_defaults)
        native_bindings = {
            "cpu": "CPU",
            "bus_type": "BUS_TYPE",
            "dma_fifo_depth": "DMA_FIFO_DEPTH",
            "accelerator_fifo_depth": "ACCELERATOR_FIFO_DEPTH",
            "accelerator_fifo_almost_full_margin": (
                "ACCELERATOR_FIFO_ALMOST_FULL_MARGIN"
            ),
        }
        for parameter, token in native_bindings.items():
            if parameter in system_design:
                configuration[token] = self._configuration_binding_value(
                    token,
                    system_design[parameter],
                )
        for parameter, token in self.parameter_bindings.items():
            if parameter in system_design:
                token_name = self._token_name(token)
                configuration[token_name] = self._configuration_binding_value(
                    token_name,
                    system_design[parameter],
                )
        configuration.update(self._memory_configuration(system_design))
        self._validate_accelerator_fifo_configuration(configuration)

        files = {
            "config/mcu-gen-config.py": self._render_template(
                "config/mcu-gen-config.py.tpl",
                configuration,
            ),
            **self._render_application(self.application_defaults),
        }
        return GRHeepConfigurationPackage(
            files=files,
            metadata={
                **self.artifact_metadata(individual),
                "package_type": "gr_heep_overlay",
                "application_name": self.application_name,
                "system_design": dict(system_design),
                "unbound_system_parameters": sorted(
                    set(system_design).difference(
                        native_bindings,
                        self.parameter_bindings,
                        {
                            "memory_total_kib",
                            "memory_bank_size_kib",
                            "memory_interleaved_ratio",
                            "memory_placement",
                        },
                    )
                ),
                "rendered_configuration": {
                    key: str(value) for key, value in sorted(configuration.items())
                },
            },
        )

    def render_safa_integration(
        self,
        *,
        top_function: str,
        rtl_files: Sequence[str | Path],
        configuration: Mapping[str, Any] | None = None,
    ) -> Mapping[str, str]:
        """Render files that bind one generated HLS module into SAFA."""

        if not _IDENTIFIER_PATTERN.fullmatch(top_function):
            raise ComposerError(f"Invalid HLS top module: {top_function!r}.")
        normalized_files = self._rtl_files(rtl_files)
        rendered_file_list = "\n".join(
            self._fusesoc_file_entry(path) for path in normalized_files
        )
        values = {
            **self.configuration_defaults,
            **dict(configuration or {}),
            "HLS_TOP_MODULE": top_function,
            "HLS_RTL_FILES": rendered_file_list,
        }
        return {
            "hw/vendor/safa/rtl/safa_wrapper.sv": self._render_template(
                "safa/safa_wrapper.sv.tpl",
                values,
            ),
            "hw/vendor/safa/hls_ip/hls_accelerator_component.core": (
                self._render_template(
                    "safa/hls_accelerator_component.core.tpl",
                    values,
                )
            ),
        }

    def render_application_config(self, values: Mapping[str, Any]) -> str:
        """Render the generated application header from HLS/runtime metadata."""

        return self._render_template(
            f"applications/{self.application_name}/genio_app_config.h.tpl",
            {**self.application_defaults, **dict(values)},
        )

    def render_hls_artifact_overlay(
        self,
        artifact: HLSRTLArtifact,
        *,
        image_path: str | Path | None = None,
        configuration: Mapping[str, Any] | None = None,
    ) -> Mapping[str, str]:
        """Copy one SAFA-compatible HLS artifact into overlay-owned files."""

        if artifact.metadata.get("interface") != "safa_fifo":
            raise ComposerError("GR-HEEP requires an HLS artifact using 'safa_fifo'.")
        source_paths = artifact.verilog_paths
        if not source_paths:
            raise ComposerError("HLS artifact does not contain Verilog RTL files.")

        rtl_names: list[str] = []
        overlay: dict[str, str] = {}
        for source_path in source_paths:
            name = self._portable_rtl_name(source_path.name)
            if name in rtl_names:
                raise ComposerError(f"Duplicate HLS RTL basename: {name!r}.")
            rtl_names.append(name)
            overlay[f"hw/vendor/safa/hls_ip/{name}"] = source_path.read_text(
                encoding="utf-8"
            )

        top_filename = f"{artifact.top_function}.v"
        if top_filename not in rtl_names:
            raise ComposerError(
                f"HLS artifact is missing its top-level RTL file {top_filename!r}."
            )
        overlay.update(
            self.render_safa_integration(
                top_function=artifact.top_function,
                rtl_files=rtl_names,
                configuration=configuration,
            )
        )
        overlay[
            f"sw/applications/{self.application_name}/genio_app_config.h"
        ] = self.render_application_config_from_artifact(artifact)
        if image_path is not None:
            overlay[f"sw/applications/{self.application_name}/main.h"] = (
                self.render_application_image_header(artifact, image_path)
            )
        return overlay

    def render_application_image_header(
        self,
        artifact: HLSRTLArtifact,
        image_path: str | Path,
    ) -> str:
        """Render the firmware input array from one functional dataset image."""

        try:
            import cv2 as cv
        except ImportError as exc:
            raise ComposerError("Rendering a firmware image requires OpenCV.") from exc

        path = Path(image_path)
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            raise ComposerError(f"Cannot read firmware input image: {path}.")

        metadata = artifact.metadata
        try:
            rows = int(metadata["input_rows"])
            cols = int(metadata["input_cols"])
            input_words = int(metadata["input_words"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ComposerError("HLS artifact has invalid input image metadata.") from exc
        if rows <= 0 or cols <= 0 or input_words <= 0:
            raise ComposerError("HLS input image dimensions and word count must be positive.")

        resized = cv.resize(image, (cols, rows), interpolation=cv.INTER_AREA)
        payload = resized.tobytes(order="C")
        expected_bytes = input_words * 4
        if len(payload) > expected_bytes:
            raise ComposerError(
                f"Input image needs {len(payload)} bytes but HLS expects "
                f"{expected_bytes}."
            )
        payload += bytes(expected_bytes - len(payload))
        words = [
            int.from_bytes(payload[offset : offset + 4], "little")
            for offset in range(0, expected_bytes, 4)
        ]
        image_words = "\n    ".join(
            ", ".join(f"0x{word:08x}u" for word in words[index : index + 8]) + ","
            for index in range(0, len(words), 8)
        )
        template = self.templates_path / "applications" / self.application_name / "main.h.tpl"
        return self._replace_tokens(
            template.read_text(encoding="utf-8"),
            {**self.application_defaults, "IMAGE_WORDS": image_words},
            source_path=template,
        )

    def render_application_config_from_artifact(
        self,
        artifact: HLSRTLArtifact,
    ) -> str:
        """Render application dimensions and SAFA word counts from an HLS artifact."""

        metadata = artifact.metadata
        required = (
            "input_rows",
            "input_cols",
            "output_rows",
            "output_cols",
            "input_words",
            "output_words",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ComposerError(
                "HLS artifact is missing application metadata: " + ", ".join(missing)
            )
        return self.render_application_config(
            {
                "INPUT_ROWS": metadata["input_rows"],
                "INPUT_COLS": metadata["input_cols"],
                "OUTPUT_ROWS": metadata["output_rows"],
                "OUTPUT_COLS": metadata["output_cols"],
                "INPUT_WORDS": metadata["input_words"],
                "OUTPUT_WORDS": metadata["output_words"],
            }
        )

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return templates and mappings that determine overlay semantics."""

        return {
            **Composer.checkpoint_signature(self),
            "templates_path": str(self.templates_path.expanduser().resolve()),
            "template_tree": self._directory_fingerprint(self.templates_path),
            "application_name": self.application_name,
            "configuration_defaults": dict(self.configuration_defaults),
            "application_defaults": dict(self.application_defaults),
            "parameter_bindings": dict(self.parameter_bindings),
        }

    def _render_application(self, values: Mapping[str, Any]) -> dict[str, str]:
        application_dir = self.templates_path / "applications" / self.application_name
        if not application_dir.is_dir():
            raise ComposerError(
                f"GR-HEEP application template does not exist: {application_dir}."
            )

        rendered: dict[str, str] = {}
        target_root = PurePosixPath("sw/applications") / self.application_name
        for source_path in sorted(path for path in application_dir.rglob("*") if path.is_file()):
            relative_path = source_path.relative_to(application_dir)
            target_path = target_root.joinpath(*relative_path.parts)
            content = source_path.read_text(encoding="utf-8")
            if source_path.suffix == ".tpl":
                target_path = target_path.with_suffix("")
                content = self._replace_tokens(content, values, source_path=source_path)
            elif _TOKEN_PATTERN.search(content):
                raise ComposerError(
                    f"Static GR-HEEP template file contains unresolved tokens: "
                    f"{source_path}."
                )
            rendered[target_path.as_posix()] = content
        return rendered

    def _render_template(
        self,
        relative_path: str,
        values: Mapping[str, Any],
    ) -> str:
        template_path = self.templates_path / relative_path
        if not template_path.is_file():
            raise ComposerError(f"GR-HEEP template does not exist: {template_path}.")
        return self._replace_tokens(
            template_path.read_text(encoding="utf-8"),
            values,
            source_path=template_path,
        )

    @staticmethod
    def _replace_tokens(
        content: str,
        values: Mapping[str, Any],
        *,
        source_path: Path,
    ) -> str:
        replacements = {
            f"@{GRHeepConfigurationComposer._token_name(key)}@": str(value)
            for key, value in values.items()
        }
        rendered = _TOKEN_PATTERN.sub(
            lambda match: replacements.get(match.group(0), match.group(0)),
            content,
        )
        unresolved = sorted(set(_TOKEN_PATTERN.findall(rendered)))
        if unresolved:
            raise ComposerError(
                f"Unresolved GR-HEEP tokens in {source_path}: {unresolved}."
            )
        return rendered

    @staticmethod
    def _token_name(value: str) -> str:
        return value.strip().removeprefix("@").removesuffix("@")

    @staticmethod
    def _rtl_files(rtl_files: Sequence[str | Path]) -> tuple[PurePosixPath, ...]:
        normalized: list[PurePosixPath] = []
        for value in rtl_files:
            path = PurePosixPath(str(value))
            if path.is_absolute() or not path.parts or ".." in path.parts:
                raise ComposerError(f"Invalid HLS RTL path: {str(value)!r}.")
            if len(path.parts) != 1:
                raise ComposerError(
                    f"HLS RTL files must use flat portable names: {str(value)!r}."
                )
            GRHeepConfigurationComposer._portable_rtl_name(path.name)
            if path.suffix not in {".v", ".sv", ".vh"}:
                raise ComposerError(f"Unsupported HLS RTL file: {str(value)!r}.")
            if path in normalized:
                raise ComposerError(f"Duplicate HLS RTL file: {path.as_posix()!r}.")
            normalized.append(path)
        if not normalized:
            raise ComposerError("SAFA integration requires at least one HLS RTL file.")
        return tuple(normalized)

    @staticmethod
    def _portable_rtl_name(value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or "\\" in value:
            raise ComposerError(f"Invalid portable HLS RTL filename: {value!r}.")
        if PurePosixPath(value).suffix not in {".v", ".sv", ".vh"}:
            raise ComposerError(f"Unsupported HLS RTL file: {value!r}.")
        return value

    @staticmethod
    def _configuration_binding_value(token: str, value: Any) -> str:
        rendered = str(value)
        if not rendered or any(character in rendered for character in "\r\n@"):
            raise ComposerError(
                f"Invalid value for GR-HEEP token {token!r}: {rendered!r}."
            )
        if token in {"CPU", "BUS_TYPE"} and not _IDENTIFIER_PATTERN.fullmatch(
            rendered
        ):
            raise ComposerError(
                f"GR-HEEP token {token!r} requires an identifier value."
            )
        return rendered

    @staticmethod
    def _validate_accelerator_fifo_configuration(
        configuration: Mapping[str, Any],
    ) -> None:
        try:
            depth = int(configuration["ACCELERATOR_FIFO_DEPTH"])
            margin = int(configuration["ACCELERATOR_FIFO_ALMOST_FULL_MARGIN"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ComposerError("SAFA FIFO parameters must be integers.") from exc
        if depth <= 0:
            raise ComposerError("accelerator_fifo_depth must be positive.")
        if margin <= 0 or margin >= depth:
            raise ComposerError(
                "accelerator_fifo_almost_full_margin must be positive and smaller "
                "than accelerator_fifo_depth."
            )

    @classmethod
    def _memory_configuration(cls, system_design: Mapping[str, Any]) -> dict[str, str]:
        """Resolve declarative memory settings into mcu-gen template values."""

        memory_keys = (
            "memory_total_kib",
            "memory_bank_size_kib",
            "memory_interleaved_ratio",
        )
        placement = system_design.get("memory_placement")
        values = tuple(system_design.get(key) for key in memory_keys)
        if all(value is None for value in values) and placement is None:
            return {}
        if any(value is None for value in values) or placement is None:
            raise ComposerError(
                "GR-HEEP memory configuration requires memory_total_kib, "
                "memory_bank_size_kib, memory_interleaved_ratio and memory_placement."
            )
        if placement not in cls.MEMORY_PLACEMENTS:
            raise ComposerError(
                f"Unsupported GR-HEEP memory placement: {placement!r}."
            )
        try:
            total_kib, bank_size_kib, interleaved_ratio = (
                int(value) for value in values
            )
        except (TypeError, ValueError) as exc:
            raise ComposerError("GR-HEEP memory parameters must be integers.") from exc
        if total_kib <= 0 or bank_size_kib <= 0:
            raise ComposerError("Memory total and bank size must be positive.")
        if not 0 <= interleaved_ratio < 100:
            raise ComposerError(
                "memory_interleaved_ratio must be between 0 and 99; at least one "
                "continuous bank is required."
            )
        interleaved_numerator = total_kib * interleaved_ratio
        if interleaved_numerator % 100:
            raise ComposerError(
                "memory_total_kib * memory_interleaved_ratio must be divisible by 100."
            )
        interleaved_kib = interleaved_numerator // 100
        continuous_kib = total_kib - interleaved_kib
        if continuous_kib % bank_size_kib or interleaved_kib % bank_size_kib:
            raise ComposerError(
                "Continuous and interleaved capacities must be divisible by "
                "memory_bank_size_kib."
            )
        continuous_count = continuous_kib // bank_size_kib
        interleaved_count = interleaved_kib // bank_size_kib
        if continuous_count == 0:
            raise ComposerError("At least one continuous memory bank is required.")
        if interleaved_count != 0 and (
            interleaved_count & (interleaved_count - 1)
        ) != 0:
            raise ComposerError(
                "The number of interleaved memory banks must be a power of two."
            )
        if placement == "input_output_interleaved" and interleaved_count == 0:
            raise ComposerError(
                "memory_placement='input_output_interleaved' requires interleaved banks."
            )
        return {
            "RAM_BANKS": f"[{bank_size_kib}] * {continuous_count}",
            "INTERLEAVED_BANK_COUNT": str(interleaved_count),
            "INTERLEAVED_BANK_SIZE": str(bank_size_kib if interleaved_count else 0),
            "MEMORY_TOTAL_KIB": str(total_kib),
            "MEMORY_CONTINUOUS_KIB": str(continuous_kib),
            "MEMORY_INTERLEAVED_KIB": str(interleaved_kib),
            **cls.MEMORY_PLACEMENTS[placement],
        }

    @staticmethod
    def _fusesoc_file_entry(path: PurePosixPath) -> str:
        if path.suffix == ".sv":
            return f"      - {path.as_posix()}: {{file_type: systemVerilogSource}}"
        if path.suffix == ".vh":
            return f"      - {path.as_posix()}: {{is_include_file: true}}"
        return f"      - {path.as_posix()}"


__all__ = ["GRHeepConfigurationComposer", "GRHeepConfigurationPackage"]
