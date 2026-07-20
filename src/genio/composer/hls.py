from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from genio.composer.base import Composer, ComposerError, ExecutionPackage
from genio.core import Individual, StageChoice


@dataclass(frozen=True, slots=True)
class HLSExecutionPackage(ExecutionPackage):
    """Portable HLS execution package suitable for Vitis HLS synthesis."""

    entrypoint: str = "hls_config.cfg"
    files: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def materialize(self, target_dir: str | Path) -> Path:
        """Write the HLS package to a target directory."""

        package_dir = Path(target_dir)
        package_dir.mkdir(parents=True, exist_ok=True)

        for relative_path, content in self.files.items():
            destination = package_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")

        if self.metadata:
            metadata_path = package_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(dict(self.metadata), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        return package_dir


class HLSImagePipelineComposer(Composer):
    """Compose image-processing Individuals into Vitis Vision HLS packages."""

    _HLS_ENUMS = {
        "binary": "XF_THRESHOLD_TYPE_BINARY",
        "binary_inv": "XF_THRESHOLD_TYPE_BINARY_INV",
        "cross": "XF_SHAPE_CROSS",
        "ellipse": "XF_SHAPE_ELLIPSE",
        "area": "XF_INTERPOLATION_AREA",
        "linear": "XF_INTERPOLATION_BILINEAR",
        "nearest": "XF_INTERPOLATION_NN",
        "rect": "XF_SHAPE_RECT",
        "tozero": "XF_THRESHOLD_TYPE_TOZERO",
        "tozero_inv": "XF_THRESHOLD_TYPE_TOZERO_INV",
        "trunc": "XF_THRESHOLD_TYPE_TRUNC",
    }

    _GRAYSCALE_STAGES = frozenset({"bgr_to_gray", "rgb_to_gray", "threshold", "in_range"})
    _MORPHOLOGY_STAGES = frozenset({"dilate", "erode", "morph_close", "morph_open"})
    _SINGLE_CHANNEL_TYPES = {
        "XF_8UC3": "XF_8UC1",
        "XF_8UC4": "XF_8UC1",
        "XF_14UC3": "XF_14UC1",
        "XF_16UC3": "XF_16UC1",
        "XF_16UC4": "XF_16UC1",
    }
    _GRADIENT_OUTPUT_TYPES = {
        "XF_8UC1": "XF_16SC1",
        "XF_8UC3": "XF_16SC3",
    }
    _UNRESOLVED_TOKEN_PATTERN = re.compile(r"@[A-Za-z0-9_]+")

    def __init__(
        self,
        stages_definitions_path: str | Path,
        *,
        templates_path: str | Path = "hls_templates/vitis_vision_image_pipeline",
        vitis_version: str = "2025.2",
        interface: str = "fifo",
        top_function: str = "top",
        rows: int = 2160,
        cols: int = 3840,
        image_type: str = "XF_8UC3",
        npc: str = "XF_NPPC1",
        axi_width: int = 32,
        axi_user_width: int = 0,
        axi_id_width: int = 0,
        axi_dest_width: int = 0,
    ) -> None:
        super().__init__(stages_definitions_path)
        self.templates_path = Path(templates_path)
        if not vitis_version.strip():
            raise ValueError("vitis_version cannot be empty.")
        self.vitis_version = vitis_version
        self.interface = interface
        self.top_function = top_function
        self.rows = rows
        self.cols = cols
        self.image_type = image_type
        self.npc = npc
        self.axi_width = axi_width
        self.axi_user_width = axi_user_width
        self.axi_id_width = axi_id_width
        self.axi_dest_width = axi_dest_width

    def compose(self, individual: Individual) -> HLSExecutionPackage:
        """Compose an individual into a Vitis Vision HLS package."""

        hls_design = individual.design.get("hls", {})
        if not isinstance(hls_design, dict):
            raise ComposerError("Individual design domain 'hls' must be a mapping.")

        includes: list[str] = []
        declarations: list[str] = []
        body: list[str] = []
        current_mat = "input_mat"
        current_type = self.image_type
        current_rows = str(self.rows)
        current_cols = str(self.cols)
        current_npc = str(hls_design.get("npc", self.npc))
        notes: list[str] = []

        active_stages = list(self.active_stage_definitions(individual))
        for stage_index, (choice, definition) in enumerate(active_stages):
            implementation = self._stage_implementation(definition)
            for include in implementation.get("include", []):
                include_line = f'#include "{include}"'
                if include_line not in includes:
                    includes.append(include_line)

            input_type = current_type
            input_rows = current_rows
            input_cols = current_cols
            input_npc = current_npc
            output_type, output_rows, output_cols, output_npc = self._stage_output_state(
                choice,
                input_type=input_type,
                input_rows=input_rows,
                input_cols=input_cols,
                input_npc=input_npc,
                notes=notes,
            )
            output_mat = f"stage_{stage_index}_output"
            declarations.append(
                f"    xf::cv::Mat<{output_type}, {output_rows}, {output_cols}, {output_npc}> "
                f"{output_mat}({output_rows}, {output_cols});"
            )

            pre_lines = self._stage_pre_lines(choice, stage_index, input_type, input_rows, input_cols, input_npc)
            body.extend(f"    {line}" for line in pre_lines)

            token_values = self._token_values(
                choice,
                definition,
                input_name=current_mat,
                output_name=output_mat,
                rows=input_rows,
                cols=input_cols,
                image_type=input_type,
                npc=input_npc,
                output_type=output_type,
                output_rows=output_rows,
                output_cols=output_cols,
                output_npc=output_npc,
                stage_index=stage_index,
                hls_design=hls_design,
            )
            for line in implementation.get("function", []):
                rendered_line = self._replace_tokens(line, token_values)
                if choice.stage in self._MORPHOLOGY_STAGES:
                    rendered_line = self._rewrite_morphology_line(rendered_line, stage_index)
                body.append(f"    {self._comment_unresolved_tokens(rendered_line)}")

            current_mat = output_mat
            current_type = output_type
            current_rows = output_rows
            current_cols = output_cols
            current_npc = output_npc

        source = self._render_template(
            includes=includes,
            declarations=declarations,
            body=body,
            output_name=current_mat,
            output_type=current_type,
            output_rows=current_rows,
            output_cols=current_cols,
            output_npc=current_npc,
            notes=notes,
            hls_design=hls_design,
        )

        files = {
            "src/pipeline.cpp": source,
            "include/xf_fifo_utils.hpp": self._read_template("include/xf_fifo_utils.hpp"),
            "hls_config.cfg": self._render_hls_config(hls_design),
        }
        if self.interface == "axi_stream":
            files["include/xf_axi_stream_utils.hpp"] = self._read_template(
                "include/xf_axi_stream_utils.hpp"
            )

        return HLSExecutionPackage(
            files=files,
            metadata={
                **self.artifact_metadata(individual),
                "backend": "vitis_vision",
                "vitis_version": self.vitis_version,
                "include_dirs": ("include",),
                "interface": self.interface,
                "package_type": "hls_image_pipeline",
                "required_backend_resources": ("vitis_libraries_path",),
                "source_files": ["src/pipeline.cpp"],
                "top_function": self.top_function,
                "input_type": self.image_type,
                "output_type": current_type,
                "rows": self.rows,
                "cols": self.cols,
                "npc": current_npc,
                "hls_design": dict(hls_design),
                "notes": tuple(notes),
            },
        )

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return HLS generation settings, definitions and template fingerprints."""

        return {
            **Composer.checkpoint_signature(self),
            "templates_path": str(self.templates_path.expanduser().resolve()),
            "template_tree": self._directory_fingerprint(self.templates_path),
            "vitis_version": self.vitis_version,
            "interface": self.interface,
            "top_function": self.top_function,
            "rows": self.rows,
            "cols": self.cols,
            "image_type": self.image_type,
            "npc": self.npc,
            "axi_width": self.axi_width,
            "axi_user_width": self.axi_user_width,
            "axi_id_width": self.axi_id_width,
            "axi_dest_width": self.axi_dest_width,
        }

    def _stage_implementation(self, definition: Mapping[str, Any]) -> Mapping[str, Any]:
        stage = str(definition.get("id", "<unknown>"))
        try:
            relative_path = definition["implementations"]["hls"]["vitis_vision"]
        except (KeyError, TypeError) as exc:
            raise ComposerError(
                f"Stage {stage!r} does not define an HLS 'vitis_vision' implementation."
            ) from exc

        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ComposerError(
                f"Stage {stage!r} HLS implementation path must be a non-empty string."
            )
        implementation_root = self.stages_definitions_path.parent.resolve()
        implementation_path = (implementation_root / relative_path).resolve()
        try:
            implementation_path.relative_to(implementation_root)
        except ValueError as exc:
            raise ComposerError(
                f"Stage {stage!r} HLS implementation escapes {implementation_root}."
            ) from exc
        if not implementation_path.is_file():
            raise ComposerError(
                f"Stage {stage!r} HLS implementation file does not exist: "
                f"{implementation_path}."
            )
        try:
            with implementation_path.open("r", encoding="utf-8") as file:
                implementation = json.load(file)
        except json.JSONDecodeError as exc:
            raise ComposerError(
                f"Stage {stage!r} HLS implementation is not valid JSON: "
                f"{implementation_path}."
            ) from exc
        if not isinstance(implementation, Mapping):
            raise ComposerError(
                f"Stage {stage!r} HLS implementation root must be an object."
            )
        if implementation.get("stage") != stage:
            raise ComposerError(
                f"HLS implementation {implementation_path} declares stage "
                f"{implementation.get('stage')!r}, expected {stage!r}."
            )

        implementation = self._versioned_implementation(
            implementation,
            stage=stage,
            implementation_path=implementation_path,
        )

        includes = self._implementation_includes(
            implementation.get("include", ()),
            stage=stage,
        )
        function = implementation.get("function")
        if (
            not isinstance(function, Sequence)
            or isinstance(function, (str, bytes, bytearray))
            or any(not isinstance(line, str) for line in function)
        ):
            raise ComposerError(
                f"Stage {stage!r} HLS implementation 'function' must be a list "
                "of strings."
            )
        return {
            **implementation,
            "include": includes,
            "function": tuple(function),
        }

    def _versioned_implementation(
        self,
        implementation: Mapping[str, Any],
        *,
        stage: str,
        implementation_path: Path,
    ) -> dict[str, Any]:
        versions = implementation.get("versions")
        if versions is None:
            return dict(implementation)
        if not isinstance(versions, Mapping):
            raise ComposerError(
                f"Stage {stage!r} HLS implementation 'versions' must be an object."
            )
        variant = versions.get(self.vitis_version, versions.get("default"))
        if variant is None:
            raise ComposerError(
                f"HLS implementation {implementation_path} does not support Vitis "
                f"{self.vitis_version!r}."
            )
        if not isinstance(variant, Mapping):
            raise ComposerError(
                f"Stage {stage!r} HLS variant {self.vitis_version!r} must be an object."
            )
        return {
            **{key: value for key, value in implementation.items() if key != "versions"},
            **variant,
        }

    @staticmethod
    def _implementation_includes(values: Any, *, stage: str) -> tuple[str, ...]:
        if isinstance(values, str):
            values = (values,)
        elif (
            not isinstance(values, Sequence)
            or isinstance(values, (bytes, bytearray))
        ):
            raise ComposerError(
                f"Stage {stage!r} HLS implementation 'include' must be a string "
                "or a list of strings."
            )

        includes: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ComposerError(
                    f"Stage {stage!r} HLS includes must be non-empty strings."
                )
            include = value.strip()
            include_path = PurePosixPath(include)
            if (
                include_path.is_absolute()
                or ".." in include_path.parts
                or "\\" in include
                or '"' in include
                or "\n" in include
                or "\r" in include
            ):
                raise ComposerError(
                    f"Stage {stage!r} has an invalid portable include: {include!r}."
                )
            if include not in includes:
                includes.append(include)
        return tuple(includes)

    def _token_values(
        self,
        choice: StageChoice,
        definition: Mapping[str, Any],
        *,
        input_name: str,
        output_name: str,
        rows: str,
        cols: str,
        image_type: str,
        npc: str,
        output_type: str,
        output_rows: str,
        output_cols: str,
        output_npc: str,
        stage_index: int,
        hls_design: Mapping[str, Any],
    ) -> dict[str, str]:
        values = {
            "@BORDER_TYPE": "XF_BORDER_CONSTANT",
            "@COLS": cols,
            "@HLS_STREAM_DEPTH": "2",
            "@MAXDOWNSCALE": "2",
            "@NPC": npc,
            "@OUT_COLS": output_cols,
            "@OUT_NPC": output_npc,
            "@OUT_ROWS": output_rows,
            "@OUT_TYPE": output_type,
            "@ROWS": rows,
            "@TYPE": image_type,
            "@USE_URAM": self._hls_literal(hls_design.get("use_uram", False)),
            "@imgAux": f"img_aux_{stage_index}",
            "@kernel": f"kernel_{stage_index}",
            "@input_0": input_name,
            "@output_0": output_name,
            "K_COLS": self._hls_literal(choice.parameters.get("kernel_cols", "K_COLS")),
            "K_ROWS": self._hls_literal(choice.parameters.get("kernel_rows", "K_ROWS")),
        }

        for parameter in definition.get("parameters", []):
            name = parameter["name"]
            token = parameter["token"]
            if name not in choice.parameters:
                continue
            values[token] = self._hls_literal(choice.parameters[name])

        return values

    def _render_template(
        self,
        *,
        includes: list[str],
        declarations: list[str],
        body: list[str],
        output_name: str,
        output_type: str,
        output_rows: str,
        output_cols: str,
        output_npc: str,
        notes: list[str],
        hls_design: Mapping[str, Any],
    ) -> str:
        template = self._read_template(self._template_name())
        replacements = {
            "@AXI_DEST_WIDTH@": str(self.axi_dest_width),
            "@AXI_ID_WIDTH@": str(self.axi_id_width),
            "@AXI_USER_WIDTH@": str(self.axi_user_width),
            "@AXI_WIDTH@": str(self.axi_width),
            "@COLS@": str(self.cols),
            "@INCLUDES@": "\n".join(includes),
            "@INTERMEDIATE_DECLARATIONS@": "\n".join(declarations),
            "@NPC@": str(hls_design.get("npc", self.npc)),
            "@OUTPUT_COLS@": output_cols,
            "@OUTPUT_NPC@": output_npc,
            "@OUTPUT_ROWS@": output_rows,
            "@OUTPUT_TYPE@": output_type,
            "@PIPELINE_BODY@": "\n".join(self._note_lines(notes) + body),
            "@ROWS@": str(self.rows),
            "@TOP_FUNCTION@": self.top_function,
            "@TOP_PRAGMAS@": self._top_pragmas(hls_design),
            "@TYPE@": self.image_type,
            "@output_0": output_name,
        }
        return self._replace_tokens(template, replacements)

    def _top_pragmas(self, hls_design: Mapping[str, Any]) -> str:
        pragmas: list[str] = []
        pipeline_ii = hls_design.get("pipeline_ii")
        if pipeline_ii is not None:
            pragmas.append(f"#pragma HLS PIPELINE II={pipeline_ii}")
        return "\n".join(pragmas)

    def _render_hls_config(self, hls_design: Mapping[str, Any]) -> str:
        config = self._read_template("hls_config.cfg")
        replacements = {
            "clock": hls_design.get("clock"),
            "flow_target": hls_design.get("flow_target"),
            "syn.file": "src/pipeline.cpp",
            "syn.top": self.top_function,
        }
        return self._replace_hls_config_values(config, replacements)

    @classmethod
    def _replace_hls_config_values(
        cls,
        config: str,
        replacements: Mapping[str, Any],
    ) -> str:
        lines = config.splitlines()
        for key, value in replacements.items():
            if value is None:
                continue
            lines = cls._replace_hls_config_value(lines, key, str(value))
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _replace_hls_config_value(lines: list[str], key: str, value: str) -> list[str]:
        prefix = f"{key}="
        for index, line in enumerate(lines):
            if line.strip().startswith(prefix):
                lines[index] = f"{key}={value}"
                return lines
        lines.append(f"{key}={value}")
        return lines

    def _template_name(self) -> str:
        if self.interface == "fifo":
            return "fifo_top.cpp.tpl"
        if self.interface == "axi_stream":
            return "axi_stream_top.cpp.tpl"
        raise ComposerError(f"Unsupported HLS image interface: {self.interface!r}.")

    def _read_template(self, relative_path: str) -> str:
        return (self.templates_path / relative_path).read_text(encoding="utf-8")

    def _output_type(self, stage: str, current_type: str) -> str:
        if stage in self._GRAYSCALE_STAGES:
            return "XF_8UC1"
        return current_type

    def _stage_output_state(
        self,
        choice: StageChoice,
        *,
        input_type: str,
        input_rows: str,
        input_cols: str,
        input_npc: str,
        notes: list[str],
    ) -> tuple[str, str, str, str]:
        output_type = self._output_type(choice.stage, input_type)
        output_rows = input_rows
        output_cols = input_cols
        output_npc = input_npc

        if choice.stage == "canny":
            self._require_input_type(choice.stage, input_type, ("XF_8UC1",))
            self._require_input_npc(choice.stage, input_npc, ("XF_NPPC1", "XF_NPPC8"))
            self._require_divisible_cols(choice.stage, input_cols, 32)
            output_type = "XF_2UC1"
            output_npc = "XF_NPPC32"
        elif choice.stage == "channel_extract":
            try:
                output_type = self._SINGLE_CHANNEL_TYPES[input_type]
            except KeyError as exc:
                supported = ", ".join(self._SINGLE_CHANNEL_TYPES)
                raise ComposerError(
                    f"Stage 'channel_extract' does not support input type {input_type!r}; "
                    f"expected one of: {supported}."
                ) from exc
        elif choice.stage == "convert_scale_abs":
            self._require_input_type(choice.stage, input_type, ("XF_8UC1",))
            output_type = "XF_8UC1"
        elif choice.stage == "remap":
            self._require_input_type(choice.stage, input_type, ("XF_8UC1", "XF_8UC3"))
            self._require_input_npc(
                choice.stage,
                input_npc,
                ("XF_NPPC1", "XF_NPPC2", "XF_NPPC4", "XF_NPPC8"),
            )
            output_type = input_type
        elif choice.stage in {"scharr", "sobel"}:
            try:
                output_type = self._GRADIENT_OUTPUT_TYPES[input_type]
            except KeyError as exc:
                supported = ", ".join(self._GRADIENT_OUTPUT_TYPES)
                raise ComposerError(
                    f"Stage {choice.stage!r} does not support input type {input_type!r}; "
                    f"expected one of: {supported}."
                ) from exc
            self._require_input_npc(
                choice.stage,
                input_npc,
                ("XF_NPPC1", "XF_NPPC8"),
            )
        elif choice.stage == "resize":
            output_rows = str(choice.parameters.get("out_rows", input_rows))
            output_cols = str(choice.parameters.get("out_cols", input_cols))

        if choice.stage == "hist_equalize" and input_type != "XF_8UC1":
            notes.append(
                "hist_equalize is generated for the current type, but Vitis Vision "
                "equalizeHist is typically used with XF_8UC1 inputs."
            )

        return output_type, output_rows, output_cols, output_npc

    @staticmethod
    def _require_input_type(
        stage: str,
        input_type: str,
        supported: tuple[str, ...],
    ) -> None:
        if input_type not in supported:
            raise ComposerError(
                f"Stage {stage!r} does not support input type {input_type!r}; "
                f"expected one of: {', '.join(supported)}."
            )

    @staticmethod
    def _require_input_npc(
        stage: str,
        input_npc: str,
        supported: tuple[str, ...],
    ) -> None:
        if input_npc not in supported:
            raise ComposerError(
                f"Stage {stage!r} does not support NPC {input_npc!r}; "
                f"expected one of: {', '.join(supported)}."
            )

    @staticmethod
    def _require_divisible_cols(stage: str, cols: str, divisor: int) -> None:
        try:
            numeric_cols = int(cols)
        except ValueError:
            return
        if numeric_cols % divisor != 0:
            raise ComposerError(
                f"Stage {stage!r} requires the input columns to be divisible by "
                f"{divisor}; got {numeric_cols}."
            )

    def _stage_pre_lines(
        self,
        choice: StageChoice,
        stage_index: int,
        image_type: str,
        rows: str,
        cols: str,
        npc: str,
    ) -> list[str]:
        if choice.stage not in self._MORPHOLOGY_STAGES:
            return []

        kernel_rows = int(choice.parameters.get("kernel_rows", 3))
        kernel_cols = int(choice.parameters.get("kernel_cols", 3))
        values = ", ".join("1" for _ in range(kernel_rows * kernel_cols))
        return [
            f"unsigned char kernel_{stage_index}[{kernel_rows * kernel_cols}] = {{{values}}};",
        ]

    def _rewrite_morphology_line(self, line: str, stage_index: int) -> str:
        return (
            line.replace("imgAux", f"img_aux_{stage_index}")
            .replace("unsigned char kernel_", "// Local kernel declaration generated by composer: kernel_")
            .replace("unsigned char kernel[", "// Local kernel declaration generated by composer: kernel[")
            .replace("kernel);", f"kernel_{stage_index});")
        )

    def _comment_unresolved_tokens(self, line: str) -> str:
        unresolved = sorted(set(self._UNRESOLVED_TOKEN_PATTERN.findall(line)))
        if not unresolved:
            return line
        return f"// TODO unresolved HLS tokens {', '.join(unresolved)}: {line}"

    @staticmethod
    def _note_lines(notes: list[str]) -> list[str]:
        return [f"    // TODO {note}" for note in notes]

    def _hls_literal(self, value: Any) -> str:
        if isinstance(value, str):
            return self._HLS_ENUMS.get(value, value)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _replace_tokens(source: str, token_values: Mapping[str, str]) -> str:
        if not token_values:
            return source

        patterns: list[str] = []
        for token in sorted(token_values, key=len, reverse=True):
            escaped = re.escape(token)
            if token.startswith("@") and not token.endswith("@"):
                escaped += r"(?![A-Za-z0-9_])"
            elif token[0].isalnum() or token[0] == "_":
                escaped = r"(?<![A-Za-z0-9_])" + escaped
                if token[-1].isalnum() or token[-1] == "_":
                    escaped += r"(?![A-Za-z0-9_])"
            patterns.append(escaped)
        token_pattern = re.compile("|".join(patterns))
        return token_pattern.sub(
            lambda match: str(token_values[match.group(0)]),
            source,
        )


__all__ = ["HLSExecutionPackage", "HLSImagePipelineComposer"]
