from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genio.composer.base import Composer, ComposerError, ExecutionPackage
from genio.core import Individual, StageChoice


@dataclass(frozen=True, slots=True)
class PythonExecutionPackage(ExecutionPackage):
    """Portable Python execution package suitable for local or remote backends."""

    entrypoint: str
    files: Mapping[str, str]
    requirements: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def materialize(self, target_dir: str | Path) -> Path:
        package_dir = Path(target_dir)
        package_dir.mkdir(parents=True, exist_ok=True)

        for relative_path, content in self.files.items():
            destination = package_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")

        if self.requirements:
            requirements = "\n".join(self.requirements) + "\n"
            (package_dir / "requirements.txt").write_text(requirements, encoding="utf-8")

        return package_dir


class PythonImagePipelineComposer(Composer):
    """Compose image-processing Individuals into portable Python packages."""

    _OPENCV_ENUMS = {
        "area": "cv.INTER_AREA",
        "binary": "cv.THRESH_BINARY",
        "binary_inv": "cv.THRESH_BINARY_INV",
        "cubic": "cv.INTER_CUBIC",
        "linear": "cv.INTER_LINEAR",
        "cross": "cv.MORPH_CROSS",
        "ellipse": "cv.MORPH_ELLIPSE",
        "nearest": "cv.INTER_NEAREST",
        "otsu": "cv.THRESH_OTSU",
        "rect": "cv.MORPH_RECT",
        "tozero": "cv.THRESH_TOZERO",
        "tozero_inv": "cv.THRESH_TOZERO_INV",
        "trunc": "cv.THRESH_TRUNC",
    }

    def __init__(
        self,
        stages_definitions_path: str | Path,
        *,
        implementation_source: str = "opencv",
    ) -> None:
        super().__init__(stages_definitions_path)
        self.implementation_source = implementation_source

    def compose(self, individual: Individual) -> PythonExecutionPackage:
        imports: list[str] = []
        body: list[str] = ["current = image"]
        metadata = self.artifact_metadata(individual)

        for stage_index, (choice, definition) in enumerate(
            self.active_stage_definitions(individual)
        ):
            implementation = self._stage_implementation(definition)
            for import_line in implementation.get("imports", []):
                if import_line not in imports:
                    imports.append(import_line)

            output_name = f"stage_{stage_index}_output"
            token_values = self._token_values(choice, definition, output_name)
            function_lines = [
                self._replace_tokens(line, token_values)
                for line in implementation.get("function", [])
            ]
            body.extend(function_lines)
            body.append(f"current = {output_name}")

        source = self._render_pipeline(imports, body)
        return PythonExecutionPackage(
            entrypoint="pipeline.py:run",
            files={"pipeline.py": source},
            requirements=("opencv-python", "numpy"),
            metadata={
                **metadata,
                "implementation_source": self.implementation_source,
                "package_type": "python_image_pipeline",
            },
        )

    def _stage_implementation(self, definition: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            relative_path = definition["implementations"]["functional"][
                self.implementation_source
            ]
        except KeyError as exc:
            stage = definition.get("id", "<unknown>")
            raise ComposerError(
                f"Stage {stage!r} does not define a functional "
                f"{self.implementation_source!r} implementation."
            ) from exc

        implementation_path = self.stages_definitions_path.parent / relative_path
        with implementation_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _token_values(
        self,
        choice: StageChoice,
        definition: Mapping[str, Any],
        output_name: str,
    ) -> dict[str, str]:
        values = {
            "@input_0": "current",
            "@output_0": output_name,
        }

        for parameter in definition.get("parameters", []):
            name = parameter["name"]
            token = parameter["token"]
            if name not in choice.parameters:
                continue
            values[token] = self._python_literal(choice.parameters[name])

        return values

    def _python_literal(self, value: Any) -> str:
        if isinstance(value, str):
            return self._OPENCV_ENUMS.get(value, repr(value))
        return repr(value)

    @staticmethod
    def _replace_tokens(line: str, token_values: Mapping[str, str]) -> str:
        rendered = line
        for token, value in sorted(token_values.items(), key=lambda item: len(item[0]), reverse=True):
            rendered = rendered.replace(token, value)
        return rendered

    @staticmethod
    def _render_pipeline(imports: list[str], body: list[str]) -> str:
        lines = [
            "from __future__ import annotations",
            "",
            "import numpy as np",
        ]
        lines.extend(import_line for import_line in imports if import_line != "import numpy as np")
        lines.extend([
            "",
            "",
            "def run(image):",
        ])
        if body:
            lines.extend(f"    {line}" for line in body)
        else:
            lines.append("    current = image")
        lines.append("    return current")
        lines.append("")
        return "\n".join(lines)


__all__ = ["PythonExecutionPackage", "PythonImagePipelineComposer"]
