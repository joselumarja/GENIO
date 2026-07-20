from __future__ import annotations

import importlib.util
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from genio.artifacts import Artifact, ImageFunctionalMetricsArtifact
from genio.composer import Composer, PythonExecutionPackage
from genio.core.individual import Individual
from genio.evaluation.step import EvaluationStep
from genio.evaluation.task import EvaluationTask, ExecutionContext


@dataclass(frozen=True, slots=True)
class _ImageFunctionalSample:
    id: str
    image_path: Path
    reference_path: Path | None = None


@dataclass(frozen=True, slots=True)
class _ImageFunctionalExecution:
    sample: _ImageFunctionalSample
    output_path: Path | None
    elapsed_seconds: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class ImageFunctionalQualityError(RuntimeError):
    """Raised when a critical functional metric reports no useful quality."""


@dataclass(frozen=True, slots=True)
class PythonImageFunctionalTask(EvaluationTask):
    """Task that will execute a composed Python image-processing pipeline."""

    _SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
    _SUPPORTED_METRICS = frozenset(
        {
            "count_error",
            "instance_f1",
            "instance_precision",
            "instance_recall",
            "mask_accuracy",
            "mask_balanced_accuracy",
            "mask_f1",
            "mask_fnr",
            "mask_fpr",
            "mask_iou",
            "mask_precision",
            "mask_recall",
            "mask_specificity",
            "mean_box_iou",
        }
    )
    _CRITICAL_ZERO_METRICS = frozenset(
        {
            "instance_f1",
            "mask_accuracy",
            "mask_f1",
            "mask_iou",
        }
    )
    _BOX_IOU_THRESHOLD = 0.5

    composer: Composer | None = None
    images_path: Path | None = None
    references_path: Path | None = None
    metrics: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def cache_inputs(self) -> Mapping[str, Any]:
        """Cache functional results by the semantic image pipeline only."""

        return {"pipeline": self._pipeline_cache_inputs()}

    def run(self, context: ExecutionContext) -> list[Artifact]:
        """Execute the composed Python pipeline and return its functional metrics."""

        images_path = self._validate_configuration(context)
        package, package_dir = self._compose_and_materialize(context)
        samples = self._build_dataset(context, images_path)

        self._write_dataset_manifest(context, package_dir, samples)
        executions = self._execute_pipeline(context, package, package_dir, samples)

        # Persist every sample outcome before reporting aggregate failures.
        self._write_execution_manifest(context, executions)
        self._raise_for_execution_failures(executions)

        metrics_artifact = self._build_metrics_artifact(executions)
        self._raise_for_zero_critical_metrics(metrics_artifact.metrics())
        return [metrics_artifact]

    def _compose_and_materialize(
        self,
        context: ExecutionContext,
    ) -> tuple[PythonExecutionPackage, Path]:
        assert self.composer is not None
        package = self.composer.compose(self.individual)
        if not isinstance(package, PythonExecutionPackage):
            raise TypeError(
                "PythonImageFunctionalTask requires composer.compose() to return "
                "PythonExecutionPackage."
            )

        package_dir = context.materialize_package(self, package)
        context.write_json(
            package_dir / "package_metadata.json",
            {
                "entrypoint": package.entrypoint,
                "requirements": package.requirements,
                "metadata": dict(package.metadata),
            },
        )
        return package, package_dir

    @staticmethod
    def _write_dataset_manifest(
        context: ExecutionContext,
        package_dir: Path,
        samples: tuple[_ImageFunctionalSample, ...],
    ) -> None:
        context.write_json(
            package_dir / "dataset_manifest.json",
            [
                {
                    "id": sample.id,
                    "image_path": str(sample.image_path),
                    "reference_path": (
                        str(sample.reference_path)
                        if sample.reference_path is not None
                        else None
                    ),
                }
                for sample in samples
            ],
        )

    def _write_execution_manifest(
        self,
        context: ExecutionContext,
        executions: tuple[_ImageFunctionalExecution, ...],
    ) -> None:
        context.write_json(
            context.artifact_path(self, "execution_manifest.json"),
            [
                {
                    "id": execution.sample.id,
                    "image_path": str(execution.sample.image_path),
                    "reference_path": (
                        str(execution.sample.reference_path)
                        if execution.sample.reference_path is not None
                        else None
                    ),
                    "output_path": (
                        str(execution.output_path)
                        if execution.output_path is not None
                        else None
                    ),
                    "elapsed_seconds": execution.elapsed_seconds,
                    "error": execution.error,
                }
                for execution in executions
            ],
        )

    @staticmethod
    def _raise_for_execution_failures(
        executions: tuple[_ImageFunctionalExecution, ...],
    ) -> None:
        failures = [execution for execution in executions if execution.error is not None]
        if failures:
            raise RuntimeError(
                "Python image pipeline failed for samples: "
                f"{[execution.sample.id for execution in failures]!r}."
            )

    def _build_metrics_artifact(
        self,
        executions: tuple[_ImageFunctionalExecution, ...],
    ) -> ImageFunctionalMetricsArtifact:
        per_sample_metrics = self._compute_metrics(executions)
        values = self._aggregate_metrics(per_sample_metrics)
        return ImageFunctionalMetricsArtifact(
            name="image_functional_metrics",
            producer=self.step_id or "python_image_functional",
            individual_id=self.individual.id,
            values=values,
            per_sample_values=per_sample_metrics,
            metadata={
                "metrics": self.metrics,
                "box_iou_threshold": self._BOX_IOU_THRESHOLD,
            },
        )

    @classmethod
    def _raise_for_zero_critical_metrics(cls, metrics: Mapping[str, float]) -> None:
        zero_metrics = sorted(
            metric
            for metric in cls._CRITICAL_ZERO_METRICS
            if metrics.get(metric) == 0.0
        )
        if zero_metrics:
            raise ImageFunctionalQualityError(
                "Python image pipeline produced zero for critical functional metrics: "
                f"{zero_metrics!r}."
            )

    def _validate_configuration(self, context: ExecutionContext) -> Path:
        if self.composer is None:
            raise ValueError("PythonImageFunctionalTask requires a composer.")
        if self.images_path is None:
            raise ValueError("PythonImageFunctionalTask requires images_path.")

        images_path = context.resolve_path(self.images_path)
        if not images_path.is_dir():
            raise ValueError(f"images_path must be an existing directory: {images_path}.")
        if not self._contains_supported_images(images_path):
            raise ValueError(
                "images_path must contain at least one supported image file "
                f"({sorted(self._SUPPORTED_IMAGE_SUFFIXES)!r}): {images_path}."
            )

        if self.metrics and self.references_path is None:
            raise ValueError("references_path is required when metrics are requested.")
        if self.references_path is not None:
            references_path = context.resolve_path(self.references_path)
            if not references_path.is_dir():
                raise ValueError(
                    f"references_path must be an existing directory: {references_path}."
                )

        unknown_metrics = sorted(set(self.metrics) - self._SUPPORTED_METRICS)
        if unknown_metrics:
            raise ValueError(f"Unsupported image functional metrics: {unknown_metrics!r}.")

        return images_path

    def _build_dataset(
        self,
        context: ExecutionContext,
        images_path: Path,
    ) -> tuple[_ImageFunctionalSample, ...]:
        references_path = (
            context.resolve_path(self.references_path)
            if self.references_path is not None
            else None
        )
        samples = tuple(
            _ImageFunctionalSample(
                id=image_path.stem,
                image_path=image_path,
                reference_path=self._match_reference(image_path, references_path),
            )
            for image_path in self._discover_images(images_path)
        )

        if references_path is not None:
            missing_references = [
                sample.image_path.name
                for sample in samples
                if sample.reference_path is None
            ]
            if missing_references:
                raise ValueError(
                    "Missing reference images for input samples: "
                    f"{missing_references!r}."
                )

        return samples

    @classmethod
    def _discover_images(cls, path: Path) -> tuple[Path, ...]:
        return tuple(
            sorted(
                (
                    file
                    for file in path.iterdir()
                    if file.is_file()
                    and file.suffix.lower() in cls._SUPPORTED_IMAGE_SUFFIXES
                ),
                key=lambda file: file.name,
            )
        )

    @classmethod
    def _match_reference(cls, image_path: Path, references_path: Path | None) -> Path | None:
        if references_path is None:
            return None

        # Prefer the exact filename, then a deterministic same-stem alternative.
        exact_match = references_path / image_path.name
        if exact_match.is_file():
            return exact_match

        candidates = [
            candidate
            for candidate in references_path.iterdir()
            if candidate.is_file()
            and candidate.stem == image_path.stem
            and candidate.suffix.lower() in cls._SUPPORTED_IMAGE_SUFFIXES
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda file: file.name)[0]

    def _execute_pipeline(
        self,
        context: ExecutionContext,
        package: PythonExecutionPackage,
        package_dir: Path,
        samples: tuple[_ImageFunctionalSample, ...],
    ) -> tuple[_ImageFunctionalExecution, ...]:
        runner = self._load_entrypoint(package, package_dir)
        outputs_dir = context.ensure_dir(context.artifact_path(self, "outputs"))
        executions: list[_ImageFunctionalExecution] = []

        for sample in samples:
            start = time.perf_counter()
            try:
                image = self._read_image(sample.image_path)
                output = runner(image)
                output_path = outputs_dir / f"{sample.id}.png"
                self._write_image(output_path, output)
                executions.append(
                    _ImageFunctionalExecution(
                        sample=sample,
                        output_path=output_path,
                        elapsed_seconds=time.perf_counter() - start,
                    )
                )
            except Exception as exc:
                executions.append(
                    _ImageFunctionalExecution(
                        sample=sample,
                        output_path=None,
                        elapsed_seconds=time.perf_counter() - start,
                        error=str(exc),
                    )
                )

        return tuple(executions)

    @staticmethod
    def _load_entrypoint(
        package: PythonExecutionPackage,
        package_dir: Path,
    ) -> Callable[[Any], Any]:
        module_path_text, separator, function_name = package.entrypoint.partition(":")
        if not separator:
            raise ValueError(f"Invalid Python entrypoint: {package.entrypoint!r}.")

        module_path = package_dir / module_path_text
        spec = importlib.util.spec_from_file_location("genio_composed_pipeline", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load Python module from {module_path}.")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        entrypoint = getattr(module, function_name, None)
        if not callable(entrypoint):
            raise RuntimeError(f"Entrypoint {package.entrypoint!r} is not callable.")
        return entrypoint

    @staticmethod
    def _read_image(path: Path) -> Any:
        import cv2 as cv

        image = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read image: {path}.")
        return image

    @staticmethod
    def _write_image(path: Path, image: Any) -> None:
        import cv2 as cv

        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv.imwrite(str(path), image):
            raise ValueError(f"Could not write output image: {path}.")

    def _compute_metrics(
        self,
        executions: tuple[_ImageFunctionalExecution, ...],
    ) -> dict[str, dict[str, float]]:
        per_sample: dict[str, dict[str, float]] = {}
        for execution in executions:
            if execution.output_path is None or execution.sample.reference_path is None:
                continue

            prediction = self._binary_mask(self._read_image(execution.output_path))
            reference = self._binary_mask(self._read_image(execution.sample.reference_path))
            if prediction.shape != reference.shape:
                # Nearest-neighbor interpolation preserves discrete reference labels.
                reference = self._resize_mask(reference, prediction.shape)

            sample_metrics = self._mask_metrics(prediction, reference)
            sample_metrics.update(self._instance_metrics(prediction, reference))
            per_sample[execution.sample.id] = {
                metric: sample_metrics[metric]
                for metric in self.metrics
                if metric in sample_metrics
            }

        return per_sample

    @staticmethod
    def _aggregate_metrics(
        per_sample_metrics: Mapping[str, Mapping[str, float]],
    ) -> dict[str, float]:
        metric_names = sorted(
            {
                metric_name
                for sample_metrics in per_sample_metrics.values()
                for metric_name in sample_metrics
            }
        )
        return {
            metric_name: sum(
                sample_metrics[metric_name]
                for sample_metrics in per_sample_metrics.values()
                if metric_name in sample_metrics
            )
            / sum(
                1
                for sample_metrics in per_sample_metrics.values()
                if metric_name in sample_metrics
            )
            for metric_name in metric_names
        }

    @staticmethod
    def _binary_mask(image: Any) -> Any:
        import numpy as np

        array = np.asarray(image)
        if array.ndim == 3:
            array = array.any(axis=2)
        return array > 0

    @staticmethod
    def _resize_mask(mask: Any, shape: tuple[int, ...]) -> Any:
        import cv2 as cv
        import numpy as np

        resized = cv.resize(
            np.asarray(mask, dtype=np.uint8),
            (shape[1], shape[0]),
            interpolation=cv.INTER_NEAREST,
        )
        return resized > 0

    @staticmethod
    def _mask_metrics(prediction: Any, reference: Any) -> dict[str, float]:
        import numpy as np

        pred = np.asarray(prediction, dtype=bool)
        ref = np.asarray(reference, dtype=bool)
        tp = float(np.logical_and(pred, ref).sum())
        fp = float(np.logical_and(pred, np.logical_not(ref)).sum())
        fn = float(np.logical_and(np.logical_not(pred), ref).sum())
        tn = float(np.logical_and(np.logical_not(pred), np.logical_not(ref)).sum())

        precision = PythonImageFunctionalTask._safe_div(tp, tp + fp)
        recall = PythonImageFunctionalTask._safe_div(tp, tp + fn)
        specificity = PythonImageFunctionalTask._safe_div(tn, tn + fp)
        return {
            "mask_accuracy": PythonImageFunctionalTask._safe_div(tp + tn, tp + tn + fp + fn),
            "mask_balanced_accuracy": (recall + specificity) / 2.0,
            "mask_iou": PythonImageFunctionalTask._safe_div(tp, tp + fp + fn),
            "mask_f1": PythonImageFunctionalTask._safe_div(2.0 * precision * recall, precision + recall),
            "mask_fnr": PythonImageFunctionalTask._safe_div(fn, fn + tp),
            "mask_fpr": PythonImageFunctionalTask._safe_div(fp, fp + tn),
            "mask_precision": precision,
            "mask_recall": recall,
            "mask_specificity": specificity,
        }

    @classmethod
    def _instance_metrics(cls, prediction: Any, reference: Any) -> dict[str, float]:
        pred_boxes = cls._bounding_boxes(prediction)
        ref_boxes = cls._bounding_boxes(reference)
        matches = cls._match_boxes(pred_boxes, ref_boxes)

        tp = float(len(matches))
        fp = float(len(pred_boxes) - len(matches))
        fn = float(len(ref_boxes) - len(matches))
        precision = cls._safe_div(tp, tp + fp)
        recall = cls._safe_div(tp, tp + fn)

        return {
            "count_error": float(abs(len(pred_boxes) - len(ref_boxes))),
            "instance_f1": cls._safe_div(2.0 * precision * recall, precision + recall),
            "instance_precision": precision,
            "instance_recall": recall,
            "mean_box_iou": cls._safe_div(sum(match_iou for _, _, match_iou in matches), tp),
        }

    @staticmethod
    def _bounding_boxes(mask: Any) -> list[_BoundingBox]:
        import cv2 as cv
        import numpy as np

        binary = np.asarray(mask, dtype=np.uint8)
        num_labels, _, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
        boxes: list[_BoundingBox] = []
        for label in range(1, num_labels):
            x = int(stats[label, cv.CC_STAT_LEFT])
            y = int(stats[label, cv.CC_STAT_TOP])
            width = int(stats[label, cv.CC_STAT_WIDTH])
            height = int(stats[label, cv.CC_STAT_HEIGHT])
            boxes.append(_BoundingBox(x, y, x + width, y + height))
        return boxes

    @classmethod
    def _match_boxes(
        cls,
        pred_boxes: list[_BoundingBox],
        ref_boxes: list[_BoundingBox],
    ) -> list[tuple[int, int, float]]:
        # Greedily select highest-IoU one-to-one matches above the threshold.
        candidates = sorted(
            (
                (pred_index, ref_index, cls._box_iou(pred_box, ref_box))
                for pred_index, pred_box in enumerate(pred_boxes)
                for ref_index, ref_box in enumerate(ref_boxes)
            ),
            key=lambda item: item[2],
            reverse=True,
        )
        matched_predictions: set[int] = set()
        matched_references: set[int] = set()
        matches: list[tuple[int, int, float]] = []

        for pred_index, ref_index, iou in candidates:
            if iou < cls._BOX_IOU_THRESHOLD:
                break
            if pred_index in matched_predictions or ref_index in matched_references:
                continue
            matched_predictions.add(pred_index)
            matched_references.add(ref_index)
            matches.append((pred_index, ref_index, iou))

        return matches

    @staticmethod
    def _box_iou(left: _BoundingBox, right: _BoundingBox) -> float:
        x_min = max(left.x_min, right.x_min)
        y_min = max(left.y_min, right.y_min)
        x_max = min(left.x_max, right.x_max)
        y_max = min(left.y_max, right.y_max)
        intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
        left_area = max(0, left.x_max - left.x_min) * max(0, left.y_max - left.y_min)
        right_area = max(0, right.x_max - right.x_min) * max(0, right.y_max - right.y_min)
        return PythonImageFunctionalTask._safe_div(
            float(intersection),
            float(left_area + right_area - intersection),
        )

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        # Metric convention: every undefined 0/0 ratio evaluates to 1.0.
        if denominator == 0.0:
            return 1.0
        return numerator / denominator

    @classmethod
    def _contains_supported_images(cls, path: Path) -> bool:
        return any(
            file.is_file() and file.suffix.lower() in cls._SUPPORTED_IMAGE_SUFFIXES
            for file in path.iterdir()
        )


@dataclass(frozen=True, slots=True)
class PythonImageFunctionalEvaluationStep(EvaluationStep):
    """Creates tasks for functional evaluation of Python image pipelines."""

    id: str = "python_image_functional"
    depends_on: tuple[str, ...] = ()
    composer: Composer | None = None
    images_path: Path | None = None
    references_path: Path | None = None
    metrics: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    task_type: type[EvaluationTask] = PythonImageFunctionalTask

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return functional dataset, metric and composer configuration."""

        return {
            **EvaluationStep.checkpoint_signature(self),
            "composer": self.composer,
            "images_path": self.images_path,
            "references_path": self.references_path,
            "metrics": list(self.metrics),
            "metadata": dict(self.metadata),
        }

    def create_task(
        self,
        individual: Individual,
        artifacts: Mapping[str, Artifact],
    ) -> EvaluationTask:
        """Create a Python image functional evaluation task for an individual."""

        return PythonImageFunctionalTask(
            individual=individual,
            step_id=self.id,
            composer=self.composer,
            images_path=self.images_path,
            references_path=self.references_path,
            metrics=self.metrics,
            metadata={
                **dict(self.metadata),
                "input_artifacts": tuple(sorted(artifacts)),
            },
        )


__all__ = [
    "ImageFunctionalQualityError",
    "PythonImageFunctionalEvaluationStep",
    "PythonImageFunctionalTask",
]
