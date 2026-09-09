from __future__ import annotations

import importlib.util
import time
from collections.abc import Iterable, Mapping
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
    """Describe one input image and its optional ground-truth image.

    Attributes:
        id: Stable sample identifier derived from the input filename stem.
        image_path: Path to the image passed to the composed pipeline.
        reference_path: Path to the matching reference image, when available.
    """

    id: str
    image_path: Path
    reference_path: Path | None = None


@dataclass(frozen=True, slots=True)
class _ImageFunctionalExecution:
    """Record the observable result of executing one image sample.

    Attributes:
        sample: Dataset sample that was executed.
        output_path: Persisted PNG output, or ``None`` when execution failed.
        elapsed_seconds: Wall-clock pipeline time measured for the sample.
        error: String representation of the caught exception, if any.
    """

    sample: _ImageFunctionalSample
    output_path: Path | None
    elapsed_seconds: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _BoundingBox:
    """Represent an axis-aligned component box with exclusive maxima.

    Attributes:
        x_min: Inclusive horizontal origin.
        y_min: Inclusive vertical origin.
        x_max: Exclusive horizontal limit.
        y_max: Exclusive vertical limit.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int


class ImageFunctionalQualityError(RuntimeError):
    """Indicate that at least one critical quality metric is exactly zero.

    This error distinguishes a successfully executed image pipeline with unusable
    functional quality from configuration, loading, and per-sample failures.
    """


@dataclass(frozen=True, slots=True)
class PythonImageFunctionalTask(EvaluationTask):
    """Execute and score a composed Python image-processing pipeline.

    The composer must produce a :class:`PythonExecutionPackage` whose entrypoint
    uses ``module_path:function_name`` syntax. The function receives an image as
    loaded by OpenCV and returns an OpenCV-compatible image, which is persisted
    as PNG. Reference images are matched first by exact filename and then by
    filename stem.

    Requested mask metrics operate on foreground pixels obtained by treating any
    nonzero channel as foreground. Instance metrics use 8-connected components
    and greedy one-to-one bounding-box matches at an IoU threshold of 0.5.

    Attributes:
        composer: Composer used to materialize the executable Python package.
        images_path: Directory containing the input image dataset.
        references_path: Optional directory containing ground-truth images.
        metrics: Ordered names of mask or instance metrics to calculate.
        metadata: Additional task metadata, including upstream artifact names.
    """

    _SUPPORTED_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})
    _MASK_METRICS = frozenset(
        {
            "mask_accuracy",
            "mask_balanced_accuracy",
            "mask_f1",
            "mask_fnr",
            "mask_fpr",
            "mask_iou",
            "mask_precision",
            "mask_recall",
            "mask_specificity",
        }
    )
    _INSTANCE_METRICS = frozenset(
        {
            "count_error",
            "instance_f1",
            "instance_precision",
            "instance_recall",
            "mean_box_iou",
        }
    )
    _MATCHED_INSTANCE_METRICS = _INSTANCE_METRICS - {"count_error"}
    _SUPPORTED_METRICS = _MASK_METRICS | _INSTANCE_METRICS
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
        """Execute the composed pipeline over the configured image dataset.

        The package directory receives package and dataset metadata. Per-sample
        PNG outputs are written below the task artifact directory, and an
        ``execution_manifest.json`` is persisted before aggregate failures are
        raised. Successful execution returns one functional-metrics artifact with
        aggregate and per-sample values.

        Args:
            context: Execution services used to resolve paths and write artifacts.

        Returns:
            A single :class:`ImageFunctionalMetricsArtifact` in a list.

        Raises:
            ValueError: If dataset paths or requested metrics are invalid.
            TypeError: If the composer returns a non-Python execution package.
            RuntimeError: If no entrypoint loader is available, the named object is
                not callable, or any sample fails.
            ImageFunctionalQualityError: If a critical requested metric is zero.
            Exception: Exceptions raised while importing the composed module
                propagate unchanged.
        """

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
        """Compose the candidate and persist its executable package metadata.

        Args:
            context: Execution context that owns the package directory.

        Returns:
            The typed Python package and its materialized directory.

        Raises:
            TypeError: If the configured composer returns another package type.
        """

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
        """Persist the ordered input-to-reference mapping beside the package."""

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
        """Persist output paths, timings, and errors for every attempted sample."""

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
        """Build the aggregate and per-sample functional metrics artifact."""

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
        """Reject exact-zero critical metrics while allowing absent metrics."""

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
        """Validate dataset and metric configuration and resolve the image path.

        References become mandatory when metrics are requested. A supplied
        reference directory may still be used with no metrics, while unsupported
        metric names are always rejected.
        """

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
        """Create a deterministic dataset and require every configured reference.

        Args:
            context: Execution context used to resolve the reference directory.
            images_path: Validated directory of supported input images.

        Returns:
            Samples ordered lexicographically by input filename.

        Raises:
            ValueError: If a reference directory is configured but any input has
                no exact-name or same-stem reference image.
        """

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
        """Select an exact-name reference or the first same-stem alternative."""

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
        """Run the package entrypoint once per sample and persist PNG outputs.

        Exceptions are captured per sample rather than raised immediately so the
        caller can persist a complete execution manifest before reporting the
        failed sample identifiers.
        """

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
        """Load a callable from the package's ``path:function`` entrypoint.

        Args:
            package: Composed package declaring the relative module entrypoint.
            package_dir: Directory containing the materialized module.

        Returns:
            A callable accepting an image object and returning an image object.

        Raises:
            ValueError: If the entrypoint omits the colon separator.
            RuntimeError: If no module loader is available or the named object is
                not callable.
            Exception: Exceptions raised while importing the composed module
                propagate unchanged.
        """

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
        """Compute requested metrics for successful samples with references.

        Reference masks are resized to prediction dimensions with nearest-neighbor
        interpolation. Failed samples and samples without references do not
        contribute entries to the returned per-sample mapping.
        """

        requested_metrics = frozenset(self.metrics)
        requested_mask_metrics = requested_metrics & self._MASK_METRICS
        requested_instance_metrics = requested_metrics & self._INSTANCE_METRICS
        per_sample: dict[str, dict[str, float]] = {}
        for execution in executions:
            if execution.output_path is None or execution.sample.reference_path is None:
                continue

            prediction = self._binary_mask(self._read_image(execution.output_path))
            reference = self._binary_mask(self._read_image(execution.sample.reference_path))
            if prediction.shape != reference.shape:
                # Nearest-neighbor interpolation preserves discrete reference labels.
                reference = self._resize_mask(reference, prediction.shape)

            sample_metrics: dict[str, float] = {}
            if requested_mask_metrics:
                sample_metrics.update(self._mask_metrics(prediction, reference))
            if requested_instance_metrics:
                sample_metrics.update(
                    self._instance_metrics(
                        prediction,
                        reference,
                        requested_instance_metrics,
                    )
                )
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
        """Average each metric over only the samples that provide that metric."""

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
        """Convert grayscale or multichannel data to a nonzero foreground mask."""

        import numpy as np

        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] > 0:
            binary = np.array(array[..., 0], dtype=bool, copy=True)
            for channel in range(1, array.shape[2]):
                np.logical_or(binary, array[..., channel], out=binary)
            return binary
        if array.ndim == 3:
            return array.any(axis=2)
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
        """Calculate pixel confusion-matrix metrics for two binary masks.

        Returns accuracy, balanced accuracy, IoU, F1, false-negative and
        false-positive rates, precision, recall, and specificity. Undefined
        zero-over-zero ratios follow :meth:`_safe_div` and evaluate to 1.0.
        """

        import numpy as np

        pred = np.asarray(prediction, dtype=bool)
        ref = np.asarray(reference, dtype=bool)
        pred, ref = np.broadcast_arrays(pred, ref)
        tp_count = int(np.count_nonzero(np.logical_and(pred, ref)))
        pred_count = int(np.count_nonzero(pred))
        ref_count = int(np.count_nonzero(ref))

        tp = float(tp_count)
        fp = float(pred_count - tp_count)
        fn = float(ref_count - tp_count)
        tn = float(pred.size - pred_count - ref_count + tp_count)

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
    def _instance_metrics(
        cls,
        prediction: Any,
        reference: Any,
        requested_metrics: frozenset[str] | None = None,
    ) -> dict[str, float]:
        """Calculate object-count and matched bounding-box metrics.

        Each 8-connected foreground component defines an instance. Precision,
        recall, F1, and mean box IoU use greedy one-to-one matches; when only
        ``count_error`` is requested, box matching is skipped.
        """

        pred_boxes = cls._bounding_boxes(prediction)
        ref_boxes = cls._bounding_boxes(reference)
        count_error = float(abs(len(pred_boxes) - len(ref_boxes)))
        if (
            requested_metrics is not None
            and requested_metrics.isdisjoint(cls._MATCHED_INSTANCE_METRICS)
        ):
            return {"count_error": count_error}

        matches = cls._match_boxes(pred_boxes, ref_boxes)

        tp = float(len(matches))
        fp = float(len(pred_boxes) - len(matches))
        fn = float(len(ref_boxes) - len(matches))
        precision = cls._safe_div(tp, tp + fp)
        recall = cls._safe_div(tp, tp + fn)

        return {
            "count_error": count_error,
            "instance_f1": cls._safe_div(2.0 * precision * recall, precision + recall),
            "instance_precision": precision,
            "instance_recall": recall,
            "mean_box_iou": cls._safe_div(sum(match_iou for _, _, match_iou in matches), tp),
        }

    @staticmethod
    def _bounding_boxes(mask: Any) -> list[_BoundingBox]:
        """Extract boxes for 8-connected foreground components, excluding background."""

        import cv2 as cv
        import numpy as np

        binary = np.asarray(mask)
        if binary.dtype == np.bool_:
            binary = binary.view(np.uint8)
        else:
            binary = np.asarray(binary, dtype=np.uint8)
        num_labels, _, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)
        return [
            _BoundingBox(x, y, x + width, y + height)
            for x, y, width, height in stats[1:num_labels, :4].tolist()
        ]

    @classmethod
    def _match_boxes(
        cls,
        pred_boxes: list[_BoundingBox],
        ref_boxes: list[_BoundingBox],
    ) -> list[tuple[int, int, float]]:
        """Greedily match highest-IoU prediction/reference box pairs.

        Returns:
            Tuples of prediction index, reference index, and IoU for unique pairs
            meeting :attr:`_BOX_IOU_THRESHOLD`, ordered by descending IoU.
        """

        import numpy as np

        # Greedily select highest-IoU one-to-one matches above the threshold.
        threshold = cls._BOX_IOU_THRESHOLD
        candidate_pairs: Iterable[tuple[Any, Any]]
        if threshold > 0.0 and pred_boxes and ref_boxes:
            pred = np.asarray(
                [(box.x_min, box.y_min, box.x_max, box.y_max) for box in pred_boxes]
            )
            ref = np.asarray(
                [(box.x_min, box.y_min, box.x_max, box.y_max) for box in ref_boxes]
            )
            candidate_mask = (
                (pred[:, None, 0] < ref[None, :, 2])
                & (ref[None, :, 0] < pred[:, None, 2])
                & (pred[:, None, 1] < ref[None, :, 3])
                & (ref[None, :, 1] < pred[:, None, 3])
            )

            # _safe_div defines the IoU of two zero-area boxes as 1.0.
            pred_zero_area = (pred[:, 2] <= pred[:, 0]) | (pred[:, 3] <= pred[:, 1])
            ref_zero_area = (ref[:, 2] <= ref[:, 0]) | (ref[:, 3] <= ref[:, 1])
            candidate_mask |= pred_zero_area[:, None] & ref_zero_area[None, :]
            candidate_pairs = zip(*np.nonzero(candidate_mask))
        else:
            candidate_pairs = (
                (pred_index, ref_index)
                for pred_index in range(len(pred_boxes))
                for ref_index in range(len(ref_boxes))
            )

        candidates: list[tuple[int, int, float]] = []
        for pred_index, ref_index in candidate_pairs:
            pred_index = int(pred_index)
            ref_index = int(ref_index)
            iou = cls._box_iou(pred_boxes[pred_index], ref_boxes[ref_index])
            if iou < threshold:
                continue
            candidates.append((pred_index, ref_index, iou))
        candidates.sort(key=lambda item: item[2], reverse=True)
        matched_predictions: set[int] = set()
        matched_references: set[int] = set()
        matches: list[tuple[int, int, float]] = []

        for pred_index, ref_index, iou in candidates:
            if pred_index in matched_predictions or ref_index in matched_references:
                continue
            matched_predictions.add(pred_index)
            matched_references.add(ref_index)
            matches.append((pred_index, ref_index, iou))

        return matches

    @staticmethod
    def _box_iou(left: _BoundingBox, right: _BoundingBox) -> float:
        """Return box intersection over union using exclusive maximum coordinates."""

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
        """Divide metric terms, defining every zero-denominator ratio as 1.0."""

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
    """Configure task creation for functional Python image evaluation.

    Attributes:
        id: Evaluation graph identifier and default artifact producer.
        depends_on: Step identifiers that must complete before this step.
        composer: Composer propagated to each functional task.
        images_path: Input image dataset directory.
        references_path: Optional ground-truth image directory.
        metrics: Ordered functional metrics requested from each task.
        metadata: Additional metadata propagated to each task.
        task_type: Concrete task class created by the step.
    """

    id: str = "python_image_functional"
    produced_artifacts = {
        "image_functional_metrics": ImageFunctionalMetricsArtifact,
    }
    depends_on: tuple[str, ...] = ()
    composer: Composer | None = None
    images_path: Path | None = None
    references_path: Path | None = None
    metrics: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    task_type: type[EvaluationTask] = PythonImageFunctionalTask

    def checkpoint_signature(self) -> Mapping[str, Any]:
        """Return functional dataset, metric, and composer checkpoint inputs."""

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
        """Create a functional task and record available upstream artifacts.

        Args:
            individual: Candidate image pipeline to evaluate.
            artifacts: Artifacts available from dependency steps.

        Returns:
            A configured :class:`PythonImageFunctionalTask`.
        """

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
