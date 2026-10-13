"""Hardware utilization objective for the insect HLS synthesis step."""

from __future__ import annotations

from math import fsum, isfinite
from numbers import Real

from genio.core.evaluation import Evaluation
from genio.objective import Objective, ObjectiveError, OptimizationDirection


RESOURCE_WEIGHTS = {resource: 0.25 for resource in ("ff", "lut", "bram", "dsp")}


class HardwareUtilizationObjective(Objective):
    """Minimize the mean used/available fraction across four FPGA resources."""

    def __init__(self, hls_step_id: str = "hls_image_pipeline_synthesis") -> None:
        if not isinstance(hls_step_id, str) or not hls_step_id.strip():
            raise ObjectiveError("HLS step id must be a non-empty string.")
        self.hls_step_id = hls_step_id

    @property
    def name(self) -> str:
        return "hardware"

    @property
    def direction(self) -> OptimizationDirection:
        return OptimizationDirection.MINIMIZE

    def value(self, evaluation: Evaluation) -> float:
        """Reject missing or invalid counts before calculating resource ratios."""

        metrics = evaluation.result.metrics
        ratios = []
        for resource, weight in RESOURCE_WEIGHTS.items():
            prefix = f"{self.hls_step_id}.hls_synthesis."
            used_key = f"{prefix}{resource}"
            available_key = f"{prefix}available_{resource}"
            values = []
            for key in (used_key, available_key):
                try:
                    raw = metrics[key]
                except KeyError as exc:
                    raise ObjectiveError(f"Missing hardware metric {key!r}.") from exc
                if isinstance(raw, bool) or not isinstance(raw, Real):
                    raise ObjectiveError(f"Hardware metric {key!r} must be finite numeric.")
                try:
                    numeric = float(raw)
                except (ValueError, OverflowError) as exc:
                    raise ObjectiveError(f"Hardware metric {key!r} must be finite.") from exc
                if not isfinite(numeric):
                    raise ObjectiveError(f"Hardware metric {key!r} must be finite.")
                values.append(numeric)
            used, available = values
            if used < 0 or available <= 0:
                raise ObjectiveError(
                    f"Hardware metrics {used_key!r} and {available_key!r} "
                    "require non-negative usage and positive capacity."
                )
            ratio = used / available
            if not isfinite(ratio):
                raise ObjectiveError(f"Hardware ratio for {resource!r} must be finite.")
            ratios.append(weight * ratio)
        total = fsum(ratios)
        if not isfinite(total):
            raise ObjectiveError("Hardware utilization must be finite.")
        return total

    def checkpoint_signature(self) -> dict[str, object]:
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "name": self.name,
            "direction": self.direction.value,
            "hls_step_id": self.hls_step_id,
            "resource_weights": dict(RESOURCE_WEIGHTS),
        }
