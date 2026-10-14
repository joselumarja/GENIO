"""Hardware utilization objective for the insect HLS synthesis step."""

from __future__ import annotations

from genio.core.evaluation import Evaluation
from genio.objective import Objective, OptimizationDirection


RESOURCE_WEIGHTS = {resource: 0.25 for resource in ("ff", "lut", "bram", "dsp")}


class HardwareUtilizationObjective(Objective):
    """Minimize the mean used/available fraction across four FPGA resources."""

    def __init__(self, hls_step_id: str = "hls_image_pipeline_synthesis") -> None:
        self.hls_step_id = hls_step_id

    @property
    def name(self) -> str:
        return "hardware"

    @property
    def direction(self) -> OptimizationDirection:
        return OptimizationDirection.MINIMIZE

    def value(self, evaluation: Evaluation) -> float:
        """Average used/available FF, LUT, BRAM and DSP fractions."""
        metrics = evaluation.result.metrics
        prefix = f"{self.hls_step_id}.hls_synthesis."
        return sum(
            weight * metrics[f"{prefix}{resource}"]
            / metrics[f"{prefix}available_{resource}"]
            for resource, weight in RESOURCE_WEIGHTS.items()
        )

    def checkpoint_signature(self) -> dict[str, object]:
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "name": self.name,
            "direction": self.direction.value,
            "hls_step_id": self.hls_step_id,
            "resource_weights": dict(RESOURCE_WEIGHTS),
        }
