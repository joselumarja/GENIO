from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genio.artifacts import Artifact
from genio.composer import Composer
from genio.core.individual import Individual
from genio.evaluation.step import EvaluationStep
from genio.evaluation.task import EvaluationTask, ExecutionContext


@dataclass(frozen=True, slots=True)
class HLSSynthesisTask(EvaluationTask):
    """Task that will synthesize an HLS implementation for one Individual."""

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

    def run(self, context: ExecutionContext) -> list[Artifact]:
        # Validate synthesis configuration before invoking any external tool.
        # - Ensure a composer has been provided.
        # - Ensure hls_config points to a Vitis HLS .cfg file, or that enough
        #   overrides are available to generate one.
        # - Ensure mandatory synthesis inputs are present either in hls_config
        #   or in config_overrides: part, hls.clock, hls.syn.file, and
        #   hls.syn.top.
        # - Resolve all paths through ExecutionContext so local and remote
        #   backends control the actual workspace location.

        # Compose the Individual into HLS source assets.
        # - Call self.composer.compose(self.individual).
        # - Materialize generated C/C++ sources, headers, test bench files,
        #   and generated configuration under context.package_dir(...)
        #   or context.materialize_package(...), depending on the package type.
        # - Persist metadata describing the selected stages, parameters,
        #   target backend, and generated top function.

        # Prepare hls_config.cfg.
        # - Copy the provided hls_config template into the materialized package,
        #   or create a new hls_config.cfg from task fields and composer output.
        # - Apply configuration in this order:
        #   hls_config template < config_defaults < Individual.design["hls"] <
        #   config_overrides.
        # - Ignore unrelated design domains such as Individual.design["system"];
        #   other evaluation steps can consume those system-level parameters.
        # - Keep the HLS domain generic; the composer decides which keys become
        #   hls_config.cfg entries and which become C/C++ macros or generated
        #   source changes.
        # - Use Vitis HLS config keys such as:
        #   part=<device>
        #   [hls]
        #   clock=<period>
        #   flow_target=vivado|vitis
        #   syn.file=<source.cpp>
        #   syn.top=<top_function>
        #   tb.file=<testbench.cpp>

        # Execute HLS synthesis.
        # - Invoke the configured HLS tool through context.run_command(...):
        #   v++ --compile --mode hls --config hls_config.cfg --work_dir <work>
        # - Capture stdout, stderr, return code, and command line.
        # - Decide whether failures should raise immediately or return a
        #   structured failure artifact with logs attached.
        # - Keep generated reports and logs under context.artifact_path(...).

        # Parse synthesis reports.
        # - Locate utilization, timing, latency, and initiation interval reports.
        # - Extract metrics such as LUT, FF, BRAM, DSP, latency min/max,
        #   achieved clock, timing slack, and II.
        # - Normalize metric names so objectives can consume them consistently.

        # Return artifacts consumed by EvaluationExecutor.
        # - Return a metric artifact with synthesis metrics.
        # - Optionally return artifacts for generated source package, TCL
        #   compatibility scripts, raw reports, logs, and failed-run diagnostics.
        raise NotImplementedError(
            "HLS synthesis evaluation requires hls_config.cfg materialization, tool invocation, "
            "and report parser implementation."
        )


@dataclass(frozen=True, slots=True)
class HLSSynthesisEvaluationStep(EvaluationStep):
    """Creates tasks for HLS synthesis evaluation."""

    id: str = "hls_synthesis"
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
    task_type: type[EvaluationTask] = HLSSynthesisTask

    def create_task(
        self,
        individual: Individual,
        artifacts: Mapping[str, Artifact],
    ) -> EvaluationTask:
        return HLSSynthesisTask(
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


__all__ = ["HLSSynthesisEvaluationStep", "HLSSynthesisTask"]
