"""Run NSGA-II functional, HLS, and X-HEEP insect optimization."""

import os
from pathlib import Path
import re
import signal
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genio import (  # noqa: E402
    CSVStatisticsCollector,
    EvaluationWorkflow,
    HLSImagePipelineComposer,
    HLSImagePipelineSynthesisEvaluationStep,
    LFUArtifactCache,
    MetricObjective,
    OptimizationSession,
    ObjectiveSet,
    OptimizationDirection,
    ParallelLocalBackend,
    PythonImageFunctionalEvaluationStep,
    PythonImagePipelineComposer,
    NSGA2Search,
    SearchSpace,
    GRHeepConfigurationComposer,
    XHeepVerilatorSimulationEvaluationStep,
)


def active_vitis_version() -> str:
    configured = os.environ.get("GENIO_VITIS_VERSION")
    if configured:
        return configured

    vitis_root = os.environ.get("XILINX_VITIS")
    if not vitis_root:
        raise RuntimeError(
            "Source the desired Vitis settings or set GENIO_VITIS_VERSION."
        )
    path = Path(vitis_root)
    for candidate in (path.name, path.parent.name):
        if re.fullmatch(r"20\d{2}\.\d+", candidate):
            return candidate
    raise RuntimeError(f"Cannot infer Vitis version from XILINX_VITIS={vitis_root!r}.")


IMAGES_PATH = Path("/home/joselu/Universidad/Doctorado/Datasets/Olive_Fly/Images")
MASKS_PATH = Path("/home/joselu/Universidad/Doctorado/Datasets/Olive_Fly/Masks")
VITIS_LIBRARIES_PATH = Path(
    os.environ.get("VITIS_LIBRARIES_PATH", ROOT / "Vitis_Libraries")
)
HLS_IMPLEMENTATIONS_INCLUDE_PATH = ROOT / "hls_implementations/include"
OUTPUT_DIR = ROOT / "tmp/insect_xheep_nsga2_search"

MAX_EVALUATIONS = 50
BATCH_SIZE = 50
MAX_GENERATIONS = (MAX_EVALUATIONS + BATCH_SIZE - 1) // BATCH_SIZE
MAX_WORKERS = 16
SEED = 0

# Keep a real-image-sized workload that fits comfortably in internal X-HEEP SRAM.
ROWS = 108
COLS = 192
FPGA_PART = "xa7a100tcsg324-1I"
HLS_TIMEOUT_SECONDS = 30 * 60

GR_HEEP_PATH = Path("/home/joselu/Universidad/Doctorado/GEN-HEEP")
XHEEP_TIMEOUT_SECONDS = 30 * 60


def stop_on_signal(signum, _frame) -> None:
    raise SystemExit(128 + signum)


def main() -> None:
    search_space = SearchSpace(
        ROOT / "search_space/tests/insect_xheep_exploration_pipeline.json",
        ROOT / "search_space/stages/definitions",
    )
    algorithm = NSGA2Search(
        population_size=BATCH_SIZE,
        max_generations=MAX_GENERATIONS,
        seed=SEED,
    )

    functional_step = PythonImageFunctionalEvaluationStep(
        composer=PythonImagePipelineComposer(ROOT / "search_space/stages/definitions"),
        images_path=IMAGES_PATH,
        references_path=MASKS_PATH,
        metrics=("mask_f1",),
    )
    hls_step = HLSImagePipelineSynthesisEvaluationStep(
        depends_on=(functional_step.id,),
        composer=HLSImagePipelineComposer(
            ROOT / "search_space/stages/definitions",
            templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
            vitis_version=active_vitis_version(),
            rows=ROWS,
            cols=COLS,
            interface="safa_fifo",
        ),
        part=FPGA_PART,
        metadata={"execution": {"timeout_seconds": HLS_TIMEOUT_SECONDS}},
    )

    xheep_step = XHeepVerilatorSimulationEvaluationStep(
        depends_on=(hls_step.id,),
        composer=GRHeepConfigurationComposer(
            ROOT / "search_space/stages/definitions",
            templates_path=ROOT / "gr_heep_templates",
            application_name="genio_trans_flash_mem",
        ),
        gr_heep_path=GR_HEEP_PATH,
        metadata={"execution": {"timeout_seconds": XHEEP_TIMEOUT_SECONDS}},
    )
    workflow = EvaluationWorkflow((functional_step, hls_step, xheep_step))
    objective_set = ObjectiveSet(
        (
            MetricObjective(
                metric=f"{functional_step.id}.mask_f1",
                direction=OptimizationDirection.MAXIMIZE,
                name="mask_f1",
                normalization_bounds=(0.0, 1.0),
            ),
            MetricObjective(
                metric=f"{hls_step.id}.hls_synthesis.lut",
                direction=OptimizationDirection.MINIMIZE,
                name="hls_lut",
            ),
            MetricObjective(
                metric=f"{xheep_step.id}.xheep_verilator.application_cycles",
                direction=OptimizationDirection.MINIMIZE,
                name="application_cycles",
            ),
        )
    )

    cache = LFUArtifactCache(
        {functional_step.id: 64, hls_step.id: 16, xheep_step.id: 8}
    )
    statistics = CSVStatisticsCollector(OUTPUT_DIR / "statistics")

    previous_sigterm = signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        with ParallelLocalBackend(
            max_workers=MAX_WORKERS,
            base_work_dir=OUTPUT_DIR / "work",
            metadata={
                "vitis_libraries_path": str(VITIS_LIBRARIES_PATH.resolve()),
                "hls_include_paths": [
                    str(HLS_IMPLEMENTATIONS_INCLUDE_PATH.resolve()),
                ],
                "gr_heep_path": str(GR_HEEP_PATH),
                "xheep_timeout_seconds": XHEEP_TIMEOUT_SECONDS,
            },
        ) as backend:
            result = OptimizationSession(
                id="insect_xheep_nsga2_search",
                run_id="insect_xheep_nsga2_search",
                search_space=search_space,
                algorithm=algorithm,
                backend=backend,
                evaluation_workflow=workflow,
                statistics=statistics,
                artifact_cache=cache,
                objective_set=objective_set,
            ).run()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    print(f"Evaluations: {len(result.evaluations)}")
    print(f"CSV: {result.statistics['individuals_csv']}")
    for evaluation in result.evaluations:
        print(
            evaluation.individual.id,
            evaluation.result.status.value,
            evaluation.result.error,
        )


if __name__ == "__main__":
    main()
