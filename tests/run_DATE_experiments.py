"""Run an NSGA-II insect search with CSV and population-analysis outputs."""

from __future__ import annotations

import os
from pathlib import Path
from random import Random
import re
import signal
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genio import (  # noqa: E402
    EvaluationWorkflow,
    GRHeepConfigurationComposer,
    HLSImagePipelineComposer,
    HLSImagePipelineSynthesisEvaluationStep,
    LFUArtifactCache,
    MetricObjective,
    NSGA2Search,
    GeneticSearch,
    RandomSearch,
    ObjectiveSet,
    OptimizationDirection,
    OptimizationSession,
    ParallelLocalBackend,
    PopulationPlotConfig,
    PopulationStatisticsCollector,
    PythonImageFunctionalEvaluationStep,
    PythonImagePipelineComposer,
    SearchSpace,
    XHeepVerilatorSimulationEvaluationStep,
)


def active_vitis_version() -> str:
    """Return the explicitly configured or sourced Vitis release."""

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


def positive_environment_integer(name: str, default: int) -> int:
    """Read a positive integer runner setting from the environment."""

    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


IMAGES_PATH = Path("/home/joselu/Universidad/Doctorado/Datasets/Background_Extraction_Grayscale/Images")
MASKS_PATH = Path("/home/joselu/Universidad/Doctorado/Datasets/Background_Extraction_Grayscale/Masks")
VITIS_LIBRARIES_PATH = Path(
    os.environ.get("VITIS_LIBRARIES_PATH", ROOT / "Vitis_Libraries")
)
HLS_IMPLEMENTATIONS_INCLUDE_PATH = ROOT / "hls_implementations/include"
GR_HEEP_PATH = Path("/home/joselu/Universidad/Doctorado/GEN-HEEP")

POPULATION_SIZE = positive_environment_integer("GENIO_POPULATION_SIZE", 48)
GENERATIONS = positive_environment_integer("GENIO_GENERATIONS", 10)
MAX_WORKERS = positive_environment_integer("GENIO_MAX_WORKERS", 16)
SEED = int(os.environ.get("GENIO_SEED", "0"))

# QVGA
ROWS = 320
COLS = 240

FPGA_PART = "xa7a100tcsg324-1I"
HLS_TIMEOUT_SECONDS = 10 * 60
XHEEP_TIMEOUT_SECONDS = 15 * 60


def stop_on_signal(signum, _frame) -> None:
    """Convert termination signals into a conventional process exit code."""

    raise SystemExit(128 + signum)


def first_input_image(images_path: Path) -> Path:
    """Return the first deterministic grayscale image used by X-HEEP."""

    supported_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    candidates = tuple(
        sorted(
            path
            for path in images_path.iterdir()
            if path.is_file() and path.suffix.lower() in supported_suffixes
        )
    )
    if not candidates:
        raise RuntimeError(f"No supported input images found in {images_path}.")
    return candidates[0]


def main(case_name) -> None:
    """Build and execute the complete multi-generation exploration."""

    OUTPUT_DIR = Path(
    os.environ.get(
        "GENIO_OUTPUT_DIR",
        ROOT / str("tmp/insect_xheep_nsga2_population_search_"+case_name),
    )
)

    definitions_path = ROOT / "search_space/stages/definitions"
    search_space = SearchSpace(
        ROOT / "search_space/tests/insect_xheep_exploration_pipeline.json",
        definitions_path,
    )
    algorithm = NSGA2Search(
        population_size=POPULATION_SIZE,
        max_generations=GENERATIONS,
        eliminate_duplicates=True,
        seed=SEED,
    )
    """algorithm = GeneticSearch(
        population_size=80,
        mutation_probability=0.05,
        max_generations=10,
        balanced_initialization=True
    )"""
    """algorithm = RandomSearch(
        max_evaluations=16,
        batch_size=16,
        unique=True,
        balanced=True,
        random=Random(SEED),
    )"""

    functional_step = PythonImageFunctionalEvaluationStep(
        composer=PythonImagePipelineComposer(definitions_path),
        images_path=IMAGES_PATH,
        references_path=MASKS_PATH,
        rows=ROWS,
        cols=COLS,
        metrics=("mask_f1",),
    )
    hls_step = HLSImagePipelineSynthesisEvaluationStep(
        depends_on=(functional_step.id,),
        composer=HLSImagePipelineComposer(
            definitions_path,
            templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
            vitis_version=active_vitis_version(),
            image_type="XF_8UC1",
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
            definitions_path,
            templates_path=ROOT / "gr_heep_templates",
            application_name=case_name,
        ),
        gr_heep_path=GR_HEEP_PATH,
        input_image_path=first_input_image(IMAGES_PATH),
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
                metric=f"{hls_step.id}.hls_synthesis.ff",
                direction=OptimizationDirection.MINIMIZE,
                name="hls_ff",
            ),
            MetricObjective(
                metric=f"{xheep_step.id}.xheep_verilator.application_cycles",
                direction=OptimizationDirection.MINIMIZE,
                name="application_cycles",
            ),
        )
    )

    """MetricObjective(
                    metric=f"{xheep_step.id}.xheep_verilator.safa_input_fifo_empty_cycles",
                    direction=OptimizationDirection.MINIMIZE,
                    name="safa_input_fifo_empty_cycles",
                ),
                MetricObjective(
                    metric=f"{xheep_step.id}.xheep_verilator.application_cycles",
                    direction=OptimizationDirection.MINIMIZE,
                    name="application_cycles",
                ),"""

    cache = LFUArtifactCache(
        {functional_step.id: 32, hls_step.id: 32, xheep_step.id: 32},
        storage_dir=OUTPUT_DIR / "artifact_cache",
    )
    statistics = PopulationStatisticsCollector(
        OUTPUT_DIR / "statistics",
        plots=PopulationPlotConfig(
            every_batches=1,
            image_format="png",
            final_plots=True,
            strict=False,
        ),
    )

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
                id="insect_xheep_nsga2_population_search_"+case_name,
                run_id="insect_xheep_nsga2_population_search_"+case_name,
                search_space=search_space,
                algorithm=algorithm,
                backend=backend,
                evaluation_workflow=workflow,
                statistics=statistics,
                artifact_cache=cache,
                objective_set=objective_set,
                keep_all_individuals=False,
            ).run()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    csv_statistics = result.statistics["csv"]
    analysis_statistics = result.statistics["analysis"]
    print("Test: "+case_name)
    print(f"Evaluations: {len(result.evaluations)}")
    print(f"Algorithm solutions: {len(result.best_individuals)}")
    print(f"CSV: {csv_statistics['individuals_csv']}")
    print(f"Analysis summary: {analysis_statistics['analysis_summary']}")
    print(f"Analysis manifest: {analysis_statistics['analysis_manifest']}")
    print(f"Generated plots: {analysis_statistics['generated_plots']}")
    for individual in result.best_individuals:
        print("BEST", individual.id, individual.genotype)


if __name__ == "__main__":
    #case_name_list = ["genio_trans_mem_mem", "genio_trans_mem_mem_sequential_saturation", "genio_trans_mem_mem_periodic_saturation", "genio_trans_mem_mem_burst_saturation", "genio_trans_mem_mem_bernoulli_saturation"]
    #case_name_list = ["genio_trans_mem_mem"]
    case_name_list = ["genio_trans_mem_mem", "genio_trans_mem_mem_sequential_saturation", "genio_trans_mem_mem_periodic_saturation", "genio_trans_mem_mem_burst_saturation", "genio_trans_mem_mem_bernoulli_saturation"]

    for case_name in case_name_list:
        main(case_name)
