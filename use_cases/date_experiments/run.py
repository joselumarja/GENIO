"""Run an NSGA-II insect search with CSV and population-analysis outputs."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genio import (  # noqa: E402
    EvaluationStep,
    EvaluationWorkflow,
    MetricObjective,
    NSGA2Search,
    ObjectiveSet,
    OptimizationDirection,
    OptimizationSession,
    PopulationPlotConfig,
    PopulationStatisticsCollector,
    SearchSpace,
)
from genio_use_cases.image_pipeline import (  # noqa: E402
    HLSImagePipelineComposer,
    HLSImagePipelineSynthesisTask,
    PythonImageFunctionalTask,
    PythonImagePipelineComposer,
)
from genio_use_cases.xheep import (  # noqa: E402
    GRHeepConfigurationComposer,
    XHeepVerilatorSimulationTask,
)


IMAGES_PATH = Path("/home/joselu/Universidad/Doctorado/Datasets/Background_Extraction_Grayscale/Images")
MASKS_PATH = Path("/home/joselu/Universidad/Doctorado/Datasets/Background_Extraction_Grayscale/Masks")
VITIS_LIBRARIES_PATH = Path(
    os.environ.get("VITIS_LIBRARIES_PATH", ROOT / "Vitis_Libraries")
)
DEFINITIONS_PATH = ROOT / "search_space/stages/definitions"
HLS_TEMPLATES_PATH = ROOT / "use_cases/insect_segmentation/hls_templates/vitis_vision_image_pipeline"
HLS_IMPLEMENTATIONS_INCLUDE_PATH = ROOT / "use_cases/insect_segmentation/hls_templates/vitis_vision_image_pipeline/include"
GR_HEEP_TEMPLATES_PATH = ROOT / "use_cases/date_experiments/gr_heep_templates"
GR_HEEP_PATH = Path("/home/joselu/Universidad/Doctorado/GEN-HEEP")

POPULATION_SIZE = int(os.environ.get("GENIO_POPULATION_SIZE", 48))
GENERATIONS = int(os.environ.get("GENIO_GENERATIONS", 10))
MAX_WORKERS = int(os.environ.get("GENIO_MAX_WORKERS", 16))
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
    return next(
        path
        for path in sorted(images_path.iterdir())
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )


def main(case_name) -> None:
    """Build and execute the complete multi-generation exploration."""

    output_dir = Path(
        os.environ.get(
            "GENIO_OUTPUT_DIR",
            ROOT / f"tmp/insect_xheep_nsga2_population_search_{case_name}",
        )
    )

    search_space = SearchSpace(
        ROOT / "use_cases/insect_segmentation/scenarios/insect_xheep_exploration_pipeline.json",
        DEFINITIONS_PATH,
    )
    algorithm = NSGA2Search(
        population_size=POPULATION_SIZE,
        max_generations=GENERATIONS,
        eliminate_duplicates=True,
        seed=SEED,
    )
    functional_step = EvaluationStep(
        id="python_image_functional",
        task_class=PythonImageFunctionalTask,
        config={
            "composer": PythonImagePipelineComposer(DEFINITIONS_PATH),
            "images_path": IMAGES_PATH,
            "references_path": MASKS_PATH,
            "rows": ROWS,
            "cols": COLS,
            "metrics": ("mask_f1",),
        },
    )
    hls_step = EvaluationStep(
        id="hls_image_pipeline_synthesis",
        task_class=HLSImagePipelineSynthesisTask,
        config={
            "composer": HLSImagePipelineComposer(
                DEFINITIONS_PATH,
                templates_path=HLS_TEMPLATES_PATH,
                vitis_version=(
                    os.environ.get("GENIO_VITIS_VERSION")
                    or Path(os.environ["XILINX_VITIS"]).name
                ),
                image_type="XF_8UC1",
                rows=ROWS,
                cols=COLS,
                interface="safa_fifo",
            ),
            "part": FPGA_PART,
        },
        timeout_seconds=HLS_TIMEOUT_SECONDS,
    )
    xheep_step = EvaluationStep(
        id="xheep_verilator_simulation",
        task_class=XHeepVerilatorSimulationTask,
        config={
            "composer": GRHeepConfigurationComposer(
                DEFINITIONS_PATH,
                templates_path=GR_HEEP_TEMPLATES_PATH,
                application_name=case_name,
            ),
            "gr_heep_path": GR_HEEP_PATH,
            "input_image_path": first_input_image(IMAGES_PATH),
        },
        timeout_seconds=XHEEP_TIMEOUT_SECONDS,
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

    statistics = PopulationStatisticsCollector(
        output_dir / "statistics",
        plots=PopulationPlotConfig(
            every_batches=1,
            image_format="png",
            final_plots=True,
            strict=False,
        ),
    )

    previous_sigterm = signal.signal(signal.SIGTERM, stop_on_signal)
    try:
        result = OptimizationSession(
            id=f"insect_xheep_nsga2_population_search_{case_name}",
            run_id=f"insect_xheep_nsga2_population_search_{case_name}",
            search_space=search_space,
            algorithm=algorithm,
            max_workers=MAX_WORKERS,
            work_dir=output_dir / "work",
            execution_metadata={
                "vitis_libraries_path": str(VITIS_LIBRARIES_PATH.resolve()),
                "hls_include_paths": [
                    str(HLS_IMPLEMENTATIONS_INCLUDE_PATH.resolve()),
                ],
                "gr_heep_path": str(GR_HEEP_PATH),
            },
            evaluation_workflow=workflow,
            statistics=statistics,
            objective_set=objective_set,
            keep_all_individuals=False,
        ).run()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    csv_statistics = result.statistics["csv"]
    analysis_statistics = result.statistics["analysis"]
    print(f"Test: {case_name}")
    print(f"Evaluations: {len(result.evaluations)}")
    print(f"Algorithm solutions: {len(result.best_individuals)}")
    print(f"CSV: {csv_statistics['individuals_csv']}")
    print(f"Analysis summary: {analysis_statistics['analysis_summary']}")
    print(f"Analysis manifest: {analysis_statistics['analysis_manifest']}")
    print(f"Generated plots: {analysis_statistics['generated_plots']}")
    for individual in result.best_individuals:
        print("BEST", individual.id, individual.genotype)


if __name__ == "__main__":
    case_name_list = [
        "genio_trans_mem_mem",
        "genio_trans_mem_mem_sequential_saturation",
        "genio_trans_mem_mem_periodic_saturation",
        "genio_trans_mem_mem_burst_saturation",
        "genio_trans_mem_mem_bernoulli_saturation",
    ]

    for case_name in case_name_list:
        main(case_name)
