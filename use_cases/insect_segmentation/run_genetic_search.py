"""Run insect F1 versus HLS hardware utilization genetic search.

Run ``python -m use_cases.insect_segmentation.run_genetic_search`` from the
repository root after configuring Vitis (``XILINX_VITIS`` or
``GENIO_VITIS_VERSION``), ``VITIS_LIBRARIES_PATH`` and the dataset paths
``GENIO_IMAGES_PATH`` / ``GENIO_MASKS_PATH``.

Each of FF, LUT, BRAM and DSP contributes 25% of the hardware subscore.
F1 and hardware each receive 50% of the final fitness group weighting; edit
RESOURCE_WEIGHTS in hardware_objective.py and OBJECTIVE_WEIGHTS below to tune them.
The source scenario contains system genes, but this run has no system step:
project to HLS plus pipeline with dataclasses.replace, preserving original slots.
Batch min/max ranges are fitted anew per generation for selection, whereas the
global winner uses the same weighted scalarizer on raw maximize-oriented values
so winners across generations are comparable.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from random import Random


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from genio import (  # noqa: E402
    CSVStatisticsCollector,
    EvaluationStep,
    EvaluationWorkflow,
    GeneticSearch,
    MetricObjective,
    MinMaxNormalizer,
    NormalizationScope,
    ObjectiveSet,
    OptimizationDirection,
    OptimizationSession,
    SearchSpace,
    WeightedMeanScalarizer,
)
from genio_use_cases.image_pipeline import (  # noqa: E402
    HLSImagePipelineComposer,
    HLSImagePipelineSynthesisTask,
    PythonImageFunctionalTask,
    PythonImagePipelineComposer,
)
from use_cases.insect_segmentation.hardware_objective import (  # noqa: E402
    HardwareUtilizationObjective,
)


SCENARIO_PATH = ROOT / "use_cases/insect_segmentation/scenarios/insect_segmentation_pipeline.json"
DEFINITIONS_PATH = ROOT / "search_space/stages/definitions"
HLS_TEMPLATES_PATH = ROOT / "hls_templates/vitis_vision_image_pipeline"
HLS_INCLUDE_PATH = ROOT / "hls_implementations/include"
IMAGES_PATH = Path(os.environ.get(
    "GENIO_IMAGES_PATH", "/home/joselu/Universidad/Doctorado/Datasets/Olive_Fly_Grayscale/Images"
))
MASKS_PATH = Path(os.environ.get(
    "GENIO_MASKS_PATH", "/home/joselu/Universidad/Doctorado/Datasets/Olive_Fly_Grayscale/Masks"
))
VITIS_LIBRARIES_PATH = Path(os.environ.get("VITIS_LIBRARIES_PATH", ROOT / "Vitis_Libraries"))
OUTPUT_DIR = Path(os.environ.get("GENIO_OUTPUT_DIR", ROOT / "tmp/insect_genetic_search"))

OBJECTIVE_WEIGHTS = {"f1": 0.5, "hardware": 0.5}
ROWS = 320
COLS = 240
FPGA_PART = "xa7a100tcsg324-1I"
HLS_TIMEOUT_SECONDS = 30 * 60


def positive_integer(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def active_vitis_version() -> str:
    configured = os.environ.get("GENIO_VITIS_VERSION")
    if configured:
        return configured
    vitis_root = os.environ.get("XILINX_VITIS")
    if not vitis_root:
        raise RuntimeError("Source Vitis settings or set GENIO_VITIS_VERSION.")
    path = Path(vitis_root)
    for candidate in (path.name, path.parent.name):
        if re.fullmatch(r"20\d{2}\.\d+", candidate):
            return candidate
    raise RuntimeError(f"Cannot infer Vitis version from XILINX_VITIS={vitis_root!r}.")


def main() -> None:
    """Evaluate only pipeline and HLS genes, never unrelated system choices."""

    source = SearchSpace(SCENARIO_PATH, DEFINITIONS_PATH).scenario
    scenario = replace(
        source,
        design_spaces={"hls": source.design_spaces["hls"]},
        domain_order=tuple(domain for domain in source.domain_order if domain != "system"),
    )
    search_space = SearchSpace.from_scenario(scenario)
    population_size = positive_integer("GENIO_POPULATION_SIZE", 16)
    if population_size % 2:
        raise ValueError("GENIO_POPULATION_SIZE must be even.")
    scalarizer = WeightedMeanScalarizer(weights=OBJECTIVE_WEIGHTS)
    algorithm = GeneticSearch(
        population_size=population_size,
        max_generations=positive_integer("GENIO_GENERATIONS", 5),
        mutation_probability=float(os.environ.get("GENIO_MUTATION_PROBABILITY", "0.05")),
        balanced_initialization=True,
        random=Random(int(os.environ.get("GENIO_SEED", "0"))),
        global_best_scalarizer=scalarizer,
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
                vitis_version=active_vitis_version(),
                image_type="XF_8UC1",
                rows=ROWS,
                cols=COLS,
                interface="safa_fifo",
            ),
            "part": FPGA_PART,
        },
        timeout_seconds=HLS_TIMEOUT_SECONDS,
    )
    objective_set = ObjectiveSet(
        (
            MetricObjective(
                metric=f"{functional_step.id}.mask_f1",
                direction=OptimizationDirection.MAXIMIZE,
                name="f1",
            ),
            HardwareUtilizationObjective(hls_step.id),
        ),
        normalizer=MinMaxNormalizer(scope=NormalizationScope.BATCH),
        scalarizer=scalarizer,
    )
    result = OptimizationSession(
        id="insect_genetic_search",
        run_id="insect_genetic_search",
        search_space=search_space,
        algorithm=algorithm,
        evaluation_workflow=EvaluationWorkflow((functional_step, hls_step)),
        statistics=CSVStatisticsCollector(OUTPUT_DIR / "statistics"),
        objective_set=objective_set,
        max_workers=positive_integer("GENIO_MAX_WORKERS", 16),
        work_dir=OUTPUT_DIR / "work",
        execution_metadata={
            "vitis_libraries_path": str(VITIS_LIBRARIES_PATH.resolve()),
            "hls_include_paths": [str(HLS_INCLUDE_PATH.resolve())],
        },
        keep_all_individuals=False,
    ).run()
    print(f"Evaluations: {len(result.evaluations)}")
    for individual in result.best_individuals:
        print("BEST", individual.id, individual.genotype)
    print(f"CSV: {result.statistics['individuals_csv']}")
    print(f"Summary: {result.statistics['run_summary']}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Work: {OUTPUT_DIR / 'work'}")


if __name__ == "__main__":
    main()
