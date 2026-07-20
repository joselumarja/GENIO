"""Run a small random search over the insect segmentation pipeline."""

import os
from pathlib import Path
from random import Random
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genio import (  # noqa: E402
    CSVStatisticsCollector,
    EvaluationWorkflow,
    HLSImagePipelineComposer,
    HLSImagePipelineSynthesisEvaluationStep,
    LFUArtifactCache,
    OptimizationSession,
    ParallelLocalBackend,
    PythonImageFunctionalEvaluationStep,
    PythonImagePipelineComposer,
    RandomSearch,
    GridSearch,
    SearchSpace,
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


VITIS_VERSION = active_vitis_version()
IMAGES_PATH = Path("/home/joselu/Universidad/Doctorado/Datasets/Olive_Fly/Images")
MASKS_PATH = Path("/home/joselu/Universidad/Doctorado/Datasets/Olive_Fly/Masks")
VITIS_LIBRARIES_PATH = Path(
    os.environ.get("VITIS_LIBRARIES_PATH", ROOT / "Vitis_Libraries")
)
HLS_IMPLEMENTATIONS_INCLUDE_PATH = ROOT / "hls_implementations/include"
OUTPUT_DIR = ROOT / "tmp/insect_random_search"

MAX_EVALUATIONS = 10
BATCH_SIZE = 2
MAX_WORKERS = 2
SEED = 0

ROWS = 2160
COLS = 3840
FPGA_PART = "xa7a100tcsg324-1I"
HLS_TIMEOUT_SECONDS = 5 * 60


def main() -> None:
    search_space = SearchSpace(ROOT / "search_space/tests/insect_segmentation_pipeline.json", ROOT / "search_space/stages/definitions")
    algorithm = RandomSearch(max_evaluations=MAX_EVALUATIONS, batch_size=BATCH_SIZE, unique=True, balanced=True, random=Random(SEED))
    #search_space = SearchSpace(ROOT / "search_space/tests/tfg_pipeline.json", ROOT / "search_space/stages/definitions")
    #algorithm = GridSearch()

    functional_step = PythonImageFunctionalEvaluationStep(
        composer=PythonImagePipelineComposer(ROOT / "search_space/stages/definitions"),
        images_path=IMAGES_PATH,
        references_path=MASKS_PATH,
        metrics=("mask_f1", ),
    )
    hls_step = HLSImagePipelineSynthesisEvaluationStep(
        depends_on=(functional_step.id,),
        composer=HLSImagePipelineComposer(
            ROOT / "search_space/stages/definitions",
            templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
            vitis_version=VITIS_VERSION,
            rows=ROWS,
            cols=COLS,
        ),
        part=FPGA_PART,
        metadata={
            "execution": {
                "timeout_seconds": HLS_TIMEOUT_SECONDS,
            }
        },
    )
    workflow = EvaluationWorkflow((functional_step, hls_step))

    cache = LFUArtifactCache(
        {
            functional_step.id: 64,
            hls_step.id: 16,
        }
    )
    statistics = CSVStatisticsCollector(OUTPUT_DIR / "statistics")

    with ParallelLocalBackend(
        max_workers=MAX_WORKERS,
        base_work_dir=OUTPUT_DIR / "work",
        metadata={
            "vitis_libraries_path": str(VITIS_LIBRARIES_PATH.resolve()),
            "hls_include_paths": [
                str(HLS_IMPLEMENTATIONS_INCLUDE_PATH.resolve()),
            ],
        },
    ) as backend:
        result = OptimizationSession(
            id="insect_random_search",
            run_id="insect_random_search",
            search_space=search_space,
            algorithm=algorithm,
            backend=backend,
            evaluation_workflow=workflow,
            statistics=statistics,
            artifact_cache=cache,
        ).run()

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
