"""Search UCAudio feature parameters using weighted goat-bleating F1.

Run from the GENIO repository root with ``--venv /path/to/ucaudio/.venv``.
Install UCAudio's ``requirements.txt`` in that environment first; the task
copies UCAudio/src into each workspace, renders FeatureParams.py and main.py from
the templates, then runs that copy with ``venv/bin/python`` without shell activation.
UCAudio's testing split is also used for validation and checkpoint selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from random import Random


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from genio import (  # noqa: E402
    CSVStatisticsCollector,
    EvaluationStep,
    EvaluationWorkflow,
    MetricObjective,
    ObjectiveSet,
    OptimizationDirection,
    OptimizationSession,
    RandomSearch,
    GeneticSearch,
    SearchSpace,
    IdentityScalarizer
)
from genio_use_cases.sound_classification import (  # noqa: E402
    GoatBleatingClassificationTask,
)


SCENARIO = (
    ROOT
    / "use_cases/goat_bleating_classification/scenarios/goat_bleating_classification.json"
)
UCAUDIO_PATH = Path("/home/joselu/Universidad/Doctorado/UCAudio")
UCAUDIO_FILES_PATH = UCAUDIO_PATH / "src"
TEMPLATES_PATH = ROOT / "use_cases/goat_bleating_classification/templates"
VENV_PATH = UCAUDIO_PATH / "venv"
DATASET_PATH = UCAUDIO_PATH / "goat_bleatings"
EPOCHS = 100
BATCH_SIZE = 32
SEED = 0
MAX_EVALUATIONS = 100
MAX_WORKERS = 1

def main() -> None:

    if not (VENV_PATH / "bin/python").is_file():
        raise FileNotFoundError(f"No Python interpreter in {VENV_PATH / 'bin/python'}")

    space = SearchSpace(SCENARIO)
    step = EvaluationStep(
        id="sound_classification",
        task_class=GoatBleatingClassificationTask,
        config={
            "repository": UCAUDIO_FILES_PATH,
            "templates": TEMPLATES_PATH,
            "venv": VENV_PATH,
            "dataset": DATASET_PATH,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        },
    )

    """algorithm = RandomSearch(
        max_evaluations=MAX_EVALUATIONS,
        batch_size=MAX_WORKERS,
        unique=True,
        random=Random(SEED),
    )"""

    algorithm = GeneticSearch(
        population_size=100,
        max_generations=10,
        balanced_initialization=False,
        random=Random(SEED)
    )

    result = OptimizationSession(
        space,
        algorithm,
        EvaluationWorkflow((step,)),
        objective_set=ObjectiveSet(
            (
                MetricObjective(
                    "sound_classification.accuracy",
                    OptimizationDirection.MAXIMIZE,
                    name="accuracy",
                ),
            ),
            scalarizer=IdentityScalarizer()
        ),
        statistics=CSVStatisticsCollector(UCAUDIO_PATH / "statistics"),
        max_workers=MAX_WORKERS,
        work_dir=UCAUDIO_PATH / "work",
        cache_capacity={"sound_classification" : 300},
        keep_all_individuals=False,
    ).run()

    best = max(
        (
            evaluation
            for evaluation in result.evaluations
            if "sound_classification.accuracy" in evaluation.result.metrics
        ),
        key=lambda evaluation: evaluation.result.metrics["sound_classification.accuracy"],
        default=None,
    )
    if best is not None:
        print("BEST", best.individual.id, best.individual.genotype, best.result.metrics)
    print(f"Evaluations: {len(result.evaluations)}")
    print(f"CSV: {result.statistics['individuals_csv']}")


if __name__ == "__main__":
    main()
