"""Run UCAudio goat-bleating training in a supplied Python environment."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from genio import EvaluationOutput, EvaluationTask, TaskContext


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass(frozen=True, slots=True)
class GoatBleatingClassificationTask(EvaluationTask):
    """Evaluate selected audio features and return UCAudio's weighted scores."""

    repository: Path
    templates: Path
    venv: Path
    dataset: Path
    epochs: int = 30
    batch_size: int = 32

    def run(self, context: TaskContext) -> EvaluationOutput:
        source = context.workspace / "ucaudio"
        shutil.copytree(self.repository.expanduser().resolve(), source)
        models = context.workspace / "models"
        logs = context.workspace / "logs"
        models.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)

        parameters = self.individual.values["preproccessing_configuration"]
        replacements = {
            "SAMPLE_RATE": parameters["sample_rate"],
            "WINDOWS_SIZE_SAMPLES": parameters["window_size_samples"],
            "WINDOW_HOP_SAMPLES": parameters["window_hop_samples"],
            "FFT_SIZE": parameters["FFT_size"],
            "HOP_LENGTH": parameters["Hop_length"],
            "MEL_BAND_NUMBER": parameters["MEL_band_number"],
            "LOWER_BAND_LIMIT_HZ": parameters["lower_band_limit_hz"],
            "UPPER_BAND_LIMIT_HZ": parameters["upper_band_limit_hz"],
            "DATASET_PATH": str(self.dataset.expanduser().resolve()),
            "RESULT_PATH": str(models),
            "EPOCHS": self.epochs,
            "BATCH_SIZE": self.batch_size,
        }
        for filename in ("FeatureParams.py", "main.py"):
            rendered = (self.templates / f"{filename}.tpl").read_text(encoding="utf-8")
            for token, value in replacements.items():
                rendered = rendered.replace(f"@{token}@", repr(value))
            (source / filename).write_text(rendered, encoding="utf-8")

        stdout_path = logs / "train_stdout.log"
        stderr_path = logs / "train_stderr.log"

        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            subprocess.run(
                (
                    str(self.venv.expanduser().resolve() / "bin/python"),
                    "-B",
                    str(source / "main.py"),
                ),
                cwd=source,
                stdout=stdout,
                stderr=stderr,
                check=True,
            )

        report = stdout_path.read_text(encoding="utf-8")
        metrics = {}
        for name, label in (
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1-score"),
            ("accuracy", "Test Accuracy"),
            ("loss", "Test Loss"),
            ("best_val_accuracy", "Best Val Accuracy"),
        ):
            match = re.search(rf"{re.escape(label)}:\s*({_NUMBER})", report)
            if match is None:
                raise ValueError(f"UCAudio did not report {label!r}; see {stdout_path}.")
            metrics[name] = float(match.group(1))
        return EvaluationOutput(metrics=metrics)
