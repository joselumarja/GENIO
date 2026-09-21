"""Convenience facade combining CSV and population-analysis statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from genio.statistics.composite import CompositeStatisticsCollector
from genio.statistics.csv import CSVStatisticsCollector
from genio.statistics.population import PopulationAnalysisCollector
from genio.statistics.population_plots import PopulationPlotConfig


class PopulationStatisticsCollector(CompositeStatisticsCollector):
    """Generate CSV statistics and population analysis through one collector.

    Args:
        output_dir: Root containing CSV/JSON run files and the ``analysis`` folder.
        individuals_filename: Filename used by the CSV collector.
        plots: Population plot cadence, selection, format, and failure policy.

    Checkpointing is available because both child collectors reconstruct their
    histories from the authoritative evaluations stored by the session.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        individuals_filename: str = "individuals.csv",
        plots: PopulationPlotConfig | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.csv_collector = CSVStatisticsCollector(
            self.output_dir,
            individuals_filename=individuals_filename,
        )
        self.analysis_collector = PopulationAnalysisCollector(
            self.output_dir / "analysis",
            plots=plots,
        )
        super().__init__((self.csv_collector, self.analysis_collector))

    def snapshot(self) -> dict[str, Any]:
        """Return user-facing CSV and population-analysis snapshots."""

        return {
            "csv": self.csv_collector.snapshot(),
            "analysis": self.analysis_collector.snapshot(),
        }


__all__ = ["PopulationStatisticsCollector"]
