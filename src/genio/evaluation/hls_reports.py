from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


HLS_SYNTHESIS_ORIGIN = "hls_synthesis"


class HLSReportParseError(RuntimeError):
    """Indicate that required Vitis HLS report artifacts cannot be located."""


@dataclass(frozen=True, slots=True)
class ParsedHLSReport:
    """Carry normalized metrics and provenance for one HLS report origin.

    Attributes:
        origin: Stable identifier for the HLS stage that generated the report.
        report_paths: Existing summary, text, and XML report paths discovered.
        metrics: Normalized timing, latency, throughput, and resource values.
        metadata: Tool, part, top-function, flow, and source-report information.
    """

    origin: str
    report_paths: tuple[Path, ...]
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def parse_hls_synthesis_report(
    work_dir: Path,
    *,
    top_function: str | None = None,
) -> ParsedHLSReport:
    """Parse reports produced by ``v++ --compile --mode hls``.

    Metrics are read from the selected ``csynth.xml`` and normalized to floats.
    Discovered report paths also include files referenced by
    ``*.hlscompile_summary`` and conventional report locations. Missing XML fields
    and values represented by ``-`` are omitted rather than treated as zero.

    Args:
        work_dir: Vitis HLS work directory containing synthesis output.
        top_function: Optional preferred ``<top>_csynth.xml`` report basename.

    Returns:
        Parsed metrics, metadata, and all existing source report paths.

    Raises:
        HLSReportParseError: If no csynth XML report can be located.
        xml.etree.ElementTree.ParseError: If the selected XML is malformed.
    """

    report_paths = _discover_hls_synthesis_report_paths(work_dir)
    xml_report_path = _select_csynth_xml_report(work_dir, top_function=top_function)
    if xml_report_path is None:
        raise HLSReportParseError(
            f"Could not locate a Vitis HLS csynth XML report under {work_dir}."
        )

    root = ET.parse(xml_report_path).getroot()
    metrics = _parse_csynth_metrics(root)
    metadata = _parse_csynth_metadata(root)
    metadata = {
        **metadata,
        "source_xml_report": str(xml_report_path),
        "report_types": _summary_report_types(work_dir),
    }
    return ParsedHLSReport(
        origin=HLS_SYNTHESIS_ORIGIN,
        report_paths=report_paths,
        metrics=metrics,
        metadata=metadata,
    )


def _discover_hls_synthesis_report_paths(work_dir: Path) -> tuple[Path, ...]:
    """Collect existing synthesis reports from summary and conventional paths."""

    paths: list[Path] = []
    paths.extend(_summary_report_paths(work_dir, report_type="HLS_SYNTHESIS"))
    paths.extend(work_dir.glob("hls/syn/report/csynth.*"))
    paths.extend(work_dir.glob("hls/syn/report/*_csynth.*"))
    paths.extend(work_dir.glob("reports/hls_compile.rpt"))
    return _deduplicate_existing_paths(paths)


def _summary_report_paths(work_dir: Path, *, report_type: str) -> tuple[Path, ...]:
    """Extract report paths of one type from Vitis HLS compile summaries.

    Summary files are treated as Vitis-generated JSON-like text because report
    blocks are extracted independently rather than decoding the whole document.
    """

    paths: list[Path] = []
    for summary_path in work_dir.glob("*.hlscompile_summary"):
        content = summary_path.read_text(encoding="utf-8")
        for block in re.findall(r'"report"\s*:\s*\{(.*?)\}', content, flags=re.DOTALL):
            parsed_type = _json_string_value(block, "reportType")
            if parsed_type != report_type:
                continue
            path = _json_string_value(block, "path")
            if path:
                paths.append(Path(path))
    return tuple(paths)


def _summary_report_types(work_dir: Path) -> tuple[str, ...]:
    report_types: list[str] = []
    for summary_path in work_dir.glob("*.hlscompile_summary"):
        content = summary_path.read_text(encoding="utf-8")
        report_types.extend(re.findall(r'"reportType"\s*:\s*"([^"]+)"', content))
    return tuple(dict.fromkeys(report_types))


def _json_string_value(content: str, key: str) -> str | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', content)
    return match.group(1) if match else None


def _select_csynth_xml_report(work_dir: Path, *, top_function: str | None) -> Path | None:
    """Select the preferred top-specific, canonical, or first csynth XML report."""

    report_dir = work_dir / "hls" / "syn" / "report"
    candidates: list[Path] = []
    if top_function:
        candidates.append(report_dir / f"{top_function}_csynth.xml")
    candidates.append(report_dir / "csynth.xml")
    candidates.extend(sorted(report_dir.glob("*_csynth.xml")))

    for path in candidates:
        if path.exists():
            return path
    return None


def _parse_csynth_metadata(root: ET.Element) -> dict[str, Any]:
    """Extract available tool, target, top-function, and flow metadata."""

    return {
        key: value
        for key, value in {
            "tool_version": _text(root, "ReportVersion/Version"),
            "product_family": _text(root, "UserAssignments/ProductFamily"),
            "part": _text(root, "UserAssignments/Part"),
            "top_function": _text(root, "UserAssignments/TopModelName"),
            "flow_target": _text(root, "UserAssignments/FlowTarget"),
        }.items()
        if value is not None
    }


def _parse_csynth_metrics(root: ET.Element) -> dict[str, float]:
    """Normalize csynth timing, latency, throughput, and resource estimates.

    Both ``bram``/``available_bram`` and explicit ``bram_18k`` aliases are
    emitted for compatibility with consumers naming the physical Vitis resource.
    """

    metrics: dict[str, float] = {}
    _add_metric(metrics, "target_clock_period_ns", root, "UserAssignments/TargetClockPeriod")
    _add_metric(metrics, "clock_uncertainty_ns", root, "UserAssignments/ClockUncertainty")
    _add_metric(
        metrics,
        "estimated_clock_period_ns",
        root,
        "PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod",
    )
    _add_metric(
        metrics,
        "latency_min_cycles",
        root,
        "PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency",
    )
    _add_metric(
        metrics,
        "latency_avg_cycles",
        root,
        "PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency",
    )
    _add_metric(
        metrics,
        "latency_max_cycles",
        root,
        "PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency",
    )
    _add_metric(
        metrics,
        "ii_min",
        root,
        "PerformanceEstimates/SummaryOfOverallLatency/Interval-min",
    )
    _add_metric(
        metrics,
        "ii_max",
        root,
        "PerformanceEstimates/SummaryOfOverallLatency/Interval-max",
    )
    _add_metric(
        metrics,
        "dataflow_pipeline_throughput",
        root,
        "PerformanceEstimates/SummaryOfOverallLatency/DataflowPipelineThroughput",
    )

    resource_paths = {
        "bram": "AreaEstimates/Resources/BRAM_18K",
        "bram_18k": "AreaEstimates/Resources/BRAM_18K",
        "dsp": "AreaEstimates/Resources/DSP",
        "ff": "AreaEstimates/Resources/FF",
        "lut": "AreaEstimates/Resources/LUT",
        "uram": "AreaEstimates/Resources/URAM",
        "available_bram": "AreaEstimates/AvailableResources/BRAM_18K",
        "available_bram_18k": "AreaEstimates/AvailableResources/BRAM_18K",
        "available_dsp": "AreaEstimates/AvailableResources/DSP",
        "available_ff": "AreaEstimates/AvailableResources/FF",
        "available_lut": "AreaEstimates/AvailableResources/LUT",
        "available_uram": "AreaEstimates/AvailableResources/URAM",
    }
    for metric_name, path in resource_paths.items():
        _add_metric(metrics, metric_name, root, path)

    return metrics


def _add_metric(metrics: dict[str, float], name: str, root: ET.Element, path: str) -> None:
    value = _float_text(root, path)
    if value is not None:
        metrics[name] = value


def _text(root: ET.Element, path: str) -> str | None:
    value = root.findtext(path)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _float_text(root: ET.Element, path: str) -> float | None:
    """Extract the first numeric token from an XML field, omitting ``-`` values."""

    value = _text(root, path)
    if value is None or value == "-":
        return None
    numeric_match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value)
    if numeric_match is None:
        return None
    return float(numeric_match.group(0))


def _deduplicate_existing_paths(paths: list[Path]) -> tuple[Path, ...]:
    """Keep existing paths once while preserving their first-seen order."""

    existing_paths: dict[Path, None] = {}
    for path in paths:
        if path.exists():
            existing_paths[path] = None
    return tuple(existing_paths)


__all__ = [
    "HLSReportParseError",
    "HLS_SYNTHESIS_ORIGIN",
    "ParsedHLSReport",
    "parse_hls_synthesis_report",
]
