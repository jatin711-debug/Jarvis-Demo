"""Turn benchmark results into a table people can read and JSON a script can compare."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any

from jarvis.bench.runner import Report, RunResult

_REPLY_SHOWN_CHARS = 160


@dataclass
class ScenarioSummary:
    scenario: str
    passed: int
    total: int
    median_s: float
    first_token_s: float | None  # median
    tools: list[str]  # distinct, in the order first used
    invalid_calls: int


def summarize(results: list[RunResult]) -> list[ScenarioSummary]:
    grouped: dict[str, list[RunResult]] = {}
    for result in results:
        grouped.setdefault(result.scenario, []).append(result)
    summaries = []
    for scenario, runs in grouped.items():
        first = [run.first_token_s for run in runs if run.first_token_s is not None]
        tools: list[str] = []
        for run in runs:
            tools += [call.name for call in run.tool_calls if call.name not in tools]
        summaries.append(ScenarioSummary(
            scenario=scenario,
            passed=sum(run.passed for run in runs),
            total=len(runs),
            median_s=statistics.median(run.seconds for run in runs),
            first_token_s=statistics.median(first) if first else None,
            tools=tools,
            invalid_calls=sum(call.outcome == "invalid" for run in runs for call in run.tool_calls),
        ))
    return summaries


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}s"


def _gb(megabytes: float | None) -> str:
    return "?" if megabytes is None else f"{megabytes / 1024:.1f} GB"


def render(report: Report) -> str:
    """The plain-text report: one row per scenario, then what failed, then the totals."""
    results = report.results
    summaries = summarize(results)
    width = max([len("scenario"), *(len(s.scenario) for s in summaries)])
    lines = [
        f"Jarvis bench: {report.provider} / {report.model}"
        f"   {len(results)} runs, startup {report.startup_s:.1f}s",
        "",
        f"{'scenario'.ljust(width)}  passed  median  first  invalid  tools",
    ]
    for s in summaries:
        lines.append(
            f"{s.scenario.ljust(width)}  {f'{s.passed}/{s.total}'.ljust(6)}  "
            f"{_seconds(s.median_s).ljust(6)}  {_seconds(s.first_token_s).ljust(5)}  "
            f"{str(s.invalid_calls).ljust(7)}  {', '.join(s.tools) or '-'}")

    failed = [run for run in results if not run.passed]
    if failed:
        lines += ["", "Failures"]
        for run in failed:
            lines.append(f"  {run.scenario} (run {run.run}): " + "; ".join(run.failures))
            reply = " ".join(run.reply.split())
            if reply:
                shown = reply if len(reply) <= _REPLY_SHOWN_CHARS else reply[:_REPLY_SHOWN_CHARS] + "..."
                lines.append(f"      reply: {shown}")
            if run.tool_calls:
                lines.append("      tools: " + ", ".join(
                    f"{call.name}" + ("" if call.outcome == "ok" else f" ({call.outcome})")
                    for call in run.tool_calls))

    times = [run.seconds for run in results]
    calls = [call for run in results for call in run.tool_calls]
    passed = sum(run.passed for run in results)
    lines += ["", f"Passed {passed}/{len(results)} runs ({100 * passed // max(1, len(results))}%)"]
    if times:
        lines[-1] += f", median {statistics.median(times):.1f}s, p95 {_percentile(times, 0.95):.1f}s"
    if calls:
        invalid = sum(call.outcome == "invalid" for call in calls)
        lines.append(f"Tool calls: {len(calls)}, rejected for bad arguments: {invalid}")

    memory = report.resources
    parts = []
    if memory.gpu_peak_mb is not None:
        extra = ""
        if memory.gpu_baseline_mb is not None:
            extra = f" (+{_gb(memory.gpu_peak_mb - memory.gpu_baseline_mb)} since start)"
        parts.append(f"GPU in use {_gb(memory.gpu_peak_mb)}{extra}")
    if memory.jarvis_mb is not None:
        parts.append(f"Jarvis {_gb(memory.jarvis_mb)}")
    if memory.server_mb is not None:
        parts.append(f"llama-server {_gb(memory.server_mb)}")
    if parts:
        lines.append("Peak memory: " + ", ".join(parts))
    return "\n".join(lines) + "\n"


def to_json_data(report: Report) -> dict[str, Any]:
    """The report as plain data, with the per-scenario summary alongside the raw runs."""
    data = asdict(report)
    data["summary"] = [asdict(summary) for summary in summarize(report.results)]
    return data
