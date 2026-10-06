"""``python -m jarvis.bench``: run Jarvis's scenarios against the configured model.

Exit codes: 0 every run passed, 1 Jarvis could not start, 2 usage error (bad arguments,
configuration or scenario file), 3 some runs failed, 130 interrupted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from jarvis.bench.report import render, to_json_data
from jarvis.bench.runner import RunResult, run_bench
from jarvis.bench.scenarios import DEFAULT_SCENARIOS, Scenario, ScenarioError, load_scenarios
from jarvis.utils.config import expand_path, load_config, resolve_config_path
from jarvis.utils.logger import configure_stdio, get_logger, setup_logging

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_STARTUP_FAILED = 1
EXIT_USAGE = 2
EXIT_FAILED_RUNS = 3
EXIT_INTERRUPTED = 130


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jarvis.bench",
        description="Run Jarvis's scenarios against the configured model and report pass rate, "
                    "latency, tool-call quality and memory.")
    parser.add_argument("--config", default=None, help="config file (default: auto-detect)")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS,
                        help="scenario file (default: the one shipped with Jarvis)")
    parser.add_argument("--only", default=None, metavar="ID[,ID...]",
                        help="run only these scenarios")
    parser.add_argument("--repeat", type=_positive, default=1,
                        help="attempts per scenario; models are not deterministic (default: 1)")
    parser.add_argument("--report", type=Path, default=None, metavar="FILE",
                        help="also write the full results as JSON")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    return parser.parse_args(argv)


def select(scenarios: list[Scenario], only: str | None) -> list[Scenario]:
    """The scenarios named in ``only`` (comma separated), in file order; all when it is None."""
    if only is None:
        return scenarios
    wanted = [name.strip() for name in only.split(",") if name.strip()]
    known = {scenario.id for scenario in scenarios}
    unknown = [name for name in wanted if name not in known]
    if unknown or not wanted:
        raise ScenarioError(f"unknown scenario(s): {', '.join(unknown) or only!r}. "
                            f"Available: {', '.join(sorted(known))}")
    return [scenario for scenario in scenarios if scenario.id in wanted]


def _progress(result: RunResult) -> None:
    detail = "" if result.passed else "  " + "; ".join(result.failures)
    print(f"  {'ok  ' if result.passed else 'FAIL'} {result.scenario} (run {result.run}) "
          f"{result.seconds:.1f}s{detail}", flush=True)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        scenarios = select(load_scenarios(args.scenarios), args.only)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        print(f"jarvis.bench: cannot load configuration: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ScenarioError as exc:
        print(f"jarvis.bench: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.list:
        for scenario in scenarios:
            print(f"{scenario.id}: {' / '.join(scenario.messages)}")
        return EXIT_OK

    # Only problems on the console, so the table is readable; the log file keeps everything.
    setup_logging(level="WARNING", log_dir=str(expand_path(config.system.log_dir)))
    config_file = resolve_config_path(args.config)
    print(f"Running {len(scenarios)} scenario(s) x {args.repeat} on "
          f"{config.llm.provider} / {config.llm.model} (config: {config_file or 'defaults'})",
          flush=True)
    try:
        report = await run_bench(config, scenarios, repeat=args.repeat, on_result=_progress)
    except RuntimeError as exc:
        print(f"jarvis.bench: Jarvis could not start: {exc}", file=sys.stderr)
        return EXIT_STARTUP_FAILED
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("jarvis.bench: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED

    report.config = str(config_file) if config_file else None
    print()
    print(render(report), end="")
    if args.report is not None:
        try:
            args.report.write_text(json.dumps(to_json_data(report), indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"jarvis.bench: cannot write {args.report}: {exc.strerror or exc}",
                  file=sys.stderr)
            return EXIT_USAGE
        print(f"Results written to {args.report}")
    return EXIT_OK if all(run.passed for run in report.results) else EXIT_FAILED_RUNS


def cli() -> None:
    configure_stdio()
    try:
        code = asyncio.run(main())
    except KeyboardInterrupt:
        code = EXIT_INTERRUPTED
    sys.exit(code)


if __name__ == "__main__":
    cli()
