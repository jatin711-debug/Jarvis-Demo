"""Text CLI for inspecting and running durable tasks without audio or LLMs.

Exit codes: 0 ok (completed, paused, cancelled, or a read-only command), 1 the command failed
(for example the desktop backend would not start), 2 usage error (bad arguments, unknown task
id, unreadable configuration), 3 the run ended in ``needs_review``, 130 interrupted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

from jarvis.agent.permissions import PermissionGateway
from jarvis.desktop.cua import CuaDesktop
from jarvis.tasks.contracts import MAX_BUDGET_ACTIONS, MAX_BUDGET_SECONDS, Budget
from jarvis.tasks.runner import TaskRunner
from jarvis.tasks.store import WORKFLOW_STEPS, StateConflict, TaskStore
from jarvis.tasks.workflows import get_workflow
from jarvis.utils.config import JarvisConfig, expand_path, load_config
from jarvis.utils.logger import configure_stdio, get_logger, setup_logging

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2           # argparse also exits 2 for malformed arguments
EXIT_NEEDS_REVIEW = 3
EXIT_INTERRUPTED = 130

_TERMINAL = frozenset({"completed", "cancelled"})
_RUN_COMMANDS = frozenset({"start-fixture", "start-calculator", "resume"})


class CliError(Exception):
    """A failure reported as one stderr line and an exit code, never a traceback."""

    def __init__(self, message: str, code: int = EXIT_FAILED) -> None:
        super().__init__(message)
        self.code = code


def _whole(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None


def _positive(text: str) -> int:
    value = _whole(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _action_cap(text: str) -> int:
    value = _positive(text)
    if value > MAX_BUDGET_ACTIONS:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_BUDGET_ACTIONS}")
    return value


def _seconds_cap(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or not 0 < value <= MAX_BUDGET_SECONDS:
        raise argparse.ArgumentTypeError(
            f"must be greater than 0 and at most {MAX_BUDGET_SECONDS:g}")
    return value


def _budget_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--max-seconds", type=_seconds_cap, default=None,
                         help="Cap on run time across resumes (default: per workflow)")
    command.add_argument("--max-actions", type=_action_cap, default=None,
                         help="Cap on mutating actions (default: per workflow)")


def _pause_option(command: argparse.ArgumentParser) -> None:
    command.add_argument("--pause-after", type=_positive, default=None,
                         help="Pause after N verified steps")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m jarvis.tasks",
                                     description="Jarvis durable task CLI")
    parser.add_argument("--config", default=None)
    parser.add_argument("--db", type=Path, default=None, help="Task database path")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start-fixture", help="Save text in an existing Jarvis smoke fixture")
    start.add_argument("--window-title", required=True)
    start.add_argument("--output", type=Path, required=True)
    start.add_argument("--text", required=True)
    _pause_option(start)
    _budget_options(start)
    calculator = commands.add_parser("start-calculator", help="Launch Calculator and verify 1+2=3")
    _pause_option(calculator)
    _budget_options(calculator)
    status = commands.add_parser("status")
    status.add_argument("task_id")
    resume = commands.add_parser("resume")
    resume.add_argument("task_id")
    _pause_option(resume)
    _budget_options(resume)
    cancel = commands.add_parser("cancel")
    cancel.add_argument("task_id")
    commands.add_parser("list")
    return parser.parse_args(argv)


def _public(task: dict, events: list[dict] | None = None) -> dict:
    result = {key: task[key] for key in ("id", "goal", "workflow", "status", "version", "current_step",
                                          "scope", "step_status", "error", "result", "budget",
                                          "actions_taken", "elapsed_seconds")}
    steps = WORKFLOW_STEPS.get(task["workflow"], ())
    result["next_step"] = (steps[task["current_step"]]
                           if task["current_step"] < len(steps) else None)
    result["evidence"] = task["evidence"]
    if events is not None:
        result["events"] = events
    return result


def _budget(base: Budget, args: argparse.Namespace) -> Budget | None:
    """The requested caps layered over ``base``, or None if no flag was given."""
    if args.max_seconds is None and args.max_actions is None:
        return None
    return Budget(max_seconds=base.max_seconds if args.max_seconds is None else args.max_seconds,
                  max_actions=base.max_actions if args.max_actions is None else args.max_actions)


def _existing(store: TaskStore, task_id: str) -> dict[str, Any]:
    try:
        return store.get(task_id)
    except KeyError:
        raise CliError(f"unknown task id: {task_id}", EXIT_USAGE) from None


def _create(store: TaskStore, args: argparse.Namespace) -> dict[str, Any]:
    try:
        if args.command == "start-fixture":
            budget = _budget(get_workflow("desktop_save_fixture").budget, args)
            return store.create_fixture(window_title=args.window_title, output_path=args.output,
                                        text=args.text, budget=budget)
        budget = _budget(get_workflow("calculator_1_plus_2").budget, args)
        return store.create_calculator(budget=budget)
    except ValueError as exc:
        raise CliError(str(exc), EXIT_USAGE) from exc


def _startup_failure(store: TaskStore, task: dict[str, Any], exc: Exception) -> CliError:
    """Record that the backend never started, so the task is not left looking untouched."""
    reason = f"Desktop backend failed to start: {str(exc) or type(exc).__name__}"
    state = ""
    if task["status"] == "pending":
        try:
            store.record(task, "startup_failed", status="needs_review", error=reason)
            state = ", marked needs_review"
        except StateConflict:
            pass  # cancelled or changed elsewhere; that state wins
    return CliError(f"{reason} (task {task['id']}{state})", EXIT_FAILED)


async def _close(desktop: Any) -> None:
    try:
        await desktop.shutdown()
    except Exception as exc:
        # The run's outcome is already recorded; a stuck shutdown must not hide it.
        logger.warning("desktop shutdown failed error=%s", exc)


async def _execute(store: TaskStore, config: JarvisConfig, task: dict[str, Any],
                   pause_after: int | None) -> dict:
    try:
        minimums = get_workflow(task["workflow"]).desktop_minimums
    except ValueError as exc:
        raise CliError(f"task {task['id']}: {exc}", EXIT_USAGE) from exc
    desktop_config = config.desktop
    limits = {key: max(getattr(desktop_config, key), minimum) for key, minimum in minimums.items()}
    if limits:
        desktop_config = desktop_config.model_copy(update=limits)
    desktop = CuaDesktop(desktop_config)
    try:
        try:
            await desktop.startup()
        except Exception as exc:
            raise _startup_failure(store, task, exc) from exc
        runner = TaskRunner(store, desktop, PermissionGateway(config.security))
        return _public(await runner.run(task["id"], pause_after=pause_after))
    finally:
        await _close(desktop)


async def _run(args: argparse.Namespace) -> dict:
    try:
        config = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise CliError(f"cannot load configuration: {exc}", EXIT_USAGE) from exc
    setup_logging(config.system.log_level)
    store = TaskStore(args.db or expand_path(config.system.data_dir) / "tasks.sqlite3")
    if args.command == "list":
        return {"tasks": [_public(task) for task in store.list()]}
    if args.command == "status":
        task = _existing(store, args.task_id)
        return _public(task, store.events(task["id"]))
    if args.command == "cancel":
        _existing(store, args.task_id)
        return _public(store.cancel(args.task_id))
    if args.command == "resume":
        task = _existing(store, args.task_id)
        budget = _budget(Budget.from_state(task["budget"]), args)
        if budget is not None and task["status"] not in _TERMINAL:
            task = store.set_budget(task["id"], budget)
    else:
        task = _create(store, args)
        # The full id, because status/resume/cancel need it; the row exists from here on.
        logger.info("task created id=%s workflow=%s", task["id"], task["workflow"])
    if task["status"] in _TERMINAL:
        return _public(task)
    return await _execute(store, config, task, args.pause_after)


def main(argv: list[str] | None = None) -> int:
    """Run one command; returns the process exit code."""
    configure_stdio()  # stdout is piped in scripts, where Windows would pick a legacy codec
    args = parse_args(argv)
    try:
        result = asyncio.run(_run(args))
    except CliError as exc:
        print(f"jarvis.tasks: {exc}", file=sys.stderr)
        return exc.code
    except StateConflict as exc:
        print(f"jarvis.tasks: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command in _RUN_COMMANDS and result.get("status") == "needs_review":
        return EXIT_NEEDS_REVIEW
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
