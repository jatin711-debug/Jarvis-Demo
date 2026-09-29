"""Text CLI for inspecting and running durable tasks without audio or LLMs."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from jarvis.agent.permissions import PermissionGateway
from jarvis.desktop.cua import CuaDesktop
from jarvis.tasks.runner import TaskRunner
from jarvis.tasks.store import WORKFLOW_STEPS, TaskStore
from jarvis.utils.config import expand_path, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jarvis durable task CLI")
    parser.add_argument("--config", default=None)
    parser.add_argument("--db", type=Path, default=None, help="Task database path")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start-fixture", help="Save text in an existing Jarvis smoke fixture")
    start.add_argument("--window-title", required=True)
    start.add_argument("--output", type=Path, required=True)
    start.add_argument("--text", required=True)
    start.add_argument("--pause-after", type=int, default=None, help="Pause after N verified steps")
    calculator = commands.add_parser("start-calculator", help="Launch Calculator and verify 1+2=3")
    calculator.add_argument("--pause-after", type=int, default=None)
    status = commands.add_parser("status")
    status.add_argument("task_id")
    resume = commands.add_parser("resume")
    resume.add_argument("task_id")
    resume.add_argument("--pause-after", type=int, default=None)
    cancel = commands.add_parser("cancel")
    cancel.add_argument("task_id")
    commands.add_parser("list")
    return parser.parse_args()


def _public(task: dict, events: list[dict] | None = None) -> dict:
    result = {key: task[key] for key in ("id", "goal", "workflow", "status", "version", "current_step",
                                          "scope", "step_status", "error", "result")}
    steps = WORKFLOW_STEPS[task["workflow"]]
    result["next_step"] = (steps[task["current_step"]]
                           if task["current_step"] < len(steps) else None)
    result["evidence"] = task["evidence"]
    if events is not None:
        result["events"] = events
    return result


async def _run(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    store = TaskStore(args.db or expand_path(config.system.data_dir) / "tasks.sqlite3")
    if args.command == "list":
        return {"tasks": [_public(task) for task in store.list()]}
    if args.command == "status":
        return _public(store.get(args.task_id), store.events(args.task_id))
    if args.command == "cancel":
        return _public(store.cancel(args.task_id))
    if args.command == "start-fixture":
        task = store.create_fixture(window_title=args.window_title, output_path=args.output,
                                    text=args.text)
        task_id = task["id"]
    elif args.command == "start-calculator":
        task_id = store.create_calculator()["id"]
    else:
        task_id = args.task_id
    if store.get(task_id)["status"] in {"completed", "cancelled"}:
        return _public(store.get(task_id))
    task = store.get(task_id)
    desktop_config = config.desktop
    if task["workflow"] == "calculator_1_plus_2":
        # Calculator's lower keypad can be omitted by the compact default tree.
        desktop_config = desktop_config.model_copy(update={
            "max_elements": max(desktop_config.max_elements, 80),
            "max_depth": max(desktop_config.max_depth, 10),
        })
    desktop = CuaDesktop(desktop_config)
    try:
        await desktop.startup()
        return _public(await TaskRunner(store, desktop, PermissionGateway(config.security)).run(
            task_id, pause_after=args.pause_after))
    finally:
        await desktop.shutdown()


def main() -> None:
    args = parse_args()
    result = asyncio.run(_run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command in {"start-fixture", "start-calculator", "resume"} and result.get("status") == "needs_review":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
