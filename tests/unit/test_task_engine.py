"""Generic task engine: contracts, budgets, timeouts, migration, and a third workflow."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from jarvis.tasks.contracts import (
    Budget,
    Check,
    RetryPolicy,
    StepContract,
    TaskSpec,
    Workflow,
)
from jarvis.tasks.runner import TaskRunner
from jarvis.tasks.store import SCHEMA_VERSION, WORKFLOW_STEPS, StateConflict, TaskStore
from jarvis.tasks.workflows import WORKFLOWS, get_workflow, register_workflow

TITLE = "Jarvis Cua Smoke engine1234"


# --- fakes ----------------------------------------------------------------

class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FixtureDesktop:
    """The smoke fixture; each dispatch can cost fake time."""

    def __init__(self, path: Path, clock: FakeClock | None = None, act_cost: float = 0.0):
        self.path = path
        self.clock = clock
        self.act_cost = act_cost
        self.value = ""
        self.generation = 0
        self.calls: list[str] = []

    async def windows(self):
        self.calls.append("windows")
        return {"windows": [{"pid": 101, "window_id": 202, "title": TITLE}], "truncated": False}

    async def observe(self, pid, window_id):
        self.calls.append("observe")
        self.generation += 1
        return {"pid": pid, "window_id": window_id, "snapshot_id": f"s{self.generation}",
                "elements": [
                    {"element_index": 0, "label": "Smoke text", "enabled": True,
                     "value": self.value},
                    {"element_index": 1, "label": "Save smoke file", "enabled": True}],
                "degraded": False, "truncated": False, "elements_complete": False}

    async def act(self, action, pid, window_id, snapshot_id, element_index, **kwargs):
        self.calls.append(action)
        if self.clock is not None:
            self.clock.advance(self.act_cost)
        if action == "type_text":
            self.value = kwargs["text"]
        else:
            self.path.write_text(self.value, encoding="utf-8")
        return {"after": await self.observe(pid, window_id)}


class Counter:
    """A backend for the in-test workflow: read a counter, bump it once."""

    def __init__(self) -> None:
        self.value = 0
        self.bumps = 0
        self.reads = 0
        self.read_delays: list[float] = []
        self.bump_mode: str | None = None

    async def read(self) -> int:
        self.reads += 1
        if self.read_delays:
            await asyncio.sleep(self.read_delays.pop(0))
        return self.value

    async def bump(self) -> None:
        self.bumps += 1
        if self.bump_mode == "hang_before":
            await asyncio.sleep(3600)
        self.value += 1
        if self.bump_mode == "hang_after":
            await asyncio.sleep(3600)


def counter_workflow(name: str = "counter_to_one", *, timeout: float = 1.0, reobserve: int = 0,
                     confirmed: bool = False) -> Workflow:
    def read(ctx):
        return ctx.desktop.read()

    def evidence(ctx, value, phase):
        return {"counter": value}

    return Workflow(
        name=name,
        steps=(
            StepContract(
                name="check_counter", target="the counter", operation="read", observe=read,
                postcondition=lambda ctx, value: Check(value in (0, 1), f"counter is {value}"),
                evidence=evidence, timeout=timeout, retry=RetryPolicy(reobserve=reobserve)),
            StepContract(
                name="bump", target="the counter", operation="mutate", observe=read,
                postcondition=lambda ctx, value: Check(value == 1, f"counter is {value}"),
                precondition=lambda ctx, value: Check(value == 0, f"counter is {value}"),
                action=lambda ctx, handle: ctx.desktop.bump(), evidence=evidence,
                timeout=timeout, uncertain="Bump outcome uncertain",
                confirmed_by="verify_counter" if confirmed else None),
            StepContract(
                name="verify_counter", target="the counter", operation="read", observe=read,
                postcondition=lambda ctx, value: Check(value == 1, f"counter is {value}"),
                evidence=evidence, timeout=timeout),
        ),
        result="Counter reads 1",
        budget=Budget(max_seconds=60, max_actions=3),
    )


def counter_spec(note: str = "n") -> TaskSpec:
    return TaskSpec(goal="Bump the counter once", scope={}, input={"note": note})


@pytest.fixture
def register():
    added: list[str] = []

    def add(workflow: Workflow) -> Workflow:
        register_workflow(workflow)
        added.append(workflow.name)
        return workflow

    yield add
    for name in added:
        WORKFLOWS.pop(name, None)


@pytest.fixture
def fixture_task(tmp_path):
    output = tmp_path / "saved.txt"
    store = TaskStore(tmp_path / "tasks.sqlite3")

    def create(**kwargs):
        return store.create_fixture(window_title=TITLE, output_path=output,
                                    text="exact contents", **kwargs)

    return store, create, output


# --- contract validation ----------------------------------------------------

def read_step(**overrides) -> StepContract:
    values = {
        "name": "look", "target": "the thing", "operation": "read",
        "observe": lambda ctx: 1, "postcondition": lambda ctx, seen: Check(True),
        "evidence": lambda ctx, seen, phase: {},
    }
    values.update(overrides)
    return StepContract(**values)


def mutate_step(**overrides) -> StepContract:
    values = {"name": "change", "operation": "mutate", "action": lambda ctx, handle: None}
    values.update(overrides)
    return read_step(**values)


@pytest.mark.parametrize("build, message", [
    (lambda: read_step(name="Bad Name"), "lower_snake_case"),
    (lambda: read_step(name=""), "lower_snake_case"),
    (lambda: read_step(target="  "), "target"),
    (lambda: read_step(operation="delete"), "operation"),
    (lambda: read_step(timeout=0), "timeout"),
    (lambda: read_step(timeout=-3), "timeout"),
    (lambda: read_step(timeout=float("nan")), "timeout"),
    (lambda: read_step(postcondition=None), "postcondition must be callable"),
    (lambda: read_step(evidence="proof"), "evidence must be callable"),
    (lambda: read_step(on_unmet="ignore"), "on_unmet"),
    (lambda: read_step(retry="always"), "RetryPolicy"),
    (lambda: read_step(action=lambda ctx, handle: None), "cannot declare action"),
    (lambda: read_step(precondition=lambda ctx, seen: Check(True)), "cannot declare precondition"),
    (lambda: read_step(confirmed_by="later"), "only a mutating step"),
    (lambda: mutate_step(action=None), "must declare an action"),
    (lambda: mutate_step(on_unmet="pause"), "cannot pause"),
    (lambda: mutate_step(retry=RetryPolicy(reobserve=1)), "never retried automatically"),
    (lambda: mutate_step(uncertain=" "), "uncertain-outcome message"),
    (lambda: RetryPolicy(reobserve=-1), "reobserve"),
    (lambda: RetryPolicy(reobserve=True), "reobserve"),
    (lambda: Budget(max_seconds=0), "max_seconds"),
    (lambda: Budget(max_seconds=float("inf")), "max_seconds"),
    (lambda: Budget(max_actions=-1), "max_actions"),
    (lambda: Budget(max_actions=True), "max_actions"),
    (lambda: Budget(max_actions=10**6), "max_actions"),
])
def test_bad_step_and_budget_definitions_are_rejected(build, message):
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.parametrize("overrides, message", [
    ({"name": "Not Snake"}, "lower_snake_case"),
    ({"steps": ()}, "at least one step"),
    ({"steps": (read_step(), read_step())}, "unique"),
    ({"steps": ("look",)}, "StepContract"),
    ({"result": ""}, "result"),
    ({"budget": {"max_seconds": 5}}, "Budget"),
    ({"steps": (mutate_step(), mutate_step(name="change_again")),
      "budget": Budget(max_seconds=5, max_actions=1)}, "2 steps mutate"),
    ({"steps": (mutate_step(confirmed_by="missing"),)}, "confirmed by a later step"),
    ({"steps": (mutate_step(name="first", confirmed_by="first"),)}, "confirmed by a later step"),
    ({"steps": (mutate_step(name="early"), mutate_step(name="late", confirmed_by="early"))},
     "confirmed by a later step"),
    ({"permissions": "read_file"}, "permissions"),
    ({"desktop_minimums": {"max_windows": 3}}, "desktop_minimums"),
    ({"desktop_minimums": {"max_depth": 0}}, "desktop_minimums"),
])
def test_bad_workflow_definitions_are_rejected(overrides, message):
    values = {"name": "tiny", "steps": (read_step(),), "result": "done"}
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        Workflow(**values)


def test_registration_rejects_duplicates_and_non_workflows(register):
    workflow = register(counter_workflow("registered_once"))
    with pytest.raises(ValueError, match="already registered"):
        register_workflow(counter_workflow("registered_once"))
    with pytest.raises(ValueError, match="already registered"):
        register_workflow(get_workflow("desktop_save_fixture"))
    assert register_workflow(counter_workflow("registered_once"), replace=True) is not workflow
    with pytest.raises(TypeError):
        register_workflow("not a workflow")  # type: ignore[arg-type]


def test_builtin_workflows_are_contracts():
    assert WORKFLOW_STEPS["desktop_save_fixture"] == (
        "discover_window", "enter_text", "save_file", "verify_file")
    assert WORKFLOW_STEPS["calculator_1_plus_2"] == (
        "discover_calculator", "clear", "enter_one", "plus", "enter_two", "equals")
    assert "nope" not in WORKFLOW_STEPS
    for name in ("desktop_save_fixture", "calculator_1_plus_2"):
        workflow = get_workflow(name)
        for step in workflow.steps:
            assert step.target and step.timeout > 0
            if step.operation == "mutate":
                assert step.action is not None and step.retry.reobserve == 0


def test_cli_budget_flags_layer_over_a_base_budget():
    import argparse

    from jarvis.tasks.__main__ import _budget

    base = Budget(max_seconds=100, max_actions=5)
    assert _budget(base, argparse.Namespace(max_seconds=None, max_actions=None)) is None
    assert _budget(base, argparse.Namespace(max_seconds=30, max_actions=None)) == Budget(30, 5)
    assert _budget(base, argparse.Namespace(max_seconds=None, max_actions=9)) == Budget(100, 9)
    with pytest.raises(ValueError):
        _budget(base, argparse.Namespace(max_seconds=-1, max_actions=None))


# --- unknown workflow -------------------------------------------------------

def test_create_rejects_unknown_workflow(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    with pytest.raises(ValueError, match="Unsupported workflow"):
        store.create("no_such_workflow", counter_spec())


async def test_runner_rejects_task_whose_workflow_is_gone(tmp_path, register):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    workflow = register(counter_workflow("short_lived"))
    task = store.create(workflow.name, counter_spec())
    WORKFLOWS.pop(workflow.name)
    with pytest.raises(ValueError, match="Unsupported workflow"):
        await TaskRunner(store, Counter()).run(task["id"])
    assert store.cancel(task["id"])["status"] == "cancelled"


# --- budgets ----------------------------------------------------------------

def test_default_budgets_are_stored_in_task_state(fixture_task):
    store, create, _ = fixture_task
    task = create()
    assert task["budget"] == {"max_seconds": 180.0, "max_actions": 4}
    assert (task["started_at"], task["elapsed_seconds"], task["actions_taken"]) == (None, 0.0, 0)
    assert store.create_calculator()["budget"] == {"max_seconds": 300.0, "max_actions": 8}
    custom = create(budget=Budget(max_seconds=45, max_actions=3))
    assert custom["budget"] == {"max_seconds": 45.0, "max_actions": 3}
    assert store.get(custom["id"])["budget"] == custom["budget"]


async def test_time_budget_stops_between_steps_never_mid_action(fixture_task):
    store, create, output = fixture_task
    clock = FakeClock()
    task = create(budget=Budget(max_seconds=10, max_actions=4))
    desktop = FixtureDesktop(output, clock, act_cost=20)
    result = await TaskRunner(store, desktop, clock=clock).run(task["id"])
    assert result["status"] == "needs_review"
    assert result["error"].startswith("Time budget exhausted")
    assert result["step_status"] == ["verified", "verified", "pending", "pending"]
    assert result["attempt"] is None
    assert result["elapsed_seconds"] == 20
    assert desktop.calls.count("type_text") == 1
    assert "click" not in desktop.calls
    assert "budget_exhausted" in [event["kind"] for event in store.events(task["id"])]

    store.set_budget(task["id"], Budget(max_seconds=100, max_actions=4))
    final = await TaskRunner(TaskStore(store.path), desktop, clock=clock).run(task["id"])
    assert final["status"] == "completed"
    assert final["elapsed_seconds"] == 40
    assert desktop.calls.count("type_text") == 1
    assert desktop.calls.count("click") == 1


async def test_time_budget_counts_run_time_not_time_between_runs(fixture_task):
    store, create, output = fixture_task
    clock = FakeClock()
    task = create(budget=Budget(max_seconds=30, max_actions=4))
    desktop = FixtureDesktop(output, clock, act_cost=5)
    paused = await TaskRunner(store, desktop, clock=clock).run(task["id"], pause_after=2)
    assert paused["status"] == "paused"
    clock.advance(86400)
    final = await TaskRunner(store, desktop, clock=clock).run(task["id"])
    assert final["status"] == "completed"
    assert final["elapsed_seconds"] == 10


async def test_action_budget_stops_before_dispatch(fixture_task):
    store, create, output = fixture_task
    task = create(budget=Budget(max_seconds=60, max_actions=1))
    desktop = FixtureDesktop(output)
    result = await TaskRunner(store, desktop).run(task["id"])
    assert result["status"] == "needs_review"
    assert result["error"].startswith("Action budget exhausted")
    assert result["actions_taken"] == 1
    assert result["attempt"] is None
    assert result["current_step"] == 2
    assert "click" not in desktop.calls
    again = await TaskRunner(store, desktop).run(task["id"])
    assert again["status"] == "needs_review"
    assert "click" not in desktop.calls

    store.set_budget(task["id"], Budget(max_seconds=60, max_actions=2))
    final = await TaskRunner(store, desktop).run(task["id"])
    assert final["status"] == "completed"
    assert final["actions_taken"] == 2
    assert final["started_at"]


def test_set_budget_rejects_terminal_tasks(fixture_task):
    store, create, _ = fixture_task
    task = create()
    store.cancel(task["id"])
    with pytest.raises(StateConflict):
        store.set_budget(task["id"], Budget())
    with pytest.raises(ValueError):
        store.set_budget(task["id"], {"max_seconds": 5})  # type: ignore[arg-type]


# --- timeouts ---------------------------------------------------------------

async def test_timed_out_mutating_step_is_uncertain_and_never_replayed(tmp_path, register):
    workflow = register(counter_workflow(timeout=0.05))
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create(workflow.name, counter_spec())
    desktop = Counter()
    desktop.bump_mode = "hang_before"
    first = await TaskRunner(store, desktop).run(task["id"])
    assert first["status"] == "needs_review"
    assert "timed out" in first["error"]
    assert first["attempt"] == {"step": "bump"}
    assert desktop.bumps == 1
    assert "step_timeout" in [event["kind"] for event in store.events(task["id"])]

    desktop.bump_mode = None
    again = await TaskRunner(TaskStore(store.path), desktop).run(task["id"])
    assert again["status"] == "needs_review"
    assert desktop.bumps == 1
    assert desktop.value == 0


async def test_timed_out_mutation_that_landed_reconciles_without_replay(tmp_path, register):
    workflow = register(counter_workflow(timeout=0.05))
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create(workflow.name, counter_spec())
    desktop = Counter()
    desktop.bump_mode = "hang_after"
    first = await TaskRunner(store, desktop).run(task["id"])
    assert first["status"] == "needs_review"
    assert desktop.value == 1
    desktop.bump_mode = None
    final = await TaskRunner(TaskStore(store.path), desktop).run(task["id"])
    assert final["status"] == "completed"
    assert desktop.bumps == 1
    assert {"step": "bump", "counter": 1, "reconciled": True} in final["evidence"]


async def test_timed_out_read_step_reobserves_then_pauses(tmp_path, register):
    workflow = register(counter_workflow(timeout=0.05, reobserve=1))
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create(workflow.name, counter_spec())
    desktop = Counter()
    desktop.read_delays = [5, 5]
    paused = await TaskRunner(store, desktop).run(task["id"])
    assert paused["status"] == "paused"
    assert "timed out" in paused["error"]
    assert desktop.reads == 2
    assert desktop.bumps == 0
    final = await TaskRunner(store, desktop).run(task["id"])
    assert final["status"] == "completed"
    assert desktop.bumps == 1


async def test_read_step_that_recovers_on_reobserve_completes_in_one_run(tmp_path, register):
    workflow = register(counter_workflow(timeout=0.05, reobserve=1))
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create(workflow.name, counter_spec())
    desktop = Counter()
    desktop.read_delays = [5]
    final = await TaskRunner(store, desktop).run(task["id"])
    assert final["status"] == "completed"
    assert desktop.bumps == 1


async def test_cancelled_run_is_marked_interrupted_and_not_replayed(tmp_path, register):
    workflow = register(counter_workflow(timeout=30))
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create(workflow.name, counter_spec())
    desktop = Counter()
    desktop.bump_mode = "hang_before"
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(TaskRunner(store, desktop).run(task["id"]), 0.2)
    current = store.get(task["id"])
    assert current["status"] == "needs_review"
    assert current["attempt"] == {"step": "bump"}
    assert "interrupted" in [event["kind"] for event in store.events(task["id"])]
    desktop.bump_mode = None
    await TaskRunner(store, desktop).run(task["id"])
    assert desktop.bumps == 1


async def test_journaled_attempt_with_unmet_postcondition_is_not_redispatched(tmp_path, register):
    workflow = register(counter_workflow())
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create(workflow.name, counter_spec())
    store.record(task, "action_intent", status="running", attempt={"step": "bump"},
                 current_step=1, step_status=["verified", "pending", "pending"], actions_taken=1)
    desktop = Counter()
    result = await TaskRunner(store, desktop).run(task["id"])
    assert result["status"] == "needs_review"
    assert result["error"].startswith("Bump outcome uncertain")
    assert result["attempt"] == {"step": "bump"}
    assert desktop.bumps == 0


# --- a third workflow -------------------------------------------------------

async def test_third_workflow_runs_on_the_generic_engine(tmp_path, register):
    workflow = register(counter_workflow(confirmed=True))
    assert WORKFLOW_STEPS[workflow.name] == ("check_counter", "bump", "verify_counter")
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create(workflow.name, counter_spec())
    assert task["step_status"] == ["pending"] * 3
    assert task["budget"] == {"max_seconds": 60.0, "max_actions": 3}
    desktop = Counter()

    paused = await TaskRunner(store, desktop).run(task["id"], pause_after=2)
    assert paused["status"] == "paused"
    assert paused["step_status"] == ["verified", "acknowledged", "pending"]

    final = await TaskRunner(TaskStore(store.path), desktop).run(task["id"])
    assert final["status"] == "completed"
    assert final["step_status"] == ["verified"] * 3
    assert final["result"] == "Counter reads 1"
    assert final["actions_taken"] == 1
    assert final["attempt"] is None
    assert [item["step"] for item in final["evidence"]] == [
        "check_counter", "bump", "verify_counter"]
    assert desktop.bumps == 1

    events = store.events(task["id"])
    assert [event["kind"] for event in events if event["kind"] != "resumed"] == [
        "created", "started", "step_verified", "action_intent", "step_verified",
        "paused_at_boundary", "step_verified"]
    by_kind = {event["kind"]: event["details"] for event in events}
    assert by_kind["action_intent"]["phase"] == "act"
    assert by_kind["action_intent"]["target"] == "the counter"
    assert [event["details"]["phase"] for event in events if event["kind"] == "step_verified"] == [
        "inspect", "verify", "inspect"]

    assert (await TaskRunner(store, desktop).run(task["id"]))["status"] == "completed"
    assert desktop.bumps == 1


async def test_precondition_failure_stops_before_any_dispatch(tmp_path, register):
    workflow = register(counter_workflow())
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create(workflow.name, counter_spec())
    desktop = Counter()
    desktop.value = 5
    result = await TaskRunner(store, desktop).run(task["id"])
    assert result["status"] == "needs_review"
    assert desktop.bumps == 0
    assert result["attempt"] is None


async def test_step_callables_must_return_checks(tmp_path, register):
    step = mutate_step(name="bump", postcondition=lambda ctx, seen: True,
                       action=lambda ctx, handle: None)
    workflow = register(Workflow(name="sloppy", steps=(step,), result="done"))
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create(workflow.name, counter_spec())
    result = await TaskRunner(store, Counter()).run(task["id"])
    assert result["status"] == "needs_review"
    assert "must return a Check" in result["error"]
    assert result["attempt"] is None


# --- logging ----------------------------------------------------------------

async def test_engine_logs_follow_the_project_convention(fixture_task, caplog):
    store, create, output = fixture_task
    caplog.set_level(logging.DEBUG, logger="jarvis.tasks")
    task = store.create_fixture(window_title=TITLE, output_path=output,
                                text="correct-horse-battery")
    desktop = FixtureDesktop(output)
    await TaskRunner(store, desktop).run(task["id"])
    messages = [record.getMessage() for record in caplog.records
                if record.name.startswith("jarvis.tasks")]
    short = task["id"][:8]
    assert f"task started id={short} workflow=desktop_save_fixture steps=4 input_len=" in messages[0]
    assert f"task step verified id={short} step=discover_window n=1/4" in messages
    assert any(m.startswith(f"task completed id={short} ") for m in messages)
    text = "\n".join(messages)
    assert "correct-horse" not in text
    assert task["id"] not in text
    assert text.isascii()
    assert all(m.startswith("task ") for m in messages)


async def test_needs_review_is_logged_with_reason(fixture_task, caplog):
    store, create, output = fixture_task
    output.write_text("someone else's data", encoding="utf-8")
    caplog.set_level(logging.INFO, logger="jarvis.tasks")
    task = create()
    await TaskRunner(store, FixtureDesktop(output)).run(task["id"])
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("task needs_review")]
    assert len(lines) == 1
    assert re.fullmatch(
        rf"task needs_review id={task['id'][:8]} step=save_file phase=act reason=.*different contents",
        lines[0])


# --- schema migration -------------------------------------------------------

V1_TASK_ID = "a" * 32
V1_KEYS = {"status", "current_step", "step_status", "scope", "input", "attempt", "evidence",
           "error", "result"}


def build_v1_database(path: Path, output: Path, *, status="paused", attempt=None,
                      extra_events=()) -> None:
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    db.execute("""CREATE TABLE tasks (
        id TEXT PRIMARY KEY, workflow TEXT NOT NULL, goal TEXT NOT NULL,
        state_json TEXT NOT NULL, version INTEGER NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    db.execute("""CREATE TABLE events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
        kind TEXT NOT NULL, details_json TEXT NOT NULL, at TEXT NOT NULL,
        FOREIGN KEY(task_id) REFERENCES tasks(id))""")
    state = {
        "status": status, "current_step": 2,
        "step_status": ["verified", "verified", "pending", "pending"],
        "scope": {"window_title": TITLE, "output_path": str(output)},
        "input": {"text": "exact contents"}, "attempt": attempt,
        "evidence": [{"step": "discover_window", "window_title": TITLE},
                     {"step": "enter_text", "observed_value": "matched"}],
        "error": None, "result": None,
    }
    assert set(state) == V1_KEYS
    db.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
               (V1_TASK_ID, "desktop_save_fixture", "Save text in isolated desktop fixture",
                json.dumps(state), 7, "2026-01-01T00:00:00+00:00", "2026-01-01T00:05:00+00:00"))
    for kind in ("created", "started", "action_intent", "step_verified", *extra_events):
        db.execute("INSERT INTO events(task_id, kind, details_json, at) VALUES (?, ?, '{}', ?)",
                   (V1_TASK_ID, kind, "2026-01-01T00:01:00+00:00"))
    db.commit()
    db.close()


def stored_version(path: Path) -> str:
    db = sqlite3.connect(path)
    try:
        return db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    finally:
        db.close()


def test_new_database_is_created_at_current_version(tmp_path):
    TaskStore(tmp_path / "tasks.sqlite3")
    assert SCHEMA_VERSION == 2
    assert stored_version(tmp_path / "tasks.sqlite3") == str(SCHEMA_VERSION)


async def test_v1_database_migrates_and_task_resumes(tmp_path):
    path = tmp_path / "tasks.sqlite3"
    output = tmp_path / "saved.txt"
    build_v1_database(path, output)
    store = TaskStore(path)
    assert stored_version(path) == "2"

    task = store.get(V1_TASK_ID)
    assert task["status"] == "paused" and task["current_step"] == 2
    assert task["step_status"] == ["verified", "verified", "pending", "pending"]
    assert len(task["evidence"]) == 2
    assert task["version"] == 7
    assert task["budget"] == {"max_seconds": 180.0, "max_actions": 4}
    assert task["actions_taken"] == 1
    assert task["elapsed_seconds"] == 0.0
    assert task["started_at"] == "2026-01-01T00:00:00+00:00"
    assert [t["id"] for t in store.list()] == [V1_TASK_ID]
    kinds = [event["kind"] for event in store.events(V1_TASK_ID)]
    assert kinds == ["created", "started", "action_intent", "step_verified", "schema_migrated"]
    assert store.events(V1_TASK_ID)[-1]["details"] == {"from": 1, "to": 2}

    assert len(TaskStore(path).events(V1_TASK_ID)) == len(kinds)

    desktop = FixtureDesktop(output)
    desktop.value = "exact contents"
    final = await TaskRunner(TaskStore(path), desktop).run(V1_TASK_ID)
    assert final["status"] == "completed"
    assert final["step_status"] == ["verified"] * 4
    assert desktop.calls.count("type_text") == 0
    assert desktop.calls.count("click") == 1
    assert output.read_text(encoding="utf-8") == "exact contents"
    assert final["actions_taken"] == 2
    assert final["started_at"] == "2026-01-01T00:00:00+00:00"


async def test_v1_task_with_journaled_attempt_still_never_replays(tmp_path):
    path = tmp_path / "tasks.sqlite3"
    output = tmp_path / "saved.txt"
    build_v1_database(path, output, status="needs_review", attempt={"step": "save_file"})
    desktop = FixtureDesktop(output)
    desktop.value = "exact contents"
    result = await TaskRunner(TaskStore(path), desktop).run(V1_TASK_ID)
    assert result["status"] == "needs_review"
    assert "click" not in desktop.calls
    assert not output.exists()

    output.write_text("exact contents", encoding="utf-8")
    final = await TaskRunner(TaskStore(path), desktop).run(V1_TASK_ID)
    assert final["status"] == "completed"
    assert "click" not in desktop.calls


@pytest.mark.parametrize("version", ["3", "99", "abc", "0"])
def test_unknown_schema_versions_are_refused_and_left_alone(tmp_path, version):
    path = tmp_path / "tasks.sqlite3"
    build_v1_database(path, tmp_path / "saved.txt")
    db = sqlite3.connect(path)
    db.execute("UPDATE meta SET value=? WHERE key='schema_version'", (version,))
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="Unsupported task database schema"):
        TaskStore(path)
    assert stored_version(path) == version


def test_status_cli_reads_a_migrated_database(tmp_path):
    path = tmp_path / "tasks.sqlite3"
    build_v1_database(path, tmp_path / "saved.txt")
    process = subprocess.run(
        [sys.executable, "-m", "jarvis.tasks", "--db", str(path), "status", V1_TASK_ID],
        capture_output=True, text=True, check=True,
    )
    shown = json.loads(process.stdout)
    assert shown["next_step"] == "save_file"
    assert shown["budget"] == {"max_seconds": 180.0, "max_actions": 4}
    assert shown["actions_taken"] == 1
    assert shown["events"][-1]["kind"] == "schema_migrated"
