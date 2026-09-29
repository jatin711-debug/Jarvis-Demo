"""Local, versioned task records kept separate from conversational memory."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis.tasks.contracts import DEFAULT_BUDGET, Budget, TaskSpec
from jarvis.tasks.workflows import (
    WORKFLOW_STEPS,
    WORKFLOWS,
    calculator_spec,
    fixture_spec,
    get_workflow,
)
from jarvis.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["SCHEMA_VERSION", "WORKFLOW_STEPS", "StateConflict", "TaskStore"]

SCHEMA_VERSION = 2
STATE_KEYS = ("status", "current_step", "step_status", "scope", "input", "attempt",
              "evidence", "error", "result", "budget", "started_at", "elapsed_seconds",
              "actions_taken")
_VALID_STATUS = {"pending", "running", "paused", "needs_review", "completed", "cancelled"}
_TERMINAL = ("completed", "cancelled")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StateConflict(RuntimeError):
    """Another process changed the task; the caller must reload it."""


def _migrate_v1_to_v2(db: sqlite3.Connection) -> int:
    """Add budget and progress fields to v1 task states, keeping everything else."""
    rows = db.execute("SELECT id, workflow, state_json, created_at FROM tasks").fetchall()
    for row in rows:
        state = json.loads(row["state_json"])
        workflow = WORKFLOWS.get(row["workflow"])
        journaled = db.execute(
            "SELECT COUNT(*) FROM events WHERE task_id=? AND kind='action_intent'",
            (row["id"],)).fetchone()[0]
        state.setdefault("budget", (workflow.budget if workflow else DEFAULT_BUDGET).as_state())
        state.setdefault("started_at",
                         None if state.get("status") == "pending" else row["created_at"])
        state.setdefault("elapsed_seconds", 0.0)
        state.setdefault("actions_taken", journaled)
        db.execute("UPDATE tasks SET state_json=? WHERE id=?", (json.dumps(state), row["id"]))
        db.execute("INSERT INTO events(task_id, kind, details_json, at) VALUES (?, ?, ?, ?)",
                   (row["id"], "schema_migrated", json.dumps({"from": 1, "to": 2}), _now()))
    return len(rows)


_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], int]] = {1: _migrate_v1_to_v2}


class TaskStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            if self._schema_version(db) != SCHEMA_VERSION:
                self._upgrade(db)

    @staticmethod
    def _schema_version(db: sqlite3.Connection) -> int | None:
        row = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            return None
        try:
            version = int(row[0])
        except ValueError:
            raise RuntimeError(f"Unsupported task database schema {row[0]!r}") from None
        if version < 1 or version > SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported task database schema {version}; "
                               f"this version understands up to {SCHEMA_VERSION}")
        return version

    def _upgrade(self, db: sqlite3.Connection) -> None:
        db.execute("BEGIN IMMEDIATE")
        version = self._schema_version(db)
        db.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, workflow TEXT NOT NULL, goal TEXT NOT NULL,
            state_json TEXT NOT NULL, version INTEGER NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
            kind TEXT NOT NULL, details_json TEXT NOT NULL, at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES tasks(id))""")
        if version is None:
            db.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            return
        for step in range(version, SCHEMA_VERSION):
            rows = _MIGRATIONS[step](db)
            logger.info("task schema migrated from=%d to=%d rows=%d", step, step + 1, rows)
        db.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),))

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _unpack(row: sqlite3.Row) -> dict[str, Any]:
        return {"id": row["id"], "workflow": row["workflow"], "goal": row["goal"],
                "version": row["version"], "created_at": row["created_at"],
                "updated_at": row["updated_at"], **json.loads(row["state_json"])}

    def create(self, workflow: str, spec: TaskSpec, *, budget: Budget | None = None
               ) -> dict[str, Any]:
        """Persist a new pending task for a registered workflow."""
        definition = get_workflow(workflow)
        if budget is not None and not isinstance(budget, Budget):
            raise ValueError("budget must be a Budget")
        task_id = uuid.uuid4().hex
        timestamp = _now()
        state = {"status": "pending", "current_step": 0,
                 "step_status": ["pending"] * len(definition.steps),
                 "scope": dict(spec.scope), "input": dict(spec.input), "attempt": None,
                 "evidence": [], "error": None, "result": None,
                 "budget": (budget or definition.budget).as_state(), "started_at": None,
                 "elapsed_seconds": 0.0, "actions_taken": 0}
        with self._connect() as db:
            db.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (task_id, workflow, spec.goal, json.dumps(state), 0, timestamp, timestamp))
            db.execute("INSERT INTO events(task_id, kind, details_json, at) VALUES (?, ?, ?, ?)",
                       (task_id, "created", "{}", timestamp))
        return self.get(task_id)

    def create_fixture(self, *, window_title: str, output_path: Path, text: str,
                       budget: Budget | None = None) -> dict[str, Any]:
        spec = fixture_spec(window_title=window_title, output_path=output_path, text=text)
        return self.create("desktop_save_fixture", spec, budget=budget)

    def create_calculator(self, *, budget: Budget | None = None) -> dict[str, Any]:
        """A narrowly scoped, repeatable desktop acceptance scenario."""
        return self.create("calculator_1_plus_2", calculator_spec(), budget=budget)

    def get(self, task_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown task: {task_id}")
        return self._unpack(row)

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._unpack(row) for row in rows]

    def events(self, task_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT kind, details_json, at FROM events WHERE task_id=? ORDER BY id",
                              (task_id,)).fetchall()
        return [{"kind": row["kind"], "details": json.loads(row["details_json"]),
                 "at": row["at"]} for row in rows]

    def record(self, task: dict[str, Any], kind: str, *, details: dict[str, Any] | None = None,
               **changes: Any) -> dict[str, Any]:
        unknown = set(changes) - set(STATE_KEYS)
        if unknown:
            raise ValueError(f"Unknown task state fields: {sorted(unknown)}")
        state = {key: task[key] for key in STATE_KEYS}
        state.update(changes)
        workflow = WORKFLOWS.get(task["workflow"])
        if state["status"] not in _VALID_STATUS or (workflow is not None and (
                len(state["step_status"]) != len(workflow.steps)
                or not 0 <= state["current_step"] <= len(workflow.steps))):
            raise ValueError("Invalid task state")
        timestamp = _now()
        with self._connect() as db:
            cursor = db.execute("""UPDATE tasks SET state_json=?, version=version+1, updated_at=?
                WHERE id=? AND version=?
                AND json_extract(state_json, '$.status') NOT IN ('completed', 'cancelled')""",
                (json.dumps(state), timestamp, task["id"], task["version"]))
            if cursor.rowcount != 1:
                raise StateConflict("Task changed or reached a terminal state; reload it")
            db.execute("INSERT INTO events(task_id, kind, details_json, at) VALUES (?, ?, ?, ?)",
                       (task["id"], kind, json.dumps(details or {}), timestamp))
        return self.get(task["id"])

    def set_budget(self, task_id: str, budget: Budget) -> dict[str, Any]:
        """Replace a live task's caps, for example to let an exhausted task resume."""
        if not isinstance(budget, Budget):
            raise ValueError("budget must be a Budget")
        return self.record(self.get(task_id), "budget_changed", budget=budget.as_state(),
                           details=budget.as_state())

    def cancel(self, task_id: str) -> dict[str, Any]:
        task = self.get(task_id)
        if task["status"] in _TERMINAL:
            return task
        return self.record(task, "cancelled", status="cancelled")
