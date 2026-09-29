"""Local, versioned task records kept separate from conversational memory."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STEP_NAMES = ("discover_window", "enter_text", "save_file", "verify_file")
CALCULATOR_STEPS = ("discover_calculator", "clear", "enter_one", "plus", "enter_two", "equals")
WORKFLOW_STEPS = {"desktop_save_fixture": STEP_NAMES, "calculator_1_plus_2": CALCULATOR_STEPS}
_VALID_STATUS = {"pending", "running", "paused", "needs_review", "completed", "cancelled"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StateConflict(RuntimeError):
    """Another process changed the task; the caller must reload it."""


class TaskStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            row = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                db.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            elif int(row[0]) != SCHEMA_VERSION:
                raise RuntimeError(f"Unsupported task database schema {row[0]}")
            db.execute("""CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, workflow TEXT NOT NULL, goal TEXT NOT NULL,
                state_json TEXT NOT NULL, version INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                kind TEXT NOT NULL, details_json TEXT NOT NULL, at TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(id))""")

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

    def create_fixture(self, *, window_title: str, output_path: Path, text: str) -> dict[str, Any]:
        if not window_title.startswith("Jarvis Cua Smoke ") or len(window_title) < 20:
            raise ValueError("This first workflow accepts only a uniquely named Jarvis Cua Smoke fixture")
        target = Path(output_path).resolve()
        if not target.is_absolute() or not target.parent.is_dir():
            raise ValueError("The output directory must already exist")
        if not text or len(text) > 2000:
            raise ValueError("Fixture text must contain 1–2000 characters")
        task_id = uuid.uuid4().hex
        timestamp = _now()
        state = {"status": "pending", "current_step": 0,
                 "step_status": ["pending"] * len(STEP_NAMES),
                 "scope": {"window_title": window_title, "output_path": str(target)},
                 "input": {"text": text}, "attempt": None, "evidence": [],
                 "error": None, "result": None}
        with self._connect() as db:
            db.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (task_id, "desktop_save_fixture", "Save text in isolated desktop fixture",
                        json.dumps(state), 0, timestamp, timestamp))
            db.execute("INSERT INTO events(task_id, kind, details_json, at) VALUES (?, ?, ?, ?)",
                       (task_id, "created", "{}", timestamp))
        return self.get(task_id)

    def create_calculator(self) -> dict[str, Any]:
        """A narrowly scoped, repeatable desktop acceptance scenario."""
        task_id = uuid.uuid4().hex
        timestamp = _now()
        state = {"status": "pending", "current_step": 0,
                 "step_status": ["pending"] * len(CALCULATOR_STEPS),
                 "scope": {"app": "Calculator", "window_title": "Calculator"},
                 "input": {"expression": "1+2", "expected": "3"},
                 "attempt": None, "evidence": [], "error": None, "result": None}
        with self._connect() as db:
            db.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (task_id, "calculator_1_plus_2", "Calculate 1+2 in Windows Calculator",
                        json.dumps(state), 0, timestamp, timestamp))
            db.execute("INSERT INTO events(task_id, kind, details_json, at) VALUES (?, ?, ?, ?)",
                       (task_id, "created", "{}", timestamp))
        return self.get(task_id)

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
        state = {key: task[key] for key in
                 ("status", "current_step", "step_status", "scope", "input", "attempt",
                  "evidence", "error", "result")}
        state.update(changes)
        steps = WORKFLOW_STEPS[task["workflow"]]
        if (state["status"] not in _VALID_STATUS
                or len(state["step_status"]) != len(steps)
                or not 0 <= state["current_step"] <= len(steps)):
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

    def cancel(self, task_id: str) -> dict[str, Any]:
        task = self.get(task_id)
        if task["status"] in {"completed", "cancelled"}:
            return task
        return self.record(task, "cancelled", status="cancelled")
