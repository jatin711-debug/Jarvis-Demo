"""Task checkpoints are tested through observable outcomes, not mock call order alone."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jarvis.tasks.runner import TaskRunner
from jarvis.tasks.store import StateConflict, TaskStore


class DesktopFixture:
    def __init__(self, path: Path, title: str):
        self.path = path
        self.title = title
        self.value = ""
        self.generation = 0
        self.calls = []
        self.fail_after_type = False
        self.fail_after_click = False
        self.fail_before_click = False

    async def windows(self):
        self.calls.append("windows")
        return {"windows": [{"pid": 101, "window_id": 202, "title": self.title}],
                "truncated": False}

    async def observe(self, pid, window_id):
        assert (pid, window_id) == (101, 202)
        self.calls.append("observe")
        self.generation += 1
        return {"pid": pid, "window_id": window_id, "snapshot_id": f"s{self.generation}",
                "elements": [
                    {"element_index": 0, "label": "Smoke text", "enabled": True, "value": self.value},
                    {"element_index": 1, "label": "Save smoke file", "enabled": True}],
                "degraded": False, "truncated": False, "elements_complete": False}

    async def act(self, action, pid, window_id, snapshot_id, element_index, **kwargs):
        self.calls.append(action)
        if action == "type_text":
            self.value = kwargs["text"]
            if self.fail_after_type:
                self.fail_after_type = False
                raise TimeoutError("response lost after typing")
        elif action == "click":
            if self.fail_before_click:
                self.fail_before_click = False
                raise TimeoutError("click outcome unknown")
            self.path.write_text(self.value, encoding="utf-8")
            if self.fail_after_click:
                self.fail_after_click = False
                raise TimeoutError("response lost after save")
        return {"after": await self.observe(pid, window_id)}


@pytest.fixture
def setup(tmp_path):
    title = "Jarvis Cua Smoke test1234"
    output = tmp_path / "saved.txt"
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_fixture(window_title=title, output_path=output, text="exact contents")
    desktop = DesktopFixture(output, title)
    return store, task, desktop, output


async def test_complete_workflow_uses_independent_file_evidence(setup):
    store, task, desktop, output = setup
    final = await TaskRunner(store, desktop).run(task["id"])
    assert final["status"] == "completed"
    assert final["step_status"] == ["verified"] * 4
    assert output.read_text(encoding="utf-8") == "exact contents"
    assert desktop.calls.count("type_text") == 1
    assert desktop.calls.count("click") == 1
    assert len(store.events(task["id"])) >= 6


async def test_pause_reopen_database_resume_without_retyping(setup):
    store, task, desktop, output = setup
    paused = await TaskRunner(store, desktop).run(task["id"], pause_after=2)
    assert paused["status"] == "paused"
    assert paused["current_step"] == 2
    reopened = TaskStore(store.path)
    result = await TaskRunner(reopened, desktop).run(task["id"])
    assert result["status"] == "completed"
    assert desktop.calls.count("type_text") == 1
    assert output.read_text(encoding="utf-8") == "exact contents"


async def test_uncertain_text_action_reconciles_before_retry(setup):
    store, task, desktop, _ = setup
    desktop.fail_after_type = True
    first = await TaskRunner(store, desktop).run(task["id"])
    assert first["status"] == "needs_review"
    assert first["attempt"] == {"step": "enter_text"}
    result = await TaskRunner(TaskStore(store.path), desktop).run(task["id"])
    assert result["status"] == "completed"
    assert desktop.calls.count("type_text") == 1


async def test_uncertain_save_checks_file_before_retry(setup):
    store, task, desktop, output = setup
    desktop.fail_after_click = True
    first = await TaskRunner(store, desktop).run(task["id"])
    assert first["status"] == "needs_review"
    assert first["attempt"] == {"step": "save_file"}
    assert output.read_text(encoding="utf-8") == "exact contents"
    result = await TaskRunner(TaskStore(store.path), desktop).run(task["id"])
    assert result["status"] == "completed"
    assert desktop.calls.count("click") == 1


async def test_uncertain_save_without_file_requires_review(setup):
    store, task, desktop, _ = setup
    desktop.fail_before_click = True
    first = await TaskRunner(store, desktop).run(task["id"])
    assert first["status"] == "needs_review"
    second = await TaskRunner(store, desktop).run(task["id"])
    assert second["status"] == "needs_review"
    assert desktop.calls.count("click") == 1


async def test_existing_different_file_is_preserved(setup):
    store, task, desktop, output = setup
    output.write_text("user's existing data", encoding="utf-8")
    result = await TaskRunner(store, desktop).run(task["id"])
    assert result["status"] == "needs_review"
    assert output.read_text(encoding="utf-8") == "user's existing data"
    assert "click" not in desktop.calls


async def test_cancel_blocks_resume_and_stale_writer(setup):
    store, task, desktop, _ = setup
    cancelled = store.cancel(task["id"])
    assert cancelled["status"] == "cancelled"
    result = await TaskRunner(store, desktop).run(task["id"])
    assert result["status"] == "cancelled"
    assert not desktop.calls
    with pytest.raises(StateConflict):
        store.record(task, "stale", status="running")


def test_status_cli_reads_task_without_backend(setup):
    store, task, _, _ = setup
    process = subprocess.run(
        [sys.executable, "-m", "jarvis.tasks", "--db", str(store.path), "status", task["id"]],
        capture_output=True, text=True, check=True,
    )
    displayed = json.loads(process.stdout)
    assert displayed["id"] == task["id"]
    assert displayed["next_step"] == "discover_window"
    assert displayed["events"][0]["kind"] == "created"
