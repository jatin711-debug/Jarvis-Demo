"""Calculator acceptance scenario: exact target, checkpoints, and uncertain actions."""

import pytest

from jarvis.tasks.runner import TaskRunner
from jarvis.tasks.store import TaskStore


class CalculatorDesktop:
    def __init__(self):
        self.display = "0"
        self.pending_plus = False
        self.calls = []
        self.app_name = "ApplicationFrameHost.exe"
        self.pid = 101
        self.window_id = 202
        self.fail_after = None
        self.fail_before = None
        self.closed = False
        self.generation = 0

    async def windows(self):
        self.calls.append("windows")
        return {"windows": [] if self.closed else [{"pid": self.pid,
                "window_id": self.window_id, "title": "Calculator", "app_name": self.app_name}],
                "truncated": False}

    async def launch(self, name):
        assert name == "Calculator"
        self.calls.append("launch")
        self.closed = False
        return {"launch": {"running": True}}

    async def observe(self, pid, window_id):
        assert (pid, window_id) == (self.pid, self.window_id)
        self.calls.append("observe")
        self.generation += 1
        labels = ("Clear", "One", "Plus", "Two", "Equals")
        return {"pid": pid, "window_id": window_id, "title": "Calculator",
                "snapshot_id": f"s{self.generation}", "degraded": False, "truncated": False,
                "elements": [{"element_index": i, "label": label, "enabled": True}
                             for i, label in enumerate(labels)] +
                            [{"element_index": 5, "label": f"Display is {self.display}"}]}

    async def act(self, action, pid, window_id, snapshot_id, element_index, **kwargs):
        assert action == "click"
        assert (pid, window_id) == (self.pid, self.window_id)
        assert snapshot_id == f"s{self.generation}"
        label = ("Clear", "One", "Plus", "Two", "Equals")[element_index]
        self.calls.append(label)
        if self.fail_before == label:
            self.fail_before = None
            raise TimeoutError("dispatch uncertain before action")
        if label == "Clear":
            self.display = "0"
            self.pending_plus = False
        elif label == "One":
            self.display = "1"
        elif label == "Plus":
            self.pending_plus = True
        elif label == "Two":
            self.display = "2"
        elif label == "Equals":
            self.display = "3" if self.pending_plus else "2"
        if self.fail_after == label:
            self.fail_after = None
            raise TimeoutError("response lost after action")
        return {"after": await self.observe(pid, window_id)}


@pytest.fixture
def setup(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_calculator()
    return store, task, CalculatorDesktop()


async def test_calculator_completes_and_repeats(setup):
    store, task, desktop = setup
    first = await TaskRunner(store, desktop).run(task["id"])
    assert first["status"] == "completed"
    assert first["result"] == "Calculator displayed 3 after 1+2"
    assert first["step_status"] == ["verified"] * 6
    assert desktop.display == "3"
    second = store.create_calculator()
    again = await TaskRunner(store, desktop).run(second["id"])
    assert again["status"] == "completed"
    assert desktop.calls.count("Clear") == 1
    assert desktop.calls.count("Equals") == 2


async def test_restart_and_changed_window_id_are_rediscovered(setup):
    store, task, desktop = setup
    paused = await TaskRunner(store, desktop).run(task["id"], pause_after=2)
    assert paused["current_step"] == 2
    desktop.window_id = 303
    result = await TaskRunner(TaskStore(store.path), desktop).run(task["id"])
    assert result["status"] == "completed"
    assert desktop.calls.count("One") == 1


async def test_wrong_app_is_not_clicked(setup):
    store, task, desktop = setup
    desktop.app_name = "NotCalculator.exe"
    result = await TaskRunner(store, desktop).run(task["id"])
    assert result["status"] == "needs_review"
    assert "click" not in desktop.calls
    assert "One" not in desktop.calls


async def test_uncertain_plus_does_not_replay(setup):
    store, task, desktop = setup
    desktop.fail_after = "Plus"
    first = await TaskRunner(store, desktop).run(task["id"])
    assert first["status"] == "needs_review"
    assert first["attempt"] == {"step": "plus"}
    second = await TaskRunner(TaskStore(store.path), desktop).run(task["id"])
    assert second["status"] == "needs_review"
    assert desktop.calls.count("Plus") == 1


async def test_uncertain_equals_reconciles_from_display(setup):
    store, task, desktop = setup
    desktop.fail_after = "Equals"
    first = await TaskRunner(store, desktop).run(task["id"])
    assert first["status"] == "needs_review"
    final = await TaskRunner(TaskStore(store.path), desktop).run(task["id"])
    assert final["status"] == "completed"
    assert final["evidence"][-1]["reconciled"] is True
    assert desktop.calls.count("Equals") == 1


async def test_denied_or_uncertain_click_stops_without_retry(setup):
    store, task, desktop = setup
    desktop.fail_before = "One"
    first = await TaskRunner(store, desktop).run(task["id"])
    assert first["status"] == "needs_review"
    again = await TaskRunner(store, desktop).run(task["id"])
    assert again["status"] == "needs_review"
    assert desktop.calls.count("One") == 1
