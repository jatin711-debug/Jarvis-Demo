"""Task CLI: exit codes, up-front validation, clean errors, and orphan-free startup failures."""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import jarvis.tasks.__main__ as cli
from jarvis.tasks.store import TaskStore
from jarvis.utils.config import JarvisConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeDesktop:
    """A Calculator backend that also records lifecycle calls and can fail on demand."""

    def __init__(self) -> None:
        self.display = "0"
        self.pending_plus = False
        self.generation = 0
        self.fail_after: str | None = None
        self.startup_error: Exception | None = None
        self.shutdown_error: Exception | None = None
        self.started = 0
        self.closed = 0

    async def startup(self):
        self.started += 1
        if self.startup_error is not None:
            raise self.startup_error

    async def shutdown(self):
        self.closed += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error

    async def windows(self):
        return {"windows": [{"pid": 101, "window_id": 202, "title": "Calculator",
                             "app_name": "ApplicationFrameHost.exe"}], "truncated": False}

    async def launch(self, name):
        return {"launch": {"running": True}}

    async def observe(self, pid, window_id):
        self.generation += 1
        labels = ("Clear", "One", "Plus", "Two", "Equals")
        return {"pid": pid, "window_id": window_id, "title": "Calculator",
                "snapshot_id": f"s{self.generation}", "degraded": False, "truncated": False,
                "elements": [{"element_index": i, "label": label, "enabled": True}
                             for i, label in enumerate(labels)] +
                            [{"element_index": 5, "label": f"Display is {self.display}"}]}

    async def act(self, action, pid, window_id, snapshot_id, element_index, **kwargs):
        label = ("Clear", "One", "Plus", "Two", "Equals")[element_index]
        if label == "Clear":
            self.display, self.pending_plus = "0", False
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


class Invoker:
    def __init__(self, db: Path, capsys) -> None:
        self.db = db
        self._capsys = capsys

    def __call__(self, *argv: str) -> tuple[int, str, str]:
        code = cli.main(["--db", str(self.db), *argv])
        out, err = self._capsys.readouterr()
        return code, out, err

    def json(self, *argv: str, expect: int = cli.EXIT_OK) -> dict:
        code, out, err = self(*argv)
        assert code == expect, err
        return json.loads(out)  # raises unless stdout is exactly one JSON document


@pytest.fixture(autouse=True)
def stub_config(monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda path=None: JarvisConfig())
    monkeypatch.setattr(cli, "setup_logging", lambda level="INFO", log_dir=None: None)


@pytest.fixture
def desktop(monkeypatch):
    fake = FakeDesktop()
    monkeypatch.setattr(cli, "CuaDesktop", lambda config: fake)
    return fake


@pytest.fixture
def run(tmp_path, capsys, desktop):
    return Invoker(tmp_path / "tasks.sqlite3", capsys)


# --- list / status / cancel -------------------------------------------------

def test_list_status_and_cancel(run):
    assert run.json("list") == {"tasks": []}
    task = TaskStore(run.db).create_calculator()

    listed = run.json("list")["tasks"]
    assert [entry["id"] for entry in listed] == [task["id"]]
    assert listed[0]["status"] == "pending" and "events" not in listed[0]

    shown = run.json("status", task["id"])
    assert shown["next_step"] == "discover_calculator"
    assert shown["events"][0]["kind"] == "created"

    cancelled = run.json("cancel", task["id"])
    assert cancelled["status"] == "cancelled"
    assert run.json("cancel", task["id"])["status"] == "cancelled"
    assert run.json("status", task["id"])["status"] == "cancelled"


def test_read_only_commands_never_start_the_desktop(run, desktop):
    task = TaskStore(run.db).create_calculator()
    run("list")
    run("status", task["id"])
    run("cancel", task["id"])
    run("resume", task["id"])  # cancelled is terminal: reported, not run
    assert desktop.started == 0


# --- unknown ids and bad arguments ------------------------------------------

@pytest.mark.parametrize("command", ["status", "cancel", "resume"])
def test_unknown_task_id_is_a_one_line_usage_error(run, desktop, command):
    code, out, err = run(command, "no-such-task")
    assert code == cli.EXIT_USAGE
    assert out == ""
    assert err.strip() == "jarvis.tasks: unknown task id: no-such-task"
    assert desktop.started == 0


@pytest.mark.parametrize("flags", [
    ["--max-seconds=0"], ["--max-seconds=-1"], ["--max-seconds=nan"], ["--max-seconds=inf"],
    ["--max-seconds=abc"], ["--max-seconds=100000"],
    ["--max-actions=0"], ["--max-actions=-3"], ["--max-actions=1.5"], ["--max-actions=5000"],
    ["--pause-after=0"], ["--pause-after=-1"], ["--pause-after=two"],
])
@pytest.mark.parametrize("command", ["start-calculator", "resume"])
def test_bad_flags_are_rejected_before_anything_starts(run, desktop, command, flags):
    extra = ["some-id"] if command == "resume" else []
    with pytest.raises(SystemExit) as exc:
        run(command, *extra, *flags)
    assert exc.value.code == cli.EXIT_USAGE
    assert desktop.started == 0
    assert not run.db.exists()  # not even the database was opened


def test_bad_flags_report_a_clear_message(run, capsys):
    with pytest.raises(SystemExit):
        run("start-calculator", "--pause-after=0")
    err = capsys.readouterr().err
    assert "--pause-after" in err and "must be at least 1" in err
    assert "Traceback" not in err


def test_valid_flag_values_are_accepted():
    args = cli.parse_args(["resume", "abc", "--pause-after", "2", "--max-seconds", "30.5",
                           "--max-actions", "7"])
    assert (args.pause_after, args.max_seconds, args.max_actions) == (2, 30.5, 7)


def test_rejected_fixture_arguments_are_a_usage_error_not_a_traceback(run, desktop, tmp_path):
    code, out, err = run("start-fixture", "--window-title", "Untitled", "--output",
                         str(tmp_path / "out.txt"), "--text", "hi")
    assert code == cli.EXIT_USAGE
    assert out == "" and err.startswith("jarvis.tasks: ") and "Traceback" not in err
    assert err.count("\n") == 1
    assert TaskStore(run.db).list() == []
    assert desktop.started == 0


def test_unreadable_configuration_is_a_usage_error(run, monkeypatch):
    def broken(path=None):
        raise OSError("no such file")

    monkeypatch.setattr(cli, "load_config", broken)
    code, out, err = run("list")
    assert (code, out) == (cli.EXIT_USAGE, "")
    assert err.strip() == "jarvis.tasks: cannot load configuration: no such file"


# --- exit codes -------------------------------------------------------------

def test_exit_code_constants_do_not_collide():
    codes = [cli.EXIT_OK, cli.EXIT_FAILED, cli.EXIT_USAGE, cli.EXIT_NEEDS_REVIEW,
             cli.EXIT_INTERRUPTED]
    assert len(set(codes)) == len(codes)
    assert cli.EXIT_NEEDS_REVIEW != 2  # argparse's usage-error code


def test_completed_and_paused_runs_exit_zero(run, desktop):
    paused = run.json("start-calculator", "--pause-after", "2")
    assert paused["status"] == "paused" and paused["next_step"] == "enter_one"
    done = run.json("resume", paused["id"])
    assert done["status"] == "completed" and done["result"] == "Calculator displayed 3 after 1+2"
    assert desktop.started == 2 and desktop.closed == 2  # always shut down


def test_needs_review_has_its_own_exit_code(run, desktop):
    desktop.fail_after = "Plus"
    first = run.json("start-calculator", expect=cli.EXIT_NEEDS_REVIEW)
    assert first["status"] == "needs_review"
    assert cli.EXIT_NEEDS_REVIEW not in (cli.EXIT_OK, cli.EXIT_FAILED, cli.EXIT_USAGE)
    again = run.json("resume", first["id"], expect=cli.EXIT_NEEDS_REVIEW)
    assert again["status"] == "needs_review"
    # Looking at a task that needs review is not itself a failure.
    assert run.json("status", first["id"])["status"] == "needs_review"


def test_resume_after_review_exits_zero_once_it_completes(run, desktop):
    desktop.fail_after = "Equals"
    first = run.json("start-calculator", expect=cli.EXIT_NEEDS_REVIEW)
    assert run.json("resume", first["id"])["status"] == "completed"


def test_budget_flags_apply_and_can_be_raised_on_resume(run):
    stopped = run.json("start-calculator", "--max-actions", "1", expect=cli.EXIT_NEEDS_REVIEW)
    assert stopped["budget"]["max_actions"] == 1
    assert "Action budget exhausted" in stopped["error"]
    finished = run.json("resume", stopped["id"], "--max-actions", "8")
    assert finished["status"] == "completed" and finished["budget"]["max_actions"] == 8


def test_ctrl_c_exits_130_without_a_traceback(run, monkeypatch):
    async def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run", interrupted)
    code, out, err = run("list")
    assert (code, out, err) == (cli.EXIT_INTERRUPTED, "", "")


# --- stdout encoding --------------------------------------------------------

def test_json_is_utf8_even_when_stdout_would_be_cp1252(run, monkeypatch):
    title = "Jarvis Cua Smoke Ωmega→日本語1234"
    with pytest.raises(UnicodeEncodeError):
        title.encode("cp1252")  # the case a legacy Windows pipe cannot print
    TaskStore(run.db).create_fixture(window_title=title, output_path=run.db.parent / "out.txt",
                                     text="hi")
    raw = io.BytesIO()
    legacy = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", legacy)

    assert cli.main(["--db", str(run.db), "list"]) == cli.EXIT_OK

    legacy.flush()
    shown = json.loads(raw.getvalue().decode("utf-8"))
    assert shown["tasks"][0]["scope"]["window_title"] == title


def _subprocess_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text("system:\n  log_level: WARNING\n", encoding="utf-8")
    return path


def test_real_process_with_a_cp1252_pipe_still_prints_the_result(tmp_path):
    db = tmp_path / "tasks.sqlite3"
    title = "Jarvis Cua Smoke Ωmega→日本語1234"
    TaskStore(db).create_fixture(window_title=title, output_path=tmp_path / "out.txt", text="hi")
    # PYTHONIOENCODING reproduces a piped Windows stdout on any platform.
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    process = subprocess.run(
        [sys.executable, "-m", "jarvis.tasks", "--config", str(_subprocess_config(tmp_path)),
         "--db", str(db), "list"],
        capture_output=True, env=env, cwd=REPO_ROOT, check=False,
    )
    assert process.returncode == cli.EXIT_OK, process.stderr
    shown = json.loads(process.stdout.decode("utf-8"))
    assert shown["tasks"][0]["scope"]["window_title"] == title


def test_real_process_exit_code_for_an_unknown_id(tmp_path):
    process = subprocess.run(
        [sys.executable, "-m", "jarvis.tasks", "--config", str(_subprocess_config(tmp_path)),
         "--db", str(tmp_path / "tasks.sqlite3"), "status", "nope"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    assert process.returncode == cli.EXIT_USAGE
    assert "unknown task id: nope" in process.stderr
    assert "Traceback" not in process.stderr


# --- startup failures -------------------------------------------------------

def test_desktop_startup_failure_does_not_leave_a_pending_task(run, desktop, caplog):
    caplog.set_level(logging.INFO, logger=cli.__name__)
    desktop.startup_error = RuntimeError("Cua is selected but missing")

    code, out, err = run("start-calculator")

    assert code == cli.EXIT_FAILED
    assert out == ""
    (task,) = TaskStore(run.db).list()
    assert task["status"] == "needs_review"
    assert task["error"] == "Desktop backend failed to start: Cua is selected but missing"
    assert [event["kind"] for event in TaskStore(run.db).events(task["id"])] == [
        "created", "startup_failed"]
    # The id is visible even though the run never produced a result.
    assert f"task created id={task['id']}" in caplog.text
    assert err.strip() == (f"jarvis.tasks: Desktop backend failed to start: Cua is selected "
                           f"but missing (task {task['id']}, marked needs_review)")
    assert desktop.closed == 1

    # Fixing the environment and resuming finishes the very same task.
    desktop.startup_error = None
    assert run.json("resume", task["id"])["status"] == "completed"


def test_the_task_id_is_logged_as_soon_as_the_row_exists(run, caplog):
    caplog.set_level(logging.INFO, logger=cli.__name__)
    done = run.json("start-calculator")
    created = [r.getMessage() for r in caplog.records if r.getMessage().startswith("task created")]
    assert created == [f"task created id={done['id']} workflow=calculator_1_plus_2"]


def test_startup_failure_leaves_a_paused_task_as_it_was(run, desktop):
    paused = run.json("start-calculator", "--pause-after", "2")
    desktop.startup_error = RuntimeError("driver crashed")
    code, _, err = run("resume", paused["id"])
    assert code == cli.EXIT_FAILED and "driver crashed" in err
    assert "marked" not in err
    task = TaskStore(run.db).get(paused["id"])
    assert task["status"] == "paused" and task["current_step"] == 2


def test_a_failing_desktop_shutdown_does_not_hide_the_result(run, desktop, caplog):
    desktop.shutdown_error = TimeoutError()
    done = run.json("start-calculator")
    assert done["status"] == "completed"
    assert "desktop shutdown failed" in caplog.text
