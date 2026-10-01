"""Behavior of the built-in system tools: one command policy, bounded work, clean cleanup."""

import asyncio
import os
import sys
import time

import psutil
import pytest

from jarvis.agent.permissions import PermissionGateway, first_executable, validate_command
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.tools import system_tools
from jarvis.tools.system_tools import register_system_tools
from jarvis.utils.config import SecurityConfig


def make_registry(**security) -> ToolRegistry:
    registry = ToolRegistry()
    register_system_tools(registry, SecurityConfig(**security))
    return registry


def python_command(code: str) -> str:
    """A shell command line that runs ``code`` with this interpreter."""
    return f'"{sys.executable}" -c "{code}"'


async def wait_until_gone(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        await asyncio.sleep(0.05)
    return not psutil.pid_exists(pid)


# -- command policy ----------------------------------------------------------


class TestFirstExecutable:
    def test_quoted_path_containing_spaces(self):
        assert first_executable('"C:\\Program Files\\Git\\git.exe" status') == "git"

    def test_plain_and_scripts(self):
        assert first_executable("Git status") == "git"
        assert first_executable("deploy.cmd /q") == "deploy"
        assert first_executable("") == ""


WHITELIST = {"command_whitelist": True, "allowed_commands": ["git", "echo"], "blocked_commands": []}


class TestValidateCommand:
    @pytest.mark.parametrize("command", ["git status", "ECHO hi", '"git" log', "echo 100"])
    def test_allowed_under_whitelist(self, command):
        assert validate_command(SecurityConfig(**WHITELIST), command) == command

    @pytest.mark.parametrize("command", [
        "git status & whoami",
        "git status; whoami",
        "git status | findstr x",
        "echo hi > out.txt",
        "echo %USERPROFILE%",
        "echo $(whoami)",
        "git status\nwhoami",
    ])
    def test_compositions_refused_under_whitelist(self, command):
        with pytest.raises(ValueError, match="compositions"):
            validate_command(SecurityConfig(**WHITELIST), command)

    @pytest.mark.parametrize("command", ["powershell -c x", "cmd /c dir", "python -c 1", "pwsh x"])
    def test_interpreters_refused_under_whitelist(self, command):
        with pytest.raises(ValueError, match="interpreters"):
            validate_command(SecurityConfig(**{**WHITELIST, "allowed_commands": ["git", "powershell",
                                                                                 "cmd", "python", "pwsh"]}),
                             command)

    @pytest.mark.parametrize("command", [
        r"C:\Users\me\Downloads\git.exe status",
        r".\git.exe status",
        "/tmp/git status",
        '"C:\\Program Files\\Git\\cmd\\git.exe" status',
    ])
    def test_path_qualified_executable_cannot_borrow_a_whitelisted_name(self, command):
        with pytest.raises(ValueError, match="by name"):
            validate_command(SecurityConfig(**WHITELIST), command)

    def test_unlisted_executable_names_itself(self):
        with pytest.raises(ValueError, match="'curl' is not in the command whitelist"):
            validate_command(SecurityConfig(**WHITELIST), "curl example.com")

    def test_catastrophic_and_blocklist_apply_in_every_profile(self):
        open_profile = SecurityConfig(command_whitelist=False, blocked_commands=["shutdown"])
        with pytest.raises(ValueError, match="formats a disk"):
            validate_command(open_profile, "format c:")
        with pytest.raises(ValueError, match="shutdown"):
            validate_command(open_profile, "shutdown /s")

    def test_open_profile_allows_shell_features(self):
        open_profile = SecurityConfig(command_whitelist=False, blocked_commands=[])
        assert validate_command(open_profile, "echo a & echo b") == "echo a & echo b"

    @pytest.mark.parametrize("raw", ["", "   ", None, 7])
    def test_requires_a_command(self, raw):
        with pytest.raises(ValueError, match="exact command"):
            validate_command(SecurityConfig(), raw)


COMMANDS = [
    "git status", "git status & whoami", "powershell -c x", "curl x", "format c:", "shutdown /s",
    r"C:\evil\git.exe", "echo ok", "echo %PATH%", "",
]


@pytest.mark.parametrize("command", COMMANDS)
async def test_tool_and_gateway_agree_on_every_command(command):
    """The handler and the gateway share one policy, so neither can be looser."""
    security = SecurityConfig(command_whitelist=True, allowed_commands=["git", "echo"],
                              blocked_commands=["shutdown"])
    gateway_allows = PermissionGateway(security).inspect("run_command", {"command": command}).allowed
    registry = ToolRegistry()
    register_system_tools(registry, security)
    handler_result = await registry.execute("run_command", command=command)
    refused_by_policy = False
    try:
        validate_command(security, command)
    except ValueError as exc:
        refused_by_policy = True
        assert handler_result == str(exc)
    assert gateway_allows is (not refused_by_policy)


# -- run_command -------------------------------------------------------------


class TestRunCommand:
    async def test_reports_a_nonzero_exit_code(self):
        registry = make_registry(command_whitelist=False, blocked_commands=[])
        result = await registry.execute("run_command", command=python_command("import sys; sys.exit(3)"))
        assert "[exit code: 3]" in result

    async def test_success_has_no_exit_marker(self):
        registry = make_registry(command_whitelist=False, blocked_commands=[])
        result = await registry.execute("run_command", command=python_command("print(42)"))
        assert result == "42"

    async def test_timeout_kills_the_whole_process_tree(self, tmp_path, monkeypatch):
        monkeypatch.setattr(system_tools, "_COMMAND_TIMEOUT_S", 1.0)
        pid_file = tmp_path / "pid"
        code = (f"import os, pathlib, time; pathlib.Path(r'{pid_file}').write_text(str(os.getpid())); "
                "time.sleep(60)")
        registry = make_registry(command_whitelist=False, blocked_commands=[])
        result = await registry.execute("run_command", command=python_command(code))
        assert "timed out after 1 seconds and was stopped" in result
        pid = int(pid_file.read_text())
        assert await wait_until_gone(pid), "the timed-out command is still running"

    async def test_cancellation_kills_the_process_too(self, tmp_path):
        pid_file = tmp_path / "pid"
        code = (f"import os, pathlib, time; pathlib.Path(r'{pid_file}').write_text(str(os.getpid())); "
                "time.sleep(60)")
        task = asyncio.create_task(system_tools._run_shell(python_command(code), timeout=60))
        for _ in range(200):
            if pid_file.exists() and pid_file.read_text():
                break
            await asyncio.sleep(0.05)
        pid = int(pid_file.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await wait_until_gone(pid), "a cancelled command kept running"


# -- file tools --------------------------------------------------------------


class TestFileTools:
    async def test_read_file_never_loads_more_than_it_returns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(system_tools, "_READ_LIMIT_CHARS", 100)
        big = tmp_path / "big.txt"
        big.write_text("x" * 100_000, encoding="utf-8")
        result = await make_registry().execute("read_file", file_path=str(big))
        assert result.startswith("x" * 100)
        assert "truncated; file is 100,000 bytes" in result
        assert len(result) < 200

    async def test_read_file_small_file_is_returned_whole(self, tmp_path):
        note = tmp_path / "note.txt"
        note.write_text("hello\nworld", encoding="utf-8")
        assert await make_registry().execute("read_file", file_path=str(note)) == "hello\nworld"

    async def test_write_file_replaces_content_and_leaves_no_temp_file(self, tmp_path):
        target = tmp_path / "sub" / "out.txt"
        registry = make_registry()
        await registry.execute("write_file", file_path=str(target), content="one")
        await registry.execute("write_file", file_path=str(target), content="two")
        assert target.read_text(encoding="utf-8") == "two"
        assert [p.name for p in target.parent.iterdir()] == ["out.txt"]

    async def test_failed_write_keeps_the_original_file(self, tmp_path, monkeypatch):
        target = tmp_path / "keep.txt"
        target.write_text("original", encoding="utf-8")

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        result = await make_registry().execute("write_file", file_path=str(target), content="new")
        assert "disk full" in result
        assert target.read_text(encoding="utf-8") == "original"
        assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]


class TestSearchFiles:
    @pytest.fixture
    def tree(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "report.txt").write_text("quarterly revenue", encoding="utf-8")
        (tmp_path / "report2.txt").write_text("nothing here", encoding="utf-8")
        (tmp_path / "notes.md").write_text("revenue", encoding="utf-8")
        return tmp_path

    async def test_matches_by_name(self, tree):
        result = await make_registry().execute("search_files", directory=str(tree), pattern="report*")
        assert "report.txt" in result and "report2.txt" in result and "notes.md" not in result

    async def test_matches_by_content_case_insensitively(self, tree):
        result = await make_registry().execute(
            "search_files", directory=str(tree), pattern="*.txt", content="REVENUE")
        assert "report.txt" in result and "report2.txt" not in result

    async def test_large_files_are_matched_by_name_only(self, tree, monkeypatch):
        monkeypatch.setattr(system_tools, "_SEARCH_MAX_CONTENT_BYTES", 5)
        result = await make_registry().execute(
            "search_files", directory=str(tree), pattern="*.txt", content="revenue")
        assert result.startswith("No files matching")

    async def test_results_are_capped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(system_tools, "_SEARCH_MAX_RESULTS", 3)
        for i in range(10):
            (tmp_path / f"f{i}.log").write_text("x", encoding="utf-8")
        result = await make_registry().execute("search_files", directory=str(tmp_path), pattern="*.log")
        assert len(result.splitlines()) == 4 and "capped at 3 results" in result

    async def test_search_stops_at_its_time_budget_and_says_so(self, tree, monkeypatch):
        monkeypatch.setattr(system_tools, "_SEARCH_TIME_BUDGET_S", -1.0)
        result = await make_registry().execute("search_files", directory=str(tree), pattern="*")
        assert "search stopped" in result

    async def test_missing_directory(self, tmp_path):
        result = await make_registry().execute(
            "search_files", directory=str(tmp_path / "nope"), pattern="*")
        assert result.startswith("Directory not found")


# -- the event loop stays free -----------------------------------------------


async def test_blocking_tool_work_does_not_stall_the_event_loop(tmp_path, monkeypatch):
    """A slow file walk runs on a worker thread, so audio/barge-in tasks keep ticking."""
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    real_walk = os.walk

    def slow_walk(*args, **kwargs):
        time.sleep(0.4)
        yield from real_walk(*args, **kwargs)

    monkeypatch.setattr(os, "walk", slow_walk)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    background = asyncio.create_task(ticker())
    try:
        result = await make_registry().execute("search_files", directory=str(tmp_path), pattern="*.txt")
    finally:
        background.cancel()
    assert "x.txt" in result
    assert ticks >= 10, f"event loop was blocked (only {ticks} ticks during a 0.4s tool call)"


# -- system information ------------------------------------------------------


class TestSystemInfo:
    async def test_uptime_is_fast_and_readable(self):
        started = time.monotonic()
        result = await make_registry().execute("get_uptime")
        assert time.monotonic() - started < 1.0
        assert result.startswith("Uptime: ") and "booted" in result

    async def test_process_listing_has_a_header_and_rows(self):
        result = await make_registry().execute("get_running_processes", filter_name="python")
        lines = result.splitlines()
        assert lines[0].split()[:2] == ["PID", "Name"]
        assert len(lines) > 1
