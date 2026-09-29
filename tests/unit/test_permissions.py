"""Permission checks are at dispatch, before confirmation and side effects."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jarvis.agent.permissions import KNOWN_TOOLS, PermissionGateway
from jarvis.agent.react_agent import ReActAgent
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.core.orchestrator import _is_explicit_yes
from jarvis.desktop.cua import register_desktop_tools
from jarvis.tasks.runner import TaskRunner
from jarvis.tasks.store import TaskStore
from jarvis.tools.browser_tools import register_browser_tools
from jarvis.tools.system_tools import register_system_tools
from jarvis.utils.config import SecurityConfig


def policy(tmp_path: Path, **overrides) -> PermissionGateway:
    return PermissionGateway(SecurityConfig(
        allowed_directories=[str(tmp_path)], blocked_commands=[],
        allowed_hosts=["example.com"], allowed_applications=["Calculator"],
        **overrides,
    ))


def write_registry(gateway, confirm=None):
    registry = ToolRegistry(confirm_callback=confirm)
    registry.set_permission_gateway(gateway)
    calls = []

    @registry.tool("write_file", "Write", {"type": "object", "properties": {
        "file_path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["file_path", "content"]})
    async def write_file(file_path, content):
        calls.append((file_path, content))
        Path(file_path).write_text(content, encoding="utf-8")
        return "written"

    return registry, calls


async def test_in_scope_write_approved_for_exact_target(tmp_path):
    seen = []

    async def approve(name, summary):
        seen.append((name, dict(summary)))
        summary["target"] = "changed in callback"  # cannot mutate dispatched args
        return True

    registry, calls = write_registry(policy(tmp_path), approve)
    target = tmp_path / "report.txt"
    assert await registry.execute("write_file", file_path=str(target), content="hello") == "written"
    assert calls == [(str(target), "hello")]
    assert target.read_text(encoding="utf-8") == "hello"
    assert seen[0][1]["effect"] == "local_edit"
    assert str(target) in seen[0][1]["target"]
    assert len(seen[0][1]["action_id"]) == 16


async def test_outside_path_and_prefix_sibling_denied_before_confirmation(tmp_path):
    confirms = []

    async def approve(name, summary):
        confirms.append(name)
        return True

    registry, calls = write_registry(policy(tmp_path), approve)
    outside = tmp_path.parent / f"{tmp_path.name}-outside" / "report.txt"
    result = await registry.execute("write_file", file_path=str(outside), content="no")
    assert result.startswith("Permission denied:")
    assert not calls and not confirms and not outside.exists()


async def test_symlink_escape_denied(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating Windows symlinks requires developer mode or privilege")
    decision = policy(tmp_path).inspect("write_file", {"file_path": str(link / "x.txt"),
                                                       "content": "secret"})
    assert not decision.allowed


async def test_refusal_and_unavailable_confirmation_stop_write(tmp_path):
    async def deny(name, summary):
        return False

    target = tmp_path / "no.txt"
    registry, calls = write_registry(policy(tmp_path), deny)
    assert await registry.execute("write_file", file_path=str(target), content="x") == (
        "Action cancelled by the user.")
    registry.set_confirm_callback(None)
    assert (await registry.execute("write_file", file_path=str(target), content="x")
            ).startswith("Permission denied: confirmation is unavailable")
    assert not target.exists() and not calls


async def test_scope_change_during_confirmation_does_not_inherit_approval(tmp_path):
    gateway = policy(tmp_path)

    async def approve(name, summary):
        gateway.security.allowed_directories = []
        return True

    registry, calls = write_registry(gateway, approve)
    result = await registry.execute("write_file", file_path=str(tmp_path / "x.txt"), content="x")
    assert result.startswith("Permission denied: target changed")
    assert not calls


async def test_explicit_no_confirmation_profile_still_checks_scope(tmp_path):
    gateway = policy(tmp_path, confirm_dangerous_ops=False)
    registry, calls = write_registry(gateway)
    inside = tmp_path / "inside.txt"
    assert await registry.execute("write_file", file_path=str(inside), content="yes") == "written"
    assert len(calls) == 1
    outside = tmp_path.parent / "out.txt"
    assert (await registry.execute("write_file", file_path=str(outside), content="no")
            ).startswith("Permission denied:")
    assert len(calls) == 1


def test_app_url_command_and_broad_delete_checks(tmp_path):
    gateway = policy(tmp_path, command_whitelist=True, allowed_commands=["echo"])
    assert gateway.inspect("desktop_launch", {"name": "Calculator"}).allowed
    assert not gateway.inspect("desktop_launch", {"name": "Notepad"}).allowed
    assert gateway.inspect("browser_navigate", {"url": "https://example.com/path"}).allowed
    assert not gateway.inspect("browser_navigate", {"url": "https://evil.example/path"}).allowed
    assert not gateway.inspect("browser_navigate", {"url": "file:///tmp/x"}).allowed
    assert not gateway.inspect("browser_click", {"selector": "Submit"}).allowed
    assert gateway.inspect("http_request", {"url": "https://example.com", "method": "GET"}
                           ).requires_confirmation is False
    assert gateway.inspect("http_request", {"url": "https://example.com", "method": "POST"}
                           ).requires_confirmation is True
    assert gateway.inspect("run_command", {"command": "echo hello"}).allowed
    assert not gateway.inspect("run_command", {"command": "echo hello & whoami"}).allowed
    assert not gateway.inspect("run_command", {"command": "powershell -EncodedCommand AAA"}).allowed
    unrestricted = policy(tmp_path, command_whitelist=False)
    assert not unrestricted.inspect("scheduler_create", {"task_name": "bad",
                                                        "command": "format c:"}).allowed
    assert not gateway.inspect("kill_process", {"target": "explorer"}).allowed
    assert not gateway.inspect("delete_file", {"path": str(tmp_path)}).allowed


def test_copy_checks_both_paths_and_workspace_delete_is_protected(tmp_path):
    gateway = policy(tmp_path, confirm_dangerous_ops=False)
    source = tmp_path / "source.txt"
    outside = tmp_path.parent / "destination.txt"
    assert not gateway.inspect("copy_file", {"source": str(source),
                                             "destination": str(outside)}).allowed
    broad = PermissionGateway(SecurityConfig(allowed_directories=[str(Path.cwd())],
                                             confirm_dangerous_ops=False))
    assert not broad.inspect("delete_file", {"path": str(Path.cwd())}).allowed


async def test_scheduler_name_injection_denied_and_safe_call_has_no_shell(tmp_path):
    gateway = policy(tmp_path, command_whitelist=True, allowed_commands=["echo"])

    async def approve(name, summary):
        return True

    registry = ToolRegistry(confirm_callback=approve)
    registry.set_permission_gateway(gateway)
    register_system_tools(registry, gateway.security)
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stderr="", stdout="")) as run:
        denied = await registry.execute("scheduler_create", task_name='bad" & whoami',
                                        command="echo hello")
        assert denied.startswith("Permission denied:")
        run.assert_not_called()
        approved = await registry.execute("scheduler_create", task_name="safe_task",
                                          command="echo hello")
        assert "Scheduled" in approved
        assert run.call_args.args[0][:5] == ["schtasks", "/create", "/tn", "Jarvis_safe_task", "/tr"]
        assert run.call_args.kwargs["shell"] is False


def test_confirmation_fingerprint_covers_content(tmp_path):
    gateway = policy(tmp_path)
    target = str(tmp_path / "x.txt")
    first = gateway.inspect("write_file", {"file_path": target, "content": "alpha"})
    second = gateway.inspect("write_file", {"file_path": target, "content": "beta"})
    assert first.target == second.target
    assert first.action_id != second.action_id


@pytest.mark.parametrize(("heard", "approved"), [
    ("yes", True), ("Yes.", True), ("yes, but no", False),
    ("yesterday", False), ("no", False), ("okay", False),
])
def test_voice_approval_requires_exact_yes(heard, approved):
    assert _is_explicit_yes(heard) is approved


def test_all_builtin_tools_have_permission_classification():
    system = ToolRegistry()
    browser = ToolRegistry()
    desktop = ToolRegistry()
    register_system_tools(system, SecurityConfig())
    register_browser_tools(browser)
    register_desktop_tools(desktop, object())
    actual = set(system.list_tools() + browser.list_tools() + desktop.list_tools())
    assert actual <= KNOWN_TOOLS


class Model:
    provider = "ollama"

    def __init__(self):
        self.calls = 0

    async def chat(self, messages=None, tools=None, images=None):
        self.calls += 1
        return {"message": {"content": "", "tool_calls": [{"function": {
            "name": "write_file", "arguments": {"file_path": "C:/outside.txt", "content": "no"}}}]}}


async def test_agent_ends_turn_after_permission_denial(tmp_path):
    registry, calls = write_registry(policy(tmp_path))
    model = Model()
    agent = ReActAgent(model, registry, "sys")
    response = await agent.run("write a file")
    assert response.startswith("Permission denied:")
    assert model.calls == 1
    assert not calls


async def test_durable_fixture_checks_path_scope_before_desktop_actions(tmp_path):
    store = TaskStore(tmp_path / "task.sqlite3")
    target = tmp_path / "output.txt"
    task = store.create_fixture(window_title="Jarvis Cua Smoke permission1234",
                                output_path=target, text="hello")

    class Desktop:
        def __init__(self):
            self.calls = 0

        async def windows(self):
            self.calls += 1
            raise AssertionError("Desktop should not be touched")

    desktop = Desktop()
    gateway = PermissionGateway(SecurityConfig(allowed_directories=[]))
    result = await TaskRunner(store, desktop, gateway).run(task["id"])
    assert result["status"] == "needs_review"
    assert "outside" in result["error"]
    assert desktop.calls == 0
