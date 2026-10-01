"""Unit tests for the shell-command security gate in jarvis.tools.system_tools."""

from jarvis.agent.permissions import catastrophic_reason as _catastrophic_reason
from jarvis.agent.permissions import first_executable as _first_executable
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.tools.system_tools import register_system_tools
from jarvis.utils.config import SecurityConfig


class TestCatastrophicReason:
    def test_format_disk(self):
        assert _catastrophic_reason("format c:") == "formats a disk"
        assert _catastrophic_reason("FORMAT D:") == "formats a disk"

    def test_diskpart(self):
        assert _catastrophic_reason("diskpart") is not None

    def test_rm_rf_root_and_home(self):
        assert _catastrophic_reason("rm -rf /") is not None
        assert _catastrophic_reason("rm -rf ~") is not None

    def test_rm_rf_subdirectory_allowed(self):
        assert _catastrophic_reason("rm -rf ./node_modules") is None

    def test_del_system_paths(self):
        assert _catastrophic_reason("del /s c:\\windows") is not None

    def test_bcdedit(self):
        assert _catastrophic_reason("bcdedit /set safeboot minimal") is not None

    def test_normal_commands_allowed(self):
        assert _catastrophic_reason("dir") is None
        assert _catastrophic_reason("echo hello") is None
        assert _catastrophic_reason("ipconfig /all") is None


class TestFirstExecutable:
    def test_plain_name(self):
        assert _first_executable("notepad") == "notepad"

    def test_quoted_path_with_extension(self):
        assert _first_executable('"C:\\Apps\\code.exe" --new-window') == "code"

    def test_script_extension_stripped(self):
        assert _first_executable("deploy.cmd /q") == "deploy"

    def test_empty_command(self):
        assert _first_executable("") == ""


def _registry_with(security: SecurityConfig) -> ToolRegistry:
    registry = ToolRegistry()
    register_system_tools(registry, security)
    return registry


class TestRunCommandGate:
    async def test_catastrophic_command_blocked(self):
        registry = _registry_with(SecurityConfig())
        result = await registry.execute("run_command", command="format c:")
        assert result.startswith("Blocked")
        assert "formats a disk" in result

    async def test_config_blocklist_enforced(self):
        registry = _registry_with(SecurityConfig())  # default blocks "shutdown"
        result = await registry.execute("run_command", command="shutdown /s /t 0")
        assert "Block" in result

    async def test_whitelist_rejects_unknown_executable(self):
        registry = _registry_with(SecurityConfig(command_whitelist=True))
        result = await registry.execute("run_command", command="someunknownexe --foo")
        assert "not in the command whitelist" in result

    async def test_whitelisted_command_runs(self):
        registry = _registry_with(SecurityConfig())  # "echo" is whitelisted by default
        result = await registry.execute("run_command", command="echo jarvis-test")
        assert "jarvis-test" in result

    async def test_whitelist_disabled_runs_anything(self):
        security = SecurityConfig(command_whitelist=False, blocked_commands=[])
        registry = _registry_with(security)
        result = await registry.execute("run_command", command="echo jarvis-open")
        assert "jarvis-open" in result
