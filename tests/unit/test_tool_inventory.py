"""Every tool that can be registered is classified, routable, and documented."""

from pathlib import Path

import pytest

from jarvis.agent.inventory import render_tool_table
from jarvis.agent.permissions import KNOWN_TOOLS, REFUSED_TOOLS, PermissionGateway
from jarvis.agent.routing import SYSTEM_CAPABILITIES, Surface, route_request
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.desktop.cua import CuaDesktop, register_desktop_tools
from jarvis.tools.browser_tools import register_browser_tools
from jarvis.tools.system_tools import register_system_tools
from jarvis.utils.config import JarvisConfig, SecurityConfig

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def registries():
    config = JarvisConfig()
    system, browser, desktop = ToolRegistry(), ToolRegistry(), ToolRegistry()
    register_system_tools(system, config.security, llm_client=object())  # object(): enables vision tools
    register_browser_tools(browser)
    register_desktop_tools(desktop, CuaDesktop(config.desktop))
    return system, browser, desktop


def test_every_registered_tool_has_a_permission_classification(registries):
    unclassified = {name for registry in registries for name in registry.list_tools()} - KNOWN_TOOLS
    assert not unclassified, f"tools with no permission classification are always denied: {sorted(unclassified)}"


def test_every_system_tool_is_reachable_by_routing_or_deliberately_refused(registries):
    system = registries[0]
    routed = set().union(*SYSTEM_CAPABILITIES.values())
    unreachable = set(system.list_tools()) - routed - REFUSED_TOOLS
    assert not unreachable, f"registered but never offered to the model: {sorted(unreachable)}"


def test_refused_tools_are_always_denied(tmp_path):
    gateway = PermissionGateway(SecurityConfig(allowed_directories=[str(tmp_path)]))
    for name in sorted(REFUSED_TOOLS):
        assert not gateway.inspect(name, {}).allowed, name


@pytest.mark.parametrize("request_text,tool", [
    ("read the pdf report", "read_pdf"),
    ("summarize this spreadsheet", "read_spreadsheet"),
    ("describe the photo", "read_image"),
    ("find the save button", "find_ui_elements"),
])
def test_document_and_element_tools_are_offered(registries, request_text, tool):
    route = route_request(request_text, system_tools=set(registries[0].list_tools()))
    assert route.surface is Surface.SYSTEM
    assert tool in route.system_tools


def test_document_readers_are_read_only_and_path_scoped(tmp_path):
    inside = tmp_path / "report.pdf"
    gateway = PermissionGateway(SecurityConfig(allowed_directories=[str(tmp_path)]))
    for name in ("read_pdf", "read_spreadsheet", "read_image"):
        allowed = gateway.inspect(name, {"path": str(inside)})
        assert allowed.allowed and allowed.effect == "read" and not allowed.requires_confirmation
        outside = gateway.inspect(name, {"path": str(tmp_path.parent / "elsewhere.pdf")})
        assert not outside.allowed


def test_tools_doc_is_current():
    expected = render_tool_table(JarvisConfig())
    actual = (ROOT / "docs" / "tools.md").read_text(encoding="utf-8")
    assert actual.replace("\r\n", "\n") == expected, (
        "docs/tools.md is stale; regenerate: python -m jarvis --list-tools > docs/tools.md")
