"""Desktop contract tests run without native dependencies or GUI access."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jarvis.agent.react_agent import ReActAgent
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.desktop.cua import CuaDesktop, register_desktop_tools
from jarvis.utils.config import DesktopConfig


class Driver:
    def __init__(self):
        self.calls = []
        self.generation = 0
        self.degraded = False
        self.shutdown = AsyncMock()

    async def call_tool(self, name, arguments):
        self.calls.append((name, json.loads(arguments)))
        if name == "list_windows":
            data = {"windows": [{"pid": 1, "window_id": 2, "title": "Fixture"}]}
        elif name == "get_window_state":
            self.generation += 1
            data = {"pid": 1, "window_id": 2, "snapshot_id": f"s{self.generation}",
                    "degraded": self.degraded, "elements": [
                        {"element_index": 0, "element_token": "token", "enabled": True,
                         "role": "edit", "label": "Text", "value": ""}]}
        else:
            data = {"dispatched": True}
        return SimpleNamespace(is_error=False, structured_json=json.dumps(data), text="")


@pytest.fixture
def desktop():
    return CuaDesktop(DesktopConfig(), driver=Driver())


async def observed(desktop):
    await desktop.windows()
    return await desktop.observe(1, 2)


async def test_requires_discovered_window(desktop):
    with pytest.raises(ValueError, match="desktop_windows"):
        await desktop.observe(1, 2)
    assert not desktop._driver.calls


async def test_windows_rejects_extra_ids_with_corrective_schema(desktop):
    registry = ToolRegistry()
    register_desktop_tools(registry, desktop)
    error = await registry.execute("desktop_windows", pid=1, window_id=2)
    assert "Unknown parameter" in error
    assert "Required parameter names: []" in error
    assert "must be removed" in error
    assert not desktop._driver.calls
    result = json.loads(await registry.execute("desktop_windows"))
    assert result["windows"][0]["pid"] == 1


async def test_observe_does_not_require_snapshot_or_element(desktop):
    registry = ToolRegistry()
    register_desktop_tools(registry, desktop)
    await registry.execute("desktop_windows")
    error = await registry.execute("desktop_observe", pid=1, window_id=2,
                                   snapshot_id="fake", element_index=0)
    assert "Required parameter names: ['pid', 'window_id']" in error
    state = json.loads(await registry.execute("desktop_observe", pid=1, window_id=2))
    assert state["snapshot_id"]


async def test_action_uses_token_and_returns_fresh_evidence(desktop):
    state = await observed(desktop)
    result = await desktop.act("type_text", 1, 2, state["snapshot_id"], 0, text="hello")
    assert result["after"]["snapshot_id"] != state["snapshot_id"]
    name, args = desktop._driver.calls[-2]
    assert name == "type_text"
    assert args["element_token"] == "token"
    assert args["delivery_mode"] == "background"
    assert "element_token" not in json.dumps(result["after"])


@pytest.mark.parametrize("mode", ["wrong", "expired", "degraded", "rediscovered"])
async def test_rejects_invalid_observation(desktop, mode):
    desktop._driver.degraded = mode == "degraded"
    state = await observed(desktop)
    if mode == "expired":
        _, snapshot, tokens = desktop._snapshots[(1, 2)]
        desktop._snapshots[(1, 2)] = (0, snapshot, tokens)
    if mode == "rediscovered":
        await desktop.windows()
    with pytest.raises(ValueError, match="stale"):
        await desktop.act("click", 1, 2, "wrong" if mode == "wrong" else state["snapshot_id"], 0)
    assert all(name != "click" for name, _ in desktop._driver.calls)


async def test_unknown_element_rejected(desktop):
    state = await observed(desktop)
    with pytest.raises(ValueError, match="Element"):
        await desktop.act("click", 1, 2, state["snapshot_id"], 99)


async def test_cancelled_native_call_disables_further_dispatch(desktop):
    desktop._driver.call_tool = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await desktop.windows()
    with pytest.raises(RuntimeError, match="interrupted"):
        await desktop.windows()
    assert desktop._driver.call_tool.await_count == 1


async def test_timeout_disables_further_dispatch(desktop):
    desktop._driver.call_tool = AsyncMock(side_effect=TimeoutError)
    with pytest.raises(TimeoutError):
        await desktop.windows()
    with pytest.raises(RuntimeError, match="interrupted"):
        await desktop.windows()


async def test_shutdown_is_idempotent(desktop):
    driver = desktop._driver
    await desktop.shutdown()
    await desktop.shutdown()
    driver.shutdown.assert_awaited_once()
    with pytest.raises(RuntimeError, match="not running"):
        await desktop.windows()


async def test_failed_post_observation_is_not_claimed_verified(desktop):
    state = await observed(desktop)
    desktop._observe = AsyncMock(side_effect=RuntimeError("window closed"))
    result = await desktop.act("click", 1, 2, state["snapshot_id"], 0)
    assert result["verified"] is False
    assert "do not repeat" in result["next"]


def test_six_tools_and_selective_routing(desktop):
    registry = ToolRegistry()
    register_desktop_tools(registry, desktop)
    assert len(registry) == 6
    assert registry.get("desktop_observe").repeatable
    assert registry.get("desktop_click").requires_confirmation
    agent = ReActAgent(llm=SimpleNamespace(provider="ollama"), tools=ToolRegistry(),
                       system_prompt="test", desktop_tools=registry)
    assert agent._needs_desktop_tools("open notepad")
    assert not agent._needs_desktop_tools("search the web and click a link")
    agent._desktop_active = True
    assert agent._needs_desktop_tools("do that again")
    assert not agent._needs_desktop_tools("what is the weather")


async def test_agent_routes_native_calls_and_allows_repeated_observations(desktop):
    registry = ToolRegistry()
    register_desktop_tools(registry, desktop)

    def call(name, args):
        return {"message": {"content": "", "tool_calls": [
            {"function": {"name": name, "arguments": args}}]}}

    llm = SimpleNamespace(provider="ollama", chat=AsyncMock(side_effect=[
        call("desktop_windows", {}),
        call("desktop_observe", {"pid": 1, "window_id": 2}),
        call("desktop_observe", {"pid": 1, "window_id": 2}),
        {"message": {"content": "The fixture has an editable text control."}},
    ]))
    agent = ReActAgent(llm=llm, tools=ToolRegistry(), system_prompt="test", desktop_tools=registry)
    answer = await agent.run("Inspect the desktop window")
    assert "editable" in answer
    assert [name for name, _ in desktop._driver.calls].count("get_window_state") == 2
    offered = llm.chat.call_args_list[0].kwargs["tools"]
    assert len(offered) == 6
    assert all(t["function"]["name"].startswith("desktop_") for t in offered)
