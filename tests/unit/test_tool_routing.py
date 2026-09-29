"""Only the tools for an active capability are exposed and executable."""

from jarvis.agent.react_agent import ReActAgent
from jarvis.agent.routing import select_system_tools
from jarvis.agent.tool_registry import ToolRegistry


class Model:
    provider = "ollama"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages=None, tools=None, images=None):
        self.calls.append({"messages": messages, "tools": tools})
        return {"message": self.responses.pop(0)}


def registry(calls):
    tools = ToolRegistry()

    @tools.tool("get_datetime", "Get date and time", {"type": "object", "properties": {}})
    async def get_datetime():
        calls.append("clock")
        return "5pm"

    @tools.tool("run_command", "Run command", {"type": "object", "properties": {
        "command": {"type": "string"}}, "required": ["command"]})
    async def run_command(command):
        calls.append(command)
        return "ran"

    @tools.tool("read_file", "Read file", {"type": "object", "properties": {
        "path": {"type": "string"}}, "required": ["path"]})
    async def read_file(path):
        calls.append(path)
        return "contents"

    return tools


async def test_only_clock_is_offered_for_time():
    model = Model([{"content": "It is 5pm."}])
    agent = ReActAgent(model, registry([]), "sys")
    assert await agent.run("What time is it?") == "It is 5pm."
    names = [t["function"]["name"] for t in model.calls[0]["tools"]]
    assert names == ["get_datetime"]


async def test_off_route_command_cannot_execute():
    calls = []
    model = Model([
        {"content": "", "tool_calls": [{"function": {"name": "run_command",
            "arguments": {"command": "echo should-not-run"}}}]},
        {"content": "I couldn't run that command."},
    ])
    agent = ReActAgent(model, registry(calls), "sys")
    assert "couldn't run" in await agent.run("What time is it?")
    assert calls == []
    assert "outside this request" in str(model.calls[1]["messages"])


async def test_explanation_has_no_tool_menu():
    model = Model([{"content": "Gravity attracts masses."}])
    agent = ReActAgent(model, registry([]), "sys")
    assert await agent.run("Explain gravity") == "Gravity attracts masses."
    assert model.calls[0]["tools"] is None


def test_file_search_does_not_route_to_browser():
    agent = ReActAgent(Model([]), registry([]), "sys", browser_tools=ToolRegistry())
    chosen, _ = select_system_tools("search the files in this project",
                                    set(agent.tools.list_tools()))
    assert chosen == {"read_file"}
