from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jarvis.agent.react_agent import ReActAgent
from jarvis.agent.tool_registry import ToolRegistry


@pytest.mark.parametrize("stream", [True, False])
async def test_declined_action_ends_turn_without_model_retry(stream):
    handler = AsyncMock()
    registry = ToolRegistry(confirm_callback=AsyncMock(return_value=False))
    registry.tool("run_command", "Run command", {"type": "object", "properties": {}},
                  requires_confirmation=True)(handler)
    calls = [{"function": {"name": "run_command", "arguments": {}}}]

    async def events(**kwargs):
        yield {"type": "done", "content": "", "tool_calls": calls}

    llm = SimpleNamespace(provider="ollama", chat=AsyncMock(return_value={
        "message": {"content": "", "tool_calls": calls}}), chat_stream_events=events)
    agent = ReActAgent(llm, registry, "test")
    if stream:
        response = "".join([text async for text in agent.run_stream("run a command")])
    else:
        response = await agent.run("run a command")
        assert llm.chat.await_count == 1
    assert response == "Okay, I cancelled that action."
    handler.assert_not_awaited()
