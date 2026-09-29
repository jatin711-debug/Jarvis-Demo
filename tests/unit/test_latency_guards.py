import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jarvis.agent.react_agent import ReActAgent
from jarvis.agent.routing import is_small_talk
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.core.orchestrator import Orchestrator
from jarvis.utils.config import JarvisConfig
from jarvis.utils.deadlines import wait_with_progress


@pytest.mark.parametrize("text", ["How are you doing?", "Hello!", "Good morning.",
                                  "How you doing?", "Hey Jarvis, how you doing?",
                                  "How are you today, Jarvis?"])
def test_greetings(text):
    assert is_small_talk(text)


@pytest.mark.parametrize("text", ["Hello, open Notepad", "How are my downloads doing?",
                                  "How you doing? Open Calculator.",
                                  "Remember my birthday", "What did I tell you yesterday?"])
def test_mixed_and_memory_requests_keep_full_path(text):
    assert not is_small_talk(text)


async def test_greeting_bypasses_memory_read_and_write():
    app = Orchestrator(JarvisConfig())
    app.memory = SimpleNamespace(get_context=AsyncMock(), add_conversation=AsyncMock())
    assert await app._get_memory_context("How are you doing?") == []
    app._save_conversation("How are you doing?", "Fine!")
    app.memory.get_context.assert_not_called()
    app.memory.add_conversation.assert_not_called()


async def test_memory_timeout_does_not_queue_more_work():
    config = JarvisConfig()
    config.memory.retrieval_timeout = 0.01
    app = Orchestrator(config)
    release = asyncio.Event()

    async def lookup(text):
        await release.wait()
        return ["late result"]

    app.memory = SimpleNamespace(get_context=AsyncMock(side_effect=lookup))
    assert await app._get_memory_context("My preferences") == []
    assert await app._get_memory_context("My schedule") == []
    assert app.memory.get_context.await_count == 1
    release.set()
    await app._memory_lookup
    config.memory.retrieval_timeout = 1.0
    assert await app._get_memory_context("My schedule") == ["late result"]


async def test_fast_path_has_compact_prompt_and_no_tools():
    llm = SimpleNamespace(provider="ollama", chat=AsyncMock(return_value={"message": {"content": "Doing well!"}}))
    agent = ReActAgent(llm, ToolRegistry(), "X" * 10000)
    assert await agent.run("How are you doing?") == "Doing well!"
    sent = llm.chat.call_args.kwargs
    assert sent["tools"] is None
    assert sum(len(m["content"]) for m in sent["messages"]) < 250


@pytest.mark.parametrize("message", ["How are you doing?", "Explain gravity"])
async def test_stream_first_response_timeout_closes_stream(message):
    closed = asyncio.Event()

    async def slow(**kwargs):
        try:
            await asyncio.Event().wait()
            yield {"type": "content", "text": "unreachable"}
        finally:
            closed.set()

    llm = SimpleNamespace(provider="ollama", config=SimpleNamespace(first_response_timeout=0.01),
                          chat_stream_events=slow)
    agent = ReActAgent(llm, ToolRegistry(), "test")
    answer = "".join([part async for part in agent.run_stream(message)])
    assert "taking too long" in answer
    assert closed.is_set()


async def test_deadline_cancels_underlying_wait():
    closed = asyncio.Event()

    async def slow():
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    with pytest.raises(TimeoutError):
        await wait_with_progress(slow(), 0.01, "test")
    assert closed.is_set()
