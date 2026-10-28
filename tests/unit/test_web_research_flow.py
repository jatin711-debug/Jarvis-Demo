"""Web requests must search, report failures honestly, and support cancellation."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.agent.react_agent import ReActAgent
from jarvis.agent.routing import is_web_research
from jarvis.agent.tool_registry import CANCELLED_MESSAGE, ToolRegistry
from jarvis.core.orchestrator import Orchestrator, _Heard
from jarvis.utils.config import JarvisConfig
from jarvis.voice.transcript import is_garbage


class LLM:
    provider = "llamacpp"

    def __init__(self, answer="Here are the results."):
        self.answer = answer
        self.requests = []
        self.config = SimpleNamespace(max_tool_result_chars=4000)

    async def chat(self, messages=None, tools=None, images=None):
        self.requests.append(list(messages))
        return {"message": {"content": self.answer}}

    async def chat_stream_events(self, **kwargs):
        yield {"type": "content", "text": self.answer}
        yield {"type": "done", "tool_calls": []}


def search_tools(calls, result="Google results for 'test':\n\n1. Example\n   https://example.com\n   Example snippet"):
    registry = ToolRegistry()

    @registry.tool("browser_search", "search", {"type": "object", "properties": {
        "query": {"type": "string"}}, "required": ["query"]})
    async def search(query):
        calls.append(query)
        return result

    return registry


async def test_explicit_research_executes_search_before_model():
    calls, llm = [], LLM()
    agent = ReActAgent(llm, ToolRegistry(), "sys", browser_tools=search_tools(calls))
    prompt = "Can you look into the latest vulnerability reports online?"
    result = await agent.run(prompt)
    assert "Example snippet" in result and "https://example.com" in result
    assert calls == [prompt]
    assert not llm.requests  # raw search evidence does not rely on model invention


async def test_affirmative_after_specific_search_offer_runs_original_request():
    calls, llm = [], LLM()
    agent = ReActAgent(llm, ToolRegistry(), "sys", browser_tools=search_tools(calls))
    history = [{"role": "user", "content": "Look into latest cyber reports online"},
               {"role": "assistant", "content": "Would you like me to search for those?"}]
    await agent.run("Yes.", conversation_history=history)
    assert calls == [history[0]["content"]]


async def test_affirmative_after_the_observed_offer_runs_original_request():
    calls, llm = [], LLM()
    agent = ReActAgent(llm, ToolRegistry(), "sys", browser_tools=search_tools(calls))
    history = [{"role": "user", "content": "Can you look into the latest vulnerability reports online?"},
               {"role": "assistant", "content": "I can search for those. Would you like me to do that now?"}]
    await agent.run("Yes.", conversation_history=history)
    assert calls == [history[0]["content"]]


async def test_yes_after_a_completed_result_does_not_repeat_search():
    calls, llm = [], LLM("Okay.")
    agent = ReActAgent(llm, ToolRegistry(), "sys", browser_tools=search_tools(calls))
    history = [{"role": "user", "content": "Search the web for cyber news"},
               {"role": "assistant", "content": "Here are the search results."}]
    assert await agent.run("Yes.", conversation_history=history) == "Okay."
    assert not calls


async def test_search_failure_is_not_summarized_as_success():
    calls, llm = [], LLM()
    agent = ReActAgent(llm, ToolRegistry(), "sys",
                       browser_tools=search_tools(calls, "Search failed: no network"))
    result = await agent.run("Search the web for current news")
    assert "couldn't retrieve" in result
    assert calls and not llm.requests


async def test_search_returns_at_most_three_unmodified_results():
    calls, llm = [], LLM()
    blocks = [f"{i}. Title {i}\n   https://example.com/{i}\n   Snippet {i}" for i in range(1, 5)]
    agent = ReActAgent(llm, ToolRegistry(), "sys", browser_tools=search_tools(
        calls, "Google results for 'test':\n\n" + "\n\n".join(blocks)))
    result = await agent.run("Search the web for current news")
    assert all(block in result for block in blocks[:3])
    assert blocks[3] not in result
    assert "snippets, not full articles" in result
    assert not llm.requests


async def test_false_progress_never_streams_to_user():
    llm = LLM("I'm still searching the web now.")
    agent = ReActAgent(llm, ToolRegistry(), "sys")
    result = "".join([token async for token in agent.run_stream("Yes.")])
    assert "haven't started" in result
    assert "still searching" not in result


async def test_regular_looking_forward_is_not_treated_as_search():
    llm = LLM("I'm looking forward to the weekend.")
    agent = ReActAgent(llm, ToolRegistry(), "sys")
    result = "".join([token async for token in agent.run_stream("How is your weekend?")])
    assert result == "I'm looking forward to the weekend."


@pytest.mark.parametrize("text", ["...", ". . . . .", "... . ... ...", "!? —"])
def test_punctuation_only_is_noise(text):
    assert is_garbage(text)


@pytest.mark.parametrize("text", ["Don't search the web", "Do not look into this online",
                                  "Can you search the web?"])
def test_no_automatic_search_for_refusal_or_capability_question(text):
    assert not is_web_research(text)


async def test_voice_search_confirm_cancel_waits_for_cleanup(monkeypatch):
    app = Orchestrator(JarvisConfig())
    app.tts = SimpleNamespace(speak=AsyncMock(), abort=MagicMock())
    app._record_and_transcribe = AsyncMock(return_value=_Heard("yes", 0, 0))
    monkeypatch.setattr("jarvis.voice.implementations.wait_for_speech", AsyncMock(return_value=True))
    stopped = asyncio.Event()

    async def search():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    assert await app._monitor_voice_search(search()) == CANCELLED_MESSAGE
    assert stopped.is_set()
    app.tts.speak.assert_awaited_once()


async def test_voice_search_declined_continues_without_cancel(monkeypatch):
    app = Orchestrator(JarvisConfig())
    app.tts = SimpleNamespace(speak=AsyncMock(), abort=MagicMock())
    app._record_and_transcribe = AsyncMock(return_value=_Heard("no", 0, 0))
    monkeypatch.setattr("jarvis.voice.implementations.wait_for_speech", AsyncMock(side_effect=[True, False]))

    async def search():
        await asyncio.sleep(0.01)
        return "found"

    assert await app._monitor_voice_search(search()) == "found"


async def test_microphone_monitor_failure_does_not_cancel_search(monkeypatch):
    app = Orchestrator(JarvisConfig())
    monkeypatch.setattr("jarvis.voice.implementations.wait_for_speech",
                        AsyncMock(side_effect=OSError("microphone unavailable")))

    async def search():
        await asyncio.sleep(0.01)
        return "found"

    assert await app._monitor_voice_search(search()) == "found"
    assert "microphone unavailable" in app._faults["search interruption monitor"]


async def test_finished_search_is_discarded_if_user_confirms_cancel(monkeypatch):
    app = Orchestrator(JarvisConfig())
    app.tts = SimpleNamespace(speak=AsyncMock(), abort=MagicMock())
    gate = asyncio.Event()
    finished = asyncio.Event()

    async def search():
        await gate.wait()
        finished.set()
        return "result that must be discarded"

    async def confirm():
        gate.set()
        await finished.wait()
        return _Heard("yes", 0, 0)

    app._record_and_transcribe = AsyncMock(side_effect=confirm)
    monkeypatch.setattr("jarvis.voice.implementations.wait_for_speech", AsyncMock(return_value=True))
    assert await app._monitor_voice_search(search()) == CANCELLED_MESSAGE


async def test_cancelling_outer_voice_turn_stops_search(monkeypatch):
    app = Orchestrator(JarvisConfig())
    stopped = asyncio.Event()

    async def monitor(*args, **kwargs):
        await asyncio.Event().wait()

    async def search():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr("jarvis.voice.implementations.wait_for_speech", monitor)
    task = asyncio.create_task(app._monitor_voice_search(search()))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
