"""Stages of the ReAct loop: history budget, vision, iteration limit, logging, error hints."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jarvis.agent.react_agent import MAX_ITERATIONS, ReActAgent
from jarvis.agent.routing import Route, Surface
from jarvis.agent.tool_registry import ToolRegistry


class ScriptedLLM:
    """Ollama-style fake: returns queued messages, records every request."""

    provider = "ollama"

    def __init__(self, replies, **config):
        self.replies = list(replies)
        self.requests = []
        self.config = SimpleNamespace(**config) if config else None

    async def chat(self, messages=None, tools=None, images=None):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools, "images": images})
        return {"message": self.replies.pop(0) if self.replies else {"content": "done"}}


def call(name, **args):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def make_tools(record=None):
    record = record if record is not None else []
    tools = ToolRegistry()

    @tools.tool("get_datetime", "Get date and time", {"type": "object", "properties": {}})
    async def get_datetime():
        record.append("clock")
        return "5pm"

    @tools.tool("read_file", "Read file", {"type": "object", "properties": {
        "file_path": {"type": "string"}}, "required": ["file_path"]})
    async def read_file(file_path):
        record.append(file_path)
        return "contents"

    return tools


class TestHistoryBudget:
    def test_keeps_whole_recent_turns_within_the_budget(self):
        agent = ReActAgent(ScriptedLLM([], max_history_chars=50), ToolRegistry(), "sys")
        history = []
        for i in range(6):
            history += [{"role": "user", "content": f"question number {i}"},
                        {"role": "assistant", "content": f"answer number {i}"}]
        kept = agent._recent_history(history)
        assert len(kept) % 2 == 0 and kept[0]["role"] == "user"
        assert kept[-1]["content"] == "answer number 5"
        assert sum(len(m["content"]) for m in kept) <= 50
        assert len(kept) < len(history)

    def test_no_history_is_fine(self):
        agent = ReActAgent(ScriptedLLM([]), ToolRegistry(), "sys")
        assert agent._recent_history(None) == []
        assert agent._recent_history([]) == []

    async def test_history_is_sent_between_system_prompt_and_question(self):
        llm = ScriptedLLM([{"content": "Yes."}])
        agent = ReActAgent(llm, make_tools(), "sys")
        history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
        await agent.run("What time is it?", conversation_history=history)
        roles = [m["role"] for m in llm.requests[0]["messages"]]
        assert roles == ["system", "user", "assistant", "user"]


class TestVision:
    async def test_screenshot_attached_only_on_the_first_model_call(self):
        llm = ScriptedLLM([call("get_datetime"), {"content": "It is 5pm."}])
        agent = ReActAgent(llm, make_tools(), "sys")
        agent._capture_screenshot = AsyncMock(return_value=["b64"])
        await agent.run("take a screenshot and tell me the time")
        assert llm.requests[0]["images"] == ["b64"]
        assert llm.requests[1]["images"] is None
        assert "VISION MODE" in llm.requests[0]["messages"][0]["content"]

    async def test_no_screenshot_for_ordinary_requests(self):
        llm = ScriptedLLM([{"content": "5pm."}])
        agent = ReActAgent(llm, make_tools(), "sys")
        agent._capture_screenshot = AsyncMock(return_value=["b64"])
        await agent.run("What time is it?")
        agent._capture_screenshot.assert_not_called()

    async def test_a_failed_screenshot_is_disclosed_to_the_model(self):
        llm = ScriptedLLM([{"content": "I could not capture the screen."}])
        agent = ReActAgent(llm, make_tools(), "sys")
        agent._capture_screenshot = AsyncMock(return_value=None)
        await agent.run("take a screenshot and tell me what you see")
        sent = llm.requests[0]
        assert sent["images"] is None
        assert "could not be captured" in sent["messages"][-1]["content"]
        assert "VISION MODE" not in sent["messages"][0]["content"]

    async def test_text_only_provider_never_gets_a_screenshot(self, monkeypatch):
        llm = ScriptedLLM([{"content": "I cannot see the screen."}])
        llm.supports_images = False
        agent = ReActAgent(llm, make_tools(), "sys")
        grabbed = []
        monkeypatch.setattr(ReActAgent, "_grab_screenshot", staticmethod(lambda: grabbed.append(1)))
        await agent.run("take a screenshot and tell me what you see")
        assert not grabbed
        assert llm.requests[0]["images"] is None

    async def test_screenshot_is_taken_off_the_event_loop(self, monkeypatch):
        import asyncio
        import threading
        import time

        main_thread = threading.get_ident()
        seen = {}

        def slow_grab():
            seen["thread"] = threading.get_ident()
            time.sleep(0.2)
            return "b64"

        monkeypatch.setattr(ReActAgent, "_grab_screenshot", staticmethod(slow_grab))
        agent = ReActAgent(ScriptedLLM([]), make_tools(), "sys")
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        background = asyncio.create_task(ticker())
        try:
            assert await agent._capture_screenshot() == ["b64"]
        finally:
            background.cancel()
        assert seen["thread"] != main_thread
        assert ticks >= 5


class TestIterationLimit:
    async def test_endless_tool_calls_stop_with_an_honest_message(self):
        record = []
        # Different arguments each time, so repeated-call suppression does not end the loop.
        llm = ScriptedLLM([call("read_file", file_path=f"file{i}.txt") for i in range(MAX_ITERATIONS + 2)])
        agent = ReActAgent(llm, make_tools(record), "sys")
        answer = await agent.run("read the file list")
        assert "step limit" in answer
        assert len(record) == MAX_ITERATIONS


class TestToolLogging:
    async def test_call_and_outcome_are_logged_with_secrets_redacted(self, caplog):
        tools = ToolRegistry()

        @tools.tool("get_datetime", "Get date and time", {"type": "object", "properties": {
            "password": {"type": "string"}}})
        async def get_datetime(password=""):
            return "5pm"

        llm = ScriptedLLM([call("get_datetime", password="hunter2"), {"content": "It is 5pm."}])
        agent = ReActAgent(llm, tools, "sys")
        with caplog.at_level(logging.INFO, logger="jarvis.agent.react_agent"):
            await agent.run("What time is it?")
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "route surface=system groups=clock" in text
        assert "tool call name=get_datetime args={password=<redacted>}" in text
        assert "hunter2" not in text
        assert "tool ok name=get_datetime" in text

    async def test_failures_are_logged_as_warnings_with_detail(self, caplog):
        tools = ToolRegistry()

        @tools.tool("get_datetime", "Get date and time", {"type": "object", "properties": {}})
        async def get_datetime():
            raise OSError("clock unavailable")

        llm = ScriptedLLM([call("get_datetime"), {"content": "I could not read the clock."}])
        agent = ReActAgent(llm, tools, "sys")
        with caplog.at_level(logging.INFO, logger="jarvis.agent.react_agent"):
            await agent.run("What time is it?")
        warning = next(r for r in caplog.records if r.getMessage().startswith("tool error"))
        assert warning.levelno == logging.WARNING
        assert "clock unavailable" in warning.getMessage()


class TestErrorHints:
    async def test_desktop_hint_only_appears_on_the_desktop_surface(self):
        llm = ScriptedLLM([call("get_datetime", bogus=1), {"content": "Sorry."}])
        agent = ReActAgent(llm, make_tools(), "sys")
        await agent.run("What time is it?")
        fed_back = str(llm.requests[1]["messages"])
        assert "desktop_windows" not in fed_back

    async def test_tool_outside_the_selected_surface_is_rejected_with_a_reason(self):
        llm = ScriptedLLM([call("read_file", file_path="x.txt"), {"content": "Cannot."}])
        record = []
        agent = ReActAgent(llm, make_tools(record), "sys")
        await agent.run("What time is it?")
        assert record == []
        assert "outside this request" in str(llm.requests[1]["messages"])


class TestToolSurfaces:
    def test_each_surface_offers_only_its_own_tools(self):
        system, browser, desktop = make_tools(), ToolRegistry(), ToolRegistry()

        @browser.tool("browser_read_page", "Read page", {"type": "object", "properties": {}})
        async def browser_read_page():
            return "page"

        @desktop.tool("desktop_windows", "List windows", {"type": "object", "properties": {}})
        async def desktop_windows():
            return "windows"

        agent = ReActAgent(SimpleNamespace(provider="ollama"), system, "sys",
                           browser_tools=browser, desktop_tools=desktop)

        def names(route):
            return sorted(t.name for t in agent._tools_for(route))

        assert names(Route(Surface.BROWSER)) == ["browser_read_page"]
        assert names(Route(Surface.DESKTOP)) == ["desktop_windows"]
        assert names(Route(Surface.SYSTEM, frozenset({"get_datetime"}))) == ["get_datetime"]
        assert names(Route(Surface.CHAT)) == []
        assert agent._resolve_tool("browser_read_page").name == "browser_read_page"
        assert agent._resolve_tool("nope") is None

    async def test_execute_tool_reaches_every_registry(self):
        system, browser = make_tools(), ToolRegistry()

        @browser.tool("browser_read_page", "Read page", {"type": "object", "properties": {}})
        async def browser_read_page():
            return "page"

        agent = ReActAgent(SimpleNamespace(provider="ollama"), system, "sys", browser_tools=browser)
        assert await agent._execute_tool("browser_read_page") == "page"
        with pytest.raises(ValueError, match="Unknown tool"):
            await agent._execute_tool("missing")


class StreamingLLM:
    """Stream fake: yields a spoken prefix, then a tool call, like a real Ollama reply."""

    provider = "ollama"

    def __init__(self):
        self.events_sent = 0
        self.stream_closed = False

    async def chat_stream_events(self, **kwargs):
        try:
            self.events_sent += 1
            yield {"type": "content", "text": "Sure. "}
            self.events_sent += 1
            yield {"type": "done", "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"file_path": "secret.txt"}}}]}
        finally:
            self.stream_closed = True


class TestStreamInterruption:
    async def test_closing_the_stream_before_the_tool_call_runs_no_tool(self):
        from contextlib import aclosing

        record = []
        llm = StreamingLLM()
        agent = ReActAgent(llm, make_tools(record), "sys")
        async with aclosing(agent.run_stream("read the file list")) as stream:
            first = await anext(stream)
        assert first == "Sure. "
        assert record == []  # the requested read never happened
        assert llm.stream_closed  # the model connection was released

    async def test_letting_the_stream_finish_does_run_the_tool(self):
        record = []
        agent = ReActAgent(StreamingLLM(), make_tools(record), "sys")
        [_ async for _ in agent.run_stream("read the file list")]
        assert record == ["secret.txt"]  # identical repeat calls are suppressed
