"""Unit tests for the ReAct agent loop with a mocked LLM.

Covers: plain answers, native tool calling, text-parsing fallback,
action-claim retry, duplicate tool-call dedup, empty-response retry,
native->fallback downgrade on HTTP 400, and streaming markup holdback.
"""

import httpx

from jarvis.agent.react_agent import ReActAgent
from jarvis.agent.tool_registry import ToolRegistry


class FakeLLM:
    """Queue-based fake LLMClient.

    For chat(): each queued item is a message dict (returned as
    {"message": item}) or an Exception to raise.
    For chat_stream_events(): each queued item is a list of event dicts.
    """

    def __init__(self, responses, provider="ollama"):
        self.provider = provider
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, messages=None, tools=None, images=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"message": item}

    def chat_stream_events(self, messages=None, tools=None, images=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools})
        events = self._responses.pop(0)

        async def _gen():
            for event in events:
                yield event

        return _gen()


def _http_400_tool_error() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    response = httpx.Response(
        400, text="model does not support tools", request=request,
    )
    return httpx.HTTPStatusError("bad request", request=request, response=response)


def _registry_with_datetime(counter: dict) -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool(name="get_datetime", description="Get the current date and time",
                   parameters={"type": "object", "properties": {}})
    async def get_datetime():
        counter["n"] = counter.get("n", 0) + 1
        return "Monday 5pm"

    return registry


def _registry_with_open_app(counter: dict) -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool(
        name="open_application", description="Launch an application",
        parameters={
            "type": "object",
            "properties": {"app_name": {"type": "string"}},
            "required": ["app_name"],
        },
    )
    async def open_application(app_name: str):
        counter.setdefault("apps", []).append(app_name)
        return f"Launched {app_name}"

    return registry


class TestPlainAnswers:
    async def test_plain_final_answer(self):
        llm = FakeLLM([{"content": "Hello there."}])
        agent = ReActAgent(llm=llm, tools=ToolRegistry(), system_prompt="sys")
        assert await agent.run("hi") == "Hello there."
        assert len(llm.calls) == 1

    async def test_empty_response_retried(self):
        llm = FakeLLM([{"content": ""}, {"content": "Hi"}])
        agent = ReActAgent(llm=llm, tools=ToolRegistry(), system_prompt="sys")
        assert await agent.run("hi") == "Hi"
        assert len(llm.calls) == 2

    async def test_persistent_empty_gives_up(self):
        llm = FakeLLM([{"content": ""}] * 5)
        agent = ReActAgent(llm=llm, tools=ToolRegistry(), system_prompt="sys")
        assert await agent.run("hi") == "I couldn't generate a response."

    async def test_think_blocks_stripped_from_answer(self):
        llm = FakeLLM([{"content": "<think>hmm let me think</think>The answer is 4."}])
        agent = ReActAgent(llm=llm, tools=ToolRegistry(), system_prompt="sys")
        assert await agent.run("what is 2+2") == "The answer is 4."


class TestNativeToolCalling:
    async def test_tool_call_then_final_answer(self):
        counter: dict = {}
        llm = FakeLLM([
            {"content": "", "tool_calls": [{"function": {"name": "get_datetime", "arguments": {}}}]},
            {"content": "It is Monday 5pm."},
        ])
        agent = ReActAgent(llm=llm, tools=_registry_with_datetime(counter), system_prompt="sys")

        assert await agent.run("what time is it") == "It is Monday 5pm."
        assert counter["n"] == 1
        # Tool result fed back with role "tool" on the second LLM call
        second_msgs = llm.calls[1]["messages"]
        assert any(m.get("role") == "tool" and "Monday 5pm" in m.get("content", "")
                   for m in second_msgs)

    async def test_duplicate_tool_call_deduplicated(self):
        counter: dict = {}
        call = {"function": {"name": "get_datetime", "arguments": {}}}
        llm = FakeLLM([
            {"content": "", "tool_calls": [call]},
            {"content": "", "tool_calls": [call]},
            {"content": "All set."},
        ])
        agent = ReActAgent(llm=llm, tools=_registry_with_datetime(counter), system_prompt="sys")

        assert await agent.run("time?") == "All set."
        assert counter["n"] == 1
        third_msgs = llm.calls[2]["messages"]
        assert any("already called" in m.get("content", "") for m in third_msgs)

    async def test_unknown_tool_reported_back(self):
        llm = FakeLLM([
            {"content": "", "tool_calls": [{"function": {"name": "fly_to_moon", "arguments": {}}}]},
            {"content": "I cannot do that."},
        ])
        agent = ReActAgent(llm=llm, tools=ToolRegistry(), system_prompt="sys")
        assert await agent.run("fly") == "I cannot do that."
        second_msgs = llm.calls[1]["messages"]
        assert any("Unknown tool" in m.get("content", "") for m in second_msgs)

    async def test_tool_exception_does_not_crash_loop(self):
        registry = ToolRegistry()

        @registry.tool(name="boom", description="x",
                       parameters={"type": "object", "properties": {}})
        async def boom():
            raise RuntimeError("kaboom")

        llm = FakeLLM([
            {"content": "", "tool_calls": [{"function": {"name": "boom", "arguments": {}}}]},
            {"content": "That failed, sorry."},
        ])
        agent = ReActAgent(llm=llm, tools=registry, system_prompt="sys")
        assert await agent.run("go") == "That failed, sorry."
        second_msgs = llm.calls[1]["messages"]
        assert any("kaboom" in m.get("content", "") for m in second_msgs)


class TestFallbackTextParsing:
    async def test_lfm_tool_call_executed(self):
        counter: dict = {}
        llm = FakeLLM(
            [
                {"content": "<|tool_call_start|>[get_datetime()]<|tool_call_end|>"},
                {"content": "It is Monday 5pm."},
            ],
            provider="huggingface",
        )
        agent = ReActAgent(llm=llm, tools=_registry_with_datetime(counter), system_prompt="sys")

        assert await agent.run("time?") == "It is Monday 5pm."
        assert counter["n"] == 1
        # Fallback path: tool result delivered as a user message
        second_msgs = llm.calls[1]["messages"]
        assert any(m.get("role") == "user" and "Monday 5pm" in m.get("content", "")
                   for m in second_msgs)

    async def test_fallback_prompt_contains_format_instructions(self):
        llm = FakeLLM([{"content": "ok"}], provider="huggingface")
        agent = ReActAgent(
            llm=llm, tools=_registry_with_datetime({}), system_prompt="sys",
        )
        await agent.run("hi")
        system_msg = llm.calls[0]["messages"][0]["content"]
        assert "tool_call_start" in system_msg
        assert "get_datetime" in system_msg

    async def test_positional_args_mapped_to_declared_params(self):
        counter: dict = {}
        llm = FakeLLM(
            [
                {"content": '<|tool_call_start|>[open_application("notepad")]<|tool_call_end|>'},
                {"content": "Opened notepad."},
            ],
            provider="huggingface",
        )
        agent = ReActAgent(llm=llm, tools=_registry_with_open_app(counter), system_prompt="sys")
        await agent.run("open notepad")
        assert counter["apps"] == ["notepad"]


class TestActionClaimRetry:
    async def test_claiming_action_without_tool_forces_retry(self):
        llm = FakeLLM([
            {"content": "I've opened Notepad for you."},
            {"content": "Sorry, I can't actually do that."},
        ])
        agent = ReActAgent(llm=llm, tools=ToolRegistry(), system_prompt="sys")

        result = await agent.run("open notepad")
        assert result == "Sorry, I can't actually do that."
        assert len(llm.calls) == 2
        nudges = [m for m in llm.calls[1]["messages"] if "did NOT call any tool" in m.get("content", "")]
        assert len(nudges) == 1

    async def test_claim_retried_only_once(self):
        llm = FakeLLM([
            {"content": "I've opened Notepad for you."},
            {"content": "I opened it, promise."},
        ])
        agent = ReActAgent(llm=llm, tools=ToolRegistry(), system_prompt="sys")
        result = await agent.run("open notepad")
        # Second claim is accepted as final (no infinite nudging)
        assert result == "I opened it, promise."
        assert len(llm.calls) == 2


class TestNativeFallbackDowngrade:
    async def test_http_400_switches_to_text_parsing(self):
        err = _http_400_tool_error()
        llm = FakeLLM([
            err, err, err,  # _chat_with_retry attempts 1 + 2 retries
            {"content": "<|tool_call_start|>[get_datetime()]<|tool_call_end|>"},
            {"content": "It is Monday 5pm."},
        ])
        counter: dict = {}
        agent = ReActAgent(llm=llm, tools=_registry_with_datetime(counter), system_prompt="sys")

        assert await agent.run("time?") == "It is Monday 5pm."
        assert agent._native_tools_ok is False
        assert counter["n"] == 1


class TestStreaming:
    async def test_streamed_tokens_collected(self):
        llm = FakeLLM([
            [
                {"type": "content", "text": "Hello, "},
                {"type": "content", "text": "world!"},
                {"type": "done", "content": "Hello, world!", "tool_calls": []},
            ],
        ])
        agent = ReActAgent(llm=llm, tools=ToolRegistry(), system_prompt="sys")
        parts = [t async for t in agent.run_stream("hi")]
        assert "".join(parts) == "Hello, world!"

    async def test_tool_markup_never_streamed_to_user(self):
        counter: dict = {}
        markup = "<|tool_call_start|>[get_datetime()]<|tool_call_end|>"
        llm = FakeLLM(
            [
                [
                    {"type": "content", "text": "Sure. "},
                    {"type": "content", "text": markup},
                    {"type": "done", "content": f"Sure. {markup}", "tool_calls": []},
                ],
                [
                    {"type": "content", "text": "It's 5pm."},
                    {"type": "done", "content": "It's 5pm.", "tool_calls": []},
                ],
            ],
            provider="huggingface",
        )
        agent = ReActAgent(llm=llm, tools=_registry_with_datetime(counter), system_prompt="sys")

        parts = [t async for t in agent.run_stream("time?")]
        streamed = "".join(parts)
        assert "tool_call_start" not in streamed
        assert "get_datetime" not in streamed
        assert streamed == "Sure. It's 5pm."
        assert counter["n"] == 1
