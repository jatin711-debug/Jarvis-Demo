"""The llama-server provider against a mock transport: startup, SSE parsing, tool calls, history shape."""

import json

import httpx
import pytest

from jarvis.agent.llm_client import LLMClient
from jarvis.agent.react_agent import ReActAgent
from jarvis.utils.config import LLMConfig


def sse(*chunks, done=True) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def text(token, finish=None):
    return {"choices": [{"delta": {"content": token}, "finish_reason": finish}]}


def call_fragment(index, *, name=None, arguments=None, call_id=None):
    function = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    fragment = {"index": index, "function": function}
    if call_id:
        fragment.update(id=call_id, type="function")
    return {"choices": [{"delta": {"tool_calls": [fragment]}, "finish_reason": None}]}


STOP = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
TOOL_STOP = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}


class Chunks(httpx.AsyncByteStream):
    def __init__(self, *parts: bytes) -> None:
        self.parts = parts
        self.closed = False

    async def __aiter__(self):
        for part in self.parts:
            yield part

    async def aclose(self) -> None:
        self.closed = True


def make_client(handler, **config) -> LLMClient:
    client = LLMClient(LLMConfig(provider="llamacpp", model="qwen3.5-4b", **config))
    client._client = httpx.AsyncClient(base_url="http://llama.test",
                                       transport=httpx.MockTransport(handler))
    return client


async def collect(client: LLMClient, **kwargs):
    return [event async for event in client.chat_stream_events(
        [{"role": "user", "content": "hi"}], **kwargs)]


class TestStreamedReply:
    async def test_content_is_streamed_and_collected(self):
        body = sse(text("Hel"), text("lo"), STOP)
        client = make_client(lambda request: httpx.Response(200, content=body))
        events = await collect(client)
        assert [e["text"] for e in events if e["type"] == "content"] == ["Hel", "lo"]
        assert events[-1] == {"type": "done", "content": "Hello", "tool_calls": []}

    async def test_comments_and_blank_lines_are_ignored(self):
        body = b": keep-alive\n\n" + sse(text("a"), STOP)
        client = make_client(lambda request: httpx.Response(200, content=body))
        assert (await collect(client))[-1]["content"] == "a"

    async def test_reasoning_text_is_never_spoken(self):
        thinking = {"choices": [{"delta": {"reasoning_content": "hmm"}, "finish_reason": None}]}
        body = sse(thinking, text("Hi"), STOP)
        client = make_client(lambda request: httpx.Response(200, content=body))
        assert (await collect(client))[-1]["content"] == "Hi"

    async def test_fragmented_tool_calls_are_assembled_with_object_arguments(self):
        body = sse(
            call_fragment(0, name="set_volume", arguments="", call_id="c1"),
            call_fragment(0, arguments='{"lev'),
            call_fragment(0, arguments='el": 30}'),
            call_fragment(1, name="get_datetime", arguments="{}", call_id="c2"),
            TOOL_STOP,
        )
        client = make_client(lambda request: httpx.Response(200, content=body))
        done = (await collect(client))[-1]
        assert done["tool_calls"] == [
            {"function": {"name": "set_volume", "arguments": {"level": 30}}},
            {"function": {"name": "get_datetime", "arguments": {}}},
        ]

    async def test_broken_argument_json_becomes_empty_arguments(self):
        body = sse(call_fragment(0, name="x", arguments='{"a": '), TOOL_STOP)
        client = make_client(lambda request: httpx.Response(200, content=body))
        assert (await collect(client))[-1]["tool_calls"] == [
            {"function": {"name": "x", "arguments": {}}}]

    async def test_an_error_chunk_is_raised_not_dropped(self):
        body = sse(text("partial"), {"error": {"message": "slot unavailable"}}, done=False)
        client = make_client(lambda request: httpx.Response(200, content=body))
        with pytest.raises(RuntimeError, match="slot unavailable"):
            await collect(client)

    async def test_a_stream_that_ends_without_a_finish_marker_is_not_a_complete_reply(self):
        body = sse(text("half a sent"), done=False)
        client = make_client(lambda request: httpx.Response(200, content=body))
        with pytest.raises(httpx.RemoteProtocolError, match="before the reply finished"):
            await collect(client)

    async def test_plain_chat_assembles_the_message(self):
        body = sse(text("Hi"), call_fragment(0, name="f", arguments="{}"), TOOL_STOP)
        client = make_client(lambda request: httpx.Response(200, content=body))
        response = await client.chat([{"role": "user", "content": "hi"}])
        assert response["message"]["content"] == "Hi"
        assert response["message"]["tool_calls"][0]["function"]["name"] == "f"


class TestRejectedRequest:
    async def test_the_error_body_is_readable_for_the_tool_fallback(self):
        message = b'{"error":{"message":"tools are not supported by this template"}}'
        client = make_client(lambda request: httpx.Response(400, stream=Chunks(message)))
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await collect(client, tools=[{"type": "function", "function": {"name": "x"}}])
        assert ReActAgent._native_tools_rejected(caught.value) is True

    async def test_server_errors_propagate(self):
        client = make_client(lambda request: httpx.Response(500, stream=Chunks(b"boom")))
        with pytest.raises(httpx.HTTPStatusError):
            await collect(client)


async def test_closing_the_event_stream_early_closes_the_http_response():
    body = Chunks(sse(text("one"), done=False), sse(text("two"), STOP))
    client = make_client(lambda request: httpx.Response(200, stream=body))
    events = client.chat_stream_events([{"role": "user", "content": "hi"}])
    assert (await anext(events))["text"] == "one"
    await events.aclose()
    assert body.closed, "an abandoned stream kept the connection open"


class TestRequestBody:
    def payload(self, messages, *, images=None, tools=None, model="qwen3.5-4b"):
        client = LLMClient(LLMConfig(provider="llamacpp", model=model))
        return client._openai_payload(messages, tools, 0.2, 99, images)

    def test_sampling_limits_and_prompt_cache(self):
        payload = self.payload([{"role": "user", "content": "hi"}])
        assert payload["stream"] is True and payload["cache_prompt"] is True
        assert payload["temperature"] == 0.2 and payload["max_tokens"] == 99

    def test_qwen35_thinking_is_switched_off_and_other_models_are_left_alone(self):
        assert self.payload([], model="Qwen3.5-4B")["chat_template_kwargs"] == {"enable_thinking": False}
        assert "chat_template_kwargs" not in self.payload([], model="lfm2.5-1.2b")

    @pytest.mark.parametrize("model,enabled", [
        ("ternary-bonsai-2-9b", False),
        ("Qwen3.5-4B", True),
    ])
    def test_explicit_thinking_setting_overrides_model_name(self, model, enabled):
        client = LLMClient(LLMConfig(provider="llamacpp", model=model, enable_thinking=enabled))
        payload = client._openai_payload([], None, None, None, None)
        assert payload["chat_template_kwargs"] == {"enable_thinking": enabled}

    def test_tools_are_forwarded_only_when_given(self):
        tools = [{"type": "function", "function": {"name": "x"}}]
        assert self.payload([], tools=tools)["tools"] == tools
        assert "tools" not in self.payload([])

    def test_images_ride_on_the_last_user_message_only(self):
        messages = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "look"}]
        sent = self.payload(messages, images=["b64"])["messages"]
        assert sent[0]["content"] == "first"
        assert sent[2]["content"][0] == {"type": "text", "text": "look"}
        assert sent[2]["content"][1]["image_url"]["url"].endswith(",b64")
        assert messages[2]["content"] == "look", "the caller's messages must not be mutated"

    def test_history_with_tool_calls_is_converted_to_openai_shape(self):
        history = [
            {"role": "user", "content": "volume 30"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "set_volume", "arguments": {"level": 30}}},
                            {"function": {"name": "get_datetime", "arguments": {}}}]},
            {"role": "tool", "name": "set_volume", "content": "ok"},
            {"role": "tool", "name": "get_datetime", "content": "noon"},
        ]
        sent = self.payload(history)["messages"]
        calls = sent[1]["tool_calls"]
        assert [c["id"] for c in calls] == ["call_1", "call_2"]
        assert all(c["type"] == "function" for c in calls)
        assert json.loads(calls[0]["function"]["arguments"]) == {"level": 30}
        assert [m["tool_call_id"] for m in sent[2:]] == ["call_1", "call_2"]
        assert history[1]["tool_calls"][0]["function"]["arguments"] == {"level": 30}


class TestStartup:
    def started(self, handler):
        client = LLMClient(LLMConfig(provider="llamacpp", model="m", context_window=4096))
        transport = httpx.MockTransport(handler)
        real = httpx.AsyncClient

        def build(*args, **kwargs):
            kwargs["transport"] = transport
            return real(*args, **kwargs)

        return client, build

    async def test_connects_and_learns_whether_images_are_supported(self, monkeypatch):
        def handler(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(200, json={
                "modalities": {"vision": True},
                "default_generation_settings": {"n_ctx": 8192}})

        client, build = self.started(handler)
        monkeypatch.setattr(httpx, "AsyncClient", build)
        await client.startup()
        assert client.supports_images is True
        assert await client.health_check() is True
        await client.shutdown()

    async def test_text_only_server(self, monkeypatch):
        client, build = self.started(lambda request: httpx.Response(
            200, json={"modalities": {"vision": False}}))
        monkeypatch.setattr(httpx, "AsyncClient", build)
        await client.startup()
        assert client.supports_images is False
        await client.shutdown()

    async def test_waits_while_the_model_is_loading(self, monkeypatch):
        health_calls = []

        def handler(request):
            if request.url.path == "/health":
                health_calls.append(1)
                return httpx.Response(503 if len(health_calls) < 3 else 200, json={})
            return httpx.Response(200, json={})

        async def instant(_seconds):
            return None

        client, build = self.started(handler)
        monkeypatch.setattr(httpx, "AsyncClient", build)
        monkeypatch.setattr("jarvis.agent.llm_client.asyncio.sleep", instant)
        await client.startup()
        assert len(health_calls) == 3
        await client.shutdown()

    async def test_an_unreachable_server_says_how_to_start_it(self, monkeypatch):
        def refuse(request):
            raise httpx.ConnectError("refused")

        client, build = self.started(refuse)
        monkeypatch.setattr(httpx, "AsyncClient", build)
        with pytest.raises(RuntimeError, match="llama-server -m"):
            await client.startup()
        assert client._client is None

    async def test_an_unhealthy_server_is_reported(self, monkeypatch):
        client, build = self.started(lambda request: httpx.Response(500, json={}))
        monkeypatch.setattr(httpx, "AsyncClient", build)
        with pytest.raises(RuntimeError, match="unhealthy"):
            await client.startup()


def test_native_tool_calling_is_assumed_for_llama_server():
    from jarvis.agent.tool_registry import ToolRegistry

    agent = ReActAgent(LLMClient(LLMConfig(provider="llamacpp", model="m")), ToolRegistry(), "sys")
    assert agent._native_tools_ok is True
