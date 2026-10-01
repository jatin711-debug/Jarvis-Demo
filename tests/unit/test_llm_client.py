"""The Ollama client against a mock transport: errors, truncation, and cleanup."""

import json

import httpx
import pytest

from jarvis.agent.llm_client import LLMClient
from jarvis.agent.react_agent import ReActAgent
from jarvis.utils.config import LLMConfig


def ndjson(*objects) -> bytes:
    return b"".join(json.dumps(o).encode() + b"\n" for o in objects)


class Chunks(httpx.AsyncByteStream):
    """A genuinely streamed body: unread until aread(), and records being closed."""

    def __init__(self, *parts: bytes) -> None:
        self.parts = parts
        self.closed = False

    async def __aiter__(self):
        for part in self.parts:
            yield part

    async def aclose(self) -> None:
        self.closed = True


def make_client(handler, **config) -> LLMClient:
    client = LLMClient(LLMConfig(model="qwen3.5:2b", **config))
    client._client = httpx.AsyncClient(base_url="http://ollama.test",
                                       transport=httpx.MockTransport(handler))
    return client


async def collect(client: LLMClient, **kwargs):
    return [event async for event in client.chat_stream_events(
        [{"role": "user", "content": "hi"}], **kwargs)]


DONE = {"message": {"content": ""}, "done": True}


class TestStreamedReply:
    async def test_content_and_tool_calls_are_collected(self):
        call = {"function": {"name": "get_datetime", "arguments": {}}}
        body = ndjson(
            {"message": {"content": "Hel"}},
            {"message": {"content": "lo"}},
            {"message": {"content": "", "tool_calls": [call]}},
            DONE,
        )
        client = make_client(lambda request: httpx.Response(200, content=body))
        events = await collect(client)
        assert [e["text"] for e in events if e["type"] == "content"] == ["Hel", "lo"]
        assert events[-1] == {"type": "done", "content": "Hello", "tool_calls": [call]}

    async def test_a_line_that_is_not_json_is_skipped(self):
        body = (ndjson({"message": {"content": "a"}}) + b"<html>proxy noise</html>\n"
                + ndjson({"message": {"content": "b"}}, DONE))
        client = make_client(lambda request: httpx.Response(200, content=body))
        assert (await collect(client))[-1]["content"] == "ab"

    async def test_an_error_line_mid_stream_is_raised_not_dropped(self):
        body = ndjson({"message": {"content": "partial"}}, {"error": "model runner has unexpectedly stopped"})
        client = make_client(lambda request: httpx.Response(200, content=body))
        with pytest.raises(RuntimeError, match="model runner has unexpectedly stopped"):
            await collect(client)

    async def test_a_stream_that_ends_before_done_is_not_a_complete_reply(self):
        body = ndjson({"message": {"content": "half a sent"}})
        client = make_client(lambda request: httpx.Response(200, content=body))
        with pytest.raises(httpx.RemoteProtocolError, match="before the reply finished"):
            await collect(client)


class TestRejectedRequest:
    async def test_the_error_body_is_readable_so_the_tool_fallback_can_fire(self):
        message = b'{"error":"registry.ollama.ai/library/tiny does not support tools"}'
        client = make_client(lambda request: httpx.Response(400, stream=Chunks(message)))
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await collect(client, tools=[{"type": "function", "function": {"name": "x"}}])
        assert "does not support tools" in caught.value.response.text
        assert ReActAgent._native_tools_rejected(caught.value) is True

    async def test_other_client_errors_are_not_mistaken_for_missing_tool_support(self):
        client = make_client(lambda request: httpx.Response(
            400, stream=Chunks(b'{"error":"context length exceeded"}')))
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await collect(client)
        assert ReActAgent._native_tools_rejected(caught.value) is False

    async def test_server_errors_propagate(self):
        client = make_client(lambda request: httpx.Response(500, stream=Chunks(b"boom")))
        with pytest.raises(httpx.HTTPStatusError):
            await collect(client)


async def test_closing_the_event_stream_early_closes_the_http_response():
    body = Chunks(ndjson({"message": {"content": "one"}}), ndjson({"message": {"content": "two"}}),
                  ndjson(DONE))
    client = make_client(lambda request: httpx.Response(200, stream=body))
    events = client.chat_stream_events([{"role": "user", "content": "hi"}])
    first = await anext(events)
    assert first["text"] == "one"
    assert not body.closed
    await events.aclose()
    assert body.closed, "an abandoned stream kept the connection open"


class TestRequestBody:
    def sent_body(self, client, **kwargs):
        captured = {}

        def handler(request):
            captured.update(json.loads(request.content))
            return httpx.Response(200, content=ndjson(DONE))

        client._client = httpx.AsyncClient(base_url="http://ollama.test",
                                           transport=httpx.MockTransport(handler))
        return captured

    async def test_images_ride_on_the_last_user_message_only(self):
        client = make_client(lambda request: httpx.Response(200, content=ndjson(DONE)))
        captured = self.sent_body(client)
        messages = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "look"}]
        [e async for e in client.chat_stream_events(messages, images=["b64"])]
        sent = captured["messages"]
        assert "images" not in sent[0] and sent[2]["images"] == ["b64"]
        assert "images" not in messages[2], "the caller's messages must not be mutated"

    async def test_streamed_and_plain_requests_share_one_body_builder(self):
        client = make_client(lambda request: httpx.Response(200, content=b"{}"))
        tools = [{"type": "function", "function": {"name": "x"}}]
        streamed = client._ollama_payload([], tools, 0.2, 99, None, stream=True)
        plain = client._ollama_payload([], tools, 0.2, 99, None, stream=False)
        assert streamed.pop("stream") is True and plain.pop("stream") is False
        assert streamed == plain
        assert streamed["think"] is False, "qwen3.5 must not spend tokens on reasoning"
        assert streamed["options"]["temperature"] == 0.2 and streamed["options"]["num_predict"] == 99
        assert streamed["tools"] == tools


def test_text_only_provider_reports_no_image_support():
    assert LLMClient(LLMConfig(provider="huggingface", model="x")).supports_images is False
    assert LLMClient(LLMConfig()).supports_images is True
