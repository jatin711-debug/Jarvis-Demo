import asyncio
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from jarvis.agent.llm_client import LLMClient
from jarvis.agent.react_agent import ReActAgent
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.utils.config import LLMConfig


@pytest.fixture
def client(monkeypatch):
    class Streamer:
        def __init__(self, tokenizer, **kwargs):
            assert kwargs == {"skip_prompt": True, "skip_special_tokens": False}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        TextStreamer=Streamer, StoppingCriteria=object, StoppingCriteriaList=list))

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, *args, **kwargs):
            assert kwargs["add_special_tokens"] is False
            return Inputs()

    instance = LLMClient(LLMConfig(provider="huggingface"))
    instance._hf_tokenizer = Tokenizer()
    return instance


async def test_preserves_tool_markers_removes_boundary_and_honors_zero_temperature(client):
    def generate(**kwargs):
        assert kwargs["do_sample"] is False
        assert "temperature" not in kwargs
        kwargs["streamer"].on_finalized_text('<|tool_call_start|>[test()]<|tool_call_end|><|im_end|>')

    client._hf_model = SimpleNamespace(device="cpu", generate=generate)
    result = await client._chat_huggingface([], temperature=0)
    assert result["message"]["content"] == '<|tool_call_start|>[test()]<|tool_call_end|>'


async def test_generation_failure_is_propagated(client):
    def generate(**kwargs):
        raise RuntimeError("fixture failure")

    client._hf_model = SimpleNamespace(device="cpu", generate=generate)
    with pytest.raises(RuntimeError, match="fixture failure"):
        await client._chat_huggingface([])


async def test_stream_emits_before_generation_finishes_and_close_stops_worker(client):
    finished = threading.Event()

    def generate(**kwargs):
        kwargs["streamer"].on_finalized_text("hello ")
        stop = kwargs["stopping_criteria"][0]
        while not stop(None, None):
            finished.wait(0.001)
        finished.set()

    client._hf_model = SimpleNamespace(device="cpu", generate=generate)
    stream = client.chat_stream_events([])
    first = await asyncio.wait_for(anext(stream), 2)
    assert first == {"type": "content", "text": "hello "}
    assert not finished.is_set()
    await stream.aclose()
    assert finished.is_set()
    assert not client._hf_lock.locked()


async def test_tool_name_argument_does_not_collide_with_dispatcher():
    registry = ToolRegistry()

    @registry.tool("lookup", "Lookup a name", {"type": "object", "properties": {"name": {"type": "string"}}})
    async def lookup(name):
        return name

    agent = ReActAgent(SimpleNamespace(provider="huggingface"), registry, "test")
    assert await agent._execute_tool("lookup", name="demo") == "demo"


def test_lfm_prompt_uses_actual_schemas_not_placeholder_names():
    llm = SimpleNamespace(provider="huggingface", config=LLMConfig(model="LiquidAI/LFM2.5-350M"))
    agent = ReActAgent(llm, ToolRegistry(), "test")
    prompt = agent._build_system_prompt(use_native=False)
    assert "List of tools:" not in prompt
    assert "tool_name" not in prompt
