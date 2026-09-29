"""Opt-in real LFM inference; downloads the checkpoint on first run.

No desktop actions: the only callable tool is a deterministic test fixture.
"""
import asyncio
import time

from jarvis.agent.llm_client import LLMClient
from jarvis.agent.react_agent import ReActAgent
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.utils.config import load_config


async def main():
    config = load_config("config.lfm.yaml")
    client = LLMClient(config.llm)
    registry = ToolRegistry()
    calls = []

    @registry.tool("get_test_code", "Return the test code for a named test fixture.",
                   {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})
    async def code(name):
        calls.append(name)
        return "The test code is ZEBRA-42."

    try:
        print("Loading LFM (first run downloads model weights)...", flush=True)
        await client.startup()
        agent = ReActAgent(client, registry, config.llm.system_prompt)
        for prompt in ("How are you doing?", "Use get_test_code for the fixture named demo and tell me its code."):
            started = time.monotonic()
            first = None
            chunks = []
            async for chunk in agent.run_stream(prompt):
                if first is None:
                    first = time.monotonic() - started
                chunks.append(chunk)
            answer = "".join(chunks)
            print(f"first={first:.2f}s total={time.monotonic()-started:.2f}s: {answer}", flush=True)
        assert calls == ["demo"], calls
        assert "ZEBRA-42" in answer, answer
        print("PASS: real LFM tool selection, execution and result interpretation", flush=True)
    finally:
        await client.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
