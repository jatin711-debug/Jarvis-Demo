"""Quick end-to-end test of the Jarvis agent pipeline."""

import asyncio
import sys

sys.path.insert(0, ".")

from jarvis.agent.llm_client import LLMClient
from jarvis.agent.react_agent import ReActAgent
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.tools.system_tools import register_system_tools
from jarvis.utils.config import load_config


async def test():
    config = load_config()

    # 1. LLM Client
    print("1. Starting LLM client...")
    llm = LLMClient(config.llm)
    await llm.startup()

    # 2. Tool registry
    print("2. Registering tools...")
    tools = ToolRegistry()
    register_system_tools(tools, config.security, llm_client=llm)
    print(f"   Registered tools: {tools.list_tools()}")

    # 3. Agent
    print("3. Creating agent...")
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        system_prompt=config.llm.system_prompt,
    )

    # 4. Test simple chat (no tool use)
    print("\n--- Test 1: Simple chat ---")
    response = await agent.run("Hi! What's your name and what can you do?")
    print(f"Response: {response}")

    # 5. Test tool use (system info)
    print("\n--- Test 2: Tool calling (get_system_info) ---")
    response = await agent.run("What are my system specs? CPU, RAM, etc.")
    print(f"Response: {response}")

    # 6. Test tool use (list directory)
    print("\n--- Test 3: Tool calling (list_directory) ---")
    response = await agent.run("List the files in the current directory.")
    print(f"Response: {response}")

    await llm.shutdown()
    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(test())
