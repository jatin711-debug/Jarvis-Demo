"""Test Mem0.ai integration with local Ollama models."""
import asyncio
import sys

sys.path.insert(0, ".")

from jarvis.memory.manager import MemoryManager
from jarvis.utils.config import load_config


async def test_memory():
    config = load_config()

    print("1. Initializing memory manager...")
    # Change the path to a temporary test directory so we don't pollute real memories
    config.memory.vector_store.path = "./.jarvis_test/vectors"

    manager = MemoryManager(config.memory, config.llm)
    await manager.startup()

    if not manager._memory:
        print("Mem0 failed to initialize")
        return

    print("2. Adding memories...")
    # Add some conversation history
    await manager.add_conversation(
        "Hi, I'm Alex. I like to code in Python and Rust.",
        "Nice to meet you, Alex! I'll remember your programming preferences.",
        user_id="test_user"
    )

    await manager.add_conversation(
        "Can you remind me of my favorite coffee?",
        "I don't know your favorite coffee yet.",
        user_id="test_user"
    )

    await manager.add_conversation(
        "Oh right. It's an iced latte with oat milk.",
        "Got it! I've updated your preferences.",
        user_id="test_user"
    )

    print("3. Retrieving context for 'What drinks do I like?'...")
    context = await manager.get_context("What drinks do I like?", user_id="test_user")

    print(f"\nRetrieved {len(context)} memories:")
    for i, mem in enumerate(context, 1):
        print(f"  {i}. {mem}")

    print("\n4. Getting all explicitly saved memories...")
    all_memories = await manager.get_all_memories(user_id="test_user")
    print(f"\nTotal memories in DB: {len(all_memories)}")
    for i, mem in enumerate(all_memories, 1):
        # Depending on mem0 version, format gets returned differently
        mem_text = mem.get('memory', mem) if isinstance(mem, dict) else mem
        print(f"  {i}. {mem_text}")

    await manager.shutdown()
    print("\nMemory test complete!")

if __name__ == "__main__":
    asyncio.run(test_memory())
