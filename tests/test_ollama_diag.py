"""Quick diagnostic — test basic Ollama chat without tools."""
import asyncio
import httpx

async def test():
    async with httpx.AsyncClient(base_url="http://localhost:11434", timeout=120) as c:
        # Test 1: Simple chat (no tools)
        print("Test 1: Simple chat...")
        r = await c.post("/api/chat", json={
            "model": "hf.co/unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M",
            "messages": [{"role": "user", "content": "Hi, say hello in one sentence."}],
            "stream": False,
        })
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            print(f"  Response: {r.json()['message']['content'][:200]}")
        else:
            print(f"  Error: {r.text[:300]}")

        # Test 2: Chat with tools
        print("\nTest 2: Chat with tool definitions...")
        r2 = await c.post("/api/chat", json={
            "model": "hf.co/unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M",
            "messages": [{"role": "user", "content": "What are my system specs?"}],
            "stream": False,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_system_info",
                    "description": "Get current system information.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
        })
        print(f"  Status: {r2.status_code}")
        if r2.status_code == 200:
            msg = r2.json()["message"]
            print(f"  Content: {msg.get('content', '(none)')[:200]}")
            print(f"  Tool calls: {msg.get('tool_calls', '(none)')}")
        else:
            print(f"  Error: {r2.text[:300]}")

asyncio.run(test())
