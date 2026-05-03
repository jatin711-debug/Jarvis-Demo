"""Quick test: Ollama with plain system prompt (no special tokens)."""
import asyncio
import json
import httpx

async def test():
    payload = {
        "model": "qwen3:4b",
        "messages": [
            {"role": "system", "content": "You are Jarvis, a helpful voice assistant. Be concise."},
            {"role": "user", "content": "Hi! What's your name?"}
        ],
        "stream": False,
        "options": {"temperature": 0.6, "num_predict": 100}
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post("http://localhost:11434/api/chat", json=payload)
        data = resp.json()
        print("FULL RESPONSE:")
        print(json.dumps(data, indent=2))
        print("\n---")
        print("Content:", repr(data.get("message", {}).get("content", "EMPTY")))

if __name__ == "__main__":
    asyncio.run(test())
