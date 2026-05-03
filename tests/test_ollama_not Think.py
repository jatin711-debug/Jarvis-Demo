"""Test Qwen3 with thinking disabled."""
import asyncio
import json
import httpx

async def test(name: str, payload: dict):
    print(f"\n=== {name} ===")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post("http://localhost:11434/api/chat", json=payload)
        data = resp.json()
        print("Content:", repr(data.get("message", {}).get("content", "EMPTY")))
        print("Thinking:", repr(data.get("message", {}).get("thinking", "NONE")))
        print("Done reason:", data.get("done_reason"))
        print("Eval count:", data.get("eval_count"))

async def main():
    # Test 1: /no_think in system prompt
    await test("/no_think in system prompt", {
        "model": "qwen3:4b",
        "messages": [
            {"role": "system", "content": "You are Jarvis, a helpful voice assistant. Be concise. /no_think"},
            {"role": "user", "content": "Hi! What's your name?"}
        ],
        "stream": False,
        "options": {"temperature": 0.6, "num_predict": 100}
    })

    # Test 2: /no_think in user message
    await test("/no_think in user message", {
        "model": "qwen3:4b",
        "messages": [
            {"role": "system", "content": "You are Jarvis, a helpful voice assistant. Be concise."},
            {"role": "user", "content": "Hi! What's your name? /no_think"}
        ],
        "stream": False,
        "options": {"temperature": 0.6, "num_predict": 100}
    })

    # Test 3: Explicit instruction (no special suffix)
    await test("Explicit no-think instruction", {
        "model": "qwen3:4b",
        "messages": [
            {"role": "system", "content": "You are Jarvis. Be concise. Do not think step-by-step. Respond immediately without reasoning."},
            {"role": "user", "content": "Hi! What's your name?"}
        ],
        "stream": False,
        "options": {"temperature": 0.6, "num_predict": 100}
    })

if __name__ == "__main__":
    asyncio.run(main())
