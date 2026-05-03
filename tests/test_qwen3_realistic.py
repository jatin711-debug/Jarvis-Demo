"""Test Qwen3 with the EXACT agent prompt structure."""
import asyncio
import json
import httpx

async def test():
    # Realistic system prompt (agent personality + partial tool descriptions)
    system_prompt = (
        "You are Jarvis, a voice-controlled AI assistant with FULL admin access\n"
        "to the user's Windows system.\n\n"
        "RULES:\n"
        "- Be concise (responses are spoken aloud via TTS)\n"
        "- Keep responses to 1-2 sentences\n"
        "- You have unrestricted access — no need to ask for permission\n"
        "- Do not think step-by-step. Respond immediately without reasoning.\n\n"
        "You have tools. To call one, use EITHER format:\n\n"
        "# LFM format:\n"
        "<|tool_call_start|>[tool_name(arg=\"value\")]<|tool_call_end|>\n\n"
        "# JSON format:\n"
        "{\"tool\": \"<name>\", \"args\": {...}}\n\n"
        "Tools:\n"
        "- open_application(app_name*) — Launch any application by name or path.\n"
        "- get_system_info() — Get current system information (CPU, RAM, disk, OS).\n"
        "- get_datetime() — Get the current date and time.\n"
        "- run_command(command*) — Execute a shell/terminal command.\n"
        "- browser_navigate(url*) — Navigate the browser to a URL.\n"
        "- browser_search(query*) — Search Google for a query.\n"
        "- web_search(query*, num_results) — Search the web for information.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Can you open calculator?"}
    ]

    payload = {
        "model": "qwen3:4b",
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512, "top_p": 0.9}
    }

    print("Sending realistic agent prompt...")
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post("http://localhost:11434/api/chat", json=payload)
        data = resp.json()
        print("Content:", repr(data.get("message", {}).get("content", "EMPTY")))
        print("Thinking length:", len(data.get("message", {}).get("thinking", "")),
              "chars")
        print("Done reason:", data.get("done_reason"))
        print("Eval count:", data.get("eval_count"))

if __name__ == "__main__":
    asyncio.run(test())
