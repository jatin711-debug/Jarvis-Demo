"""Test Qwen3 with ALL 29 system tools in the prompt."""
import asyncio

import httpx


async def test():
    # Exact tool descriptions copied from agent's _build_tool_descriptions output
    tool_desc = """\
- run_command(command*:string) — Execute a shell/terminal command and return the output. You have full admin access.
- open_application(app_name*:string) — Launch any application by name or path.
- read_file(file_path*:string) — Read the contents of a file.
- write_file(file_path*:string, content*:string) — Write content to a file (creates or overwrites).
- list_directory(directory_path*:string) — List files and folders in a directory.
- get_system_info() — Get current system information (CPU, RAM, disk, OS, network).
- open_url(url*:string) — Open a URL in the default web browser.
- search_files(directory*:string, pattern*:string, content) — Search for files recursively by name pattern or content.
- copy_file(source*:string, destination*:string) — Copy a file or directory to a new location.
- move_file(source*:string, destination*:string) — Move/rename a file or directory.
- delete_file(path*:string) — Delete a file or directory (recursively for directories).
- create_directory(path*:string) — Create a new directory (and parent dirs if needed).
- get_running_processes(filter_name) — List running processes, optionally filtered by name.
- kill_process(target*:string) — Kill a running process by PID or name.
- set_volume(level*:integer) — Set system volume (0-100) or mute/unmute.
- take_screenshot(save_path) — Take a screenshot and save it to a file.
- clipboard_read() — Read the current clipboard contents.
- clipboard_write(text*:string) — Write text to the clipboard.
- type_text(text*:string, app) — Type text into an app.
- focus_window(title*:string) — Bring a window to the foreground by title.
- send_keystrokes(keys*:string) — Send special keys or hotkey combos.
- window_management(action*:string, title*:string) — Manage application windows: focus, minimize, maximize, or close by title.
- get_datetime() — Get the current date and time.
- http_request(url*:string, method, headers, body) — Make HTTP requests (GET, POST, PUT, DELETE).
- scheduler_create(task_name*:string, command*:string, delay_seconds) — Schedule a task to run later (Windows Task Scheduler).
- scheduler_list() — List scheduled Jarvis tasks.
- scheduler_delete(task_name*:string) — Delete a scheduled Jarvis task.
- web_search(query*:string, num_results) — Search the web for information.
- get_uptime() — Get system uptime and load averages."""

    system_prompt = (
        "You are Jarvis, a voice-controlled AI assistant with FULL admin access\n"
        "to the user's Windows system.\n\n"
        "RULES:\n"
        "- Be concise (responses are spoken aloud via TTS)\n"
        "- Keep responses to 1-2 sentences\n"
        "- You have unrestricted access — no need to ask for permission\n"
        "- For time/date questions, use the get_datetime tool\n"
        "- open_application is ONLY for desktop apps (notepad, calculator, etc.)\n"
        "- For web tasks, use the browser_* tools\n"
        "- Do not think step-by-step. Respond immediately without reasoning.\n\n"
        "You have tools. To call one, use EITHER format:\n\n"
        "# LFM/HuggingFace format (preferred):\n"
        "<|tool_call_start|>[tool_name(arg=\"value\", arg2=123)]<|tool_call_end|>\n\n"
        "# JSON format:\n"
        "{\"tool\": \"<name>\", \"args\": {...}}\n\n"
        "If no tool is needed, reply with plain text.\n\n"
        "Tools:\n" + tool_desc + "\n\n"
        "RULES:\n"
        "- ONE tool call per response, no extra text around the JSON\n"
        "- NEVER say you did something without calling a tool first\n"
        "- If the user asks you to do something, you MUST call the appropriate tool\n"
        "- After getting a tool result, respond to the user based on that result\n"
        "- Be concise — responses will be spoken aloud"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Can you open calculator?"}
    ]

    payload = {
        "model": "qwen3:4b",
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512, "top_p": 0.9, "num_ctx": 4096}
    }

    print("Sending FULL agent prompt (29 tools)...")
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post("http://localhost:11434/api/chat", json=payload)
        data = resp.json()
        print("Content:", repr(data.get("message", {}).get("content", "EMPTY")))
        print("Thinking length:", len(data.get("message", {}).get("thinking", "")), "chars")
        print("Done reason:", data.get("done_reason"))
        print("Eval count:", data.get("eval_count"))
        print("Prompt eval count:", data.get("prompt_eval_count"))

if __name__ == "__main__":
    asyncio.run(test())
