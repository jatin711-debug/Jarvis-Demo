"""
JARVIS Built-in System Tools
==============================
Core tools for shell commands, file operations, app launching, etc.
Full admin access — no whitelist, no blocked commands.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
from pathlib import Path

import psutil

from jarvis.agent.tool_registry import ToolRegistry
from jarvis.utils.logger import get_logger
from jarvis.utils.config import SecurityConfig

logger = get_logger(__name__)


def register_system_tools(
    registry: ToolRegistry,
    security: SecurityConfig,
) -> None:
    """Register all built-in system tools."""

    # ── Shell Command Execution ──

    @registry.tool(
        name="run_command",
        description="Execute a shell/terminal command and return the output. You have full admin access.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
            },
            "required": ["command"],
        },
    )
    async def run_command(command: str) -> str:
        try:
            result = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                result.communicate(), timeout=120,
            )
            output = stdout.decode("utf-8", errors="replace").strip()
            errors = stderr.decode("utf-8", errors="replace").strip()
            if errors:
                output += f"\n[stderr]: {errors}"
            return output or "(no output)"
        except asyncio.TimeoutError:
            return "❌ Command timed out after 120 seconds."
        except Exception as e:
            return f"❌ Error: {e}"

    # ── App name aliases (shared by open_application and type_text) ──
    _WIN_ALIASES: dict[str, str] = {
        "calculator": "calc",
        "calc": "calc",
        "notepad": "notepad",
        "paint": "mspaint",
        "snipping tool": "snippingtool",
        "task manager": "taskmgr",
        "control panel": "control",
        "file explorer": "explorer",
        "explorer": "explorer",
        "command prompt": "cmd",
        "cmd": "cmd",
        "powershell": "powershell",
        "settings": "ms-settings:",
        "store": "ms-windows-store:",
        "mail": "outlookmail:",
        "calendar": "outlookcal:",
        "clock": "ms-clock:",
        "alarms": "ms-clock:",
        "weather": "bingweather:",
        "maps": "bingmaps:",
        "camera": "microsoft.windows.camera:",
        "photos": "ms-photos:",
        "xbox": "xbox:",
        "word": "winword",
        "excel": "excel",
        "powerpoint": "powerpnt",
        "spotify": "spotify",
        "discord": "discord",
        "teams": "msteams:",
        "terminal": "wt",
        "windows terminal": "wt",
    }

    # ── Open Application ──

    @registry.tool(
        name="open_application",
        description="Launch any application by name or path.",
        parameters={
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Name or command of the application to launch.",
                },
            },
            "required": ["app_name"],
        },
    )
    async def open_application(app_name: str) -> str:
        try:
            key = app_name.strip().lower()
            if platform.system() == "Windows":
                cmd = _WIN_ALIASES.get(key, app_name)
                subprocess.Popen(
                    f'start "" "{cmd}"',
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    [app_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return f"✅ Launched '{app_name}'"
        except Exception as e:
            return f"❌ Failed to launch '{app_name}': {e}"

    # ── File Read ──

    @registry.tool(
        name="read_file",
        description="Read the contents of a file.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file.",
                },
            },
            "required": ["file_path"],
        },
    )
    async def read_file(file_path: str) -> str:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return f"❌ File not found: {path}"
        if not path.is_file():
            return f"❌ Not a file: {path}"
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 50000:
                return content[:50000] + f"\n\n... (truncated, {len(content)} total chars)"
            return content
        except Exception as e:
            return f"❌ Error reading file: {e}"

    # ── File Write ──

    @registry.tool(
        name="write_file",
        description="Write content to a file (creates or overwrites).",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            "required": ["file_path", "content"],
        },
    )
    async def write_file(file_path: str, content: str) -> str:
        path = Path(file_path).expanduser().resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"✅ Written {len(content)} chars to {path}"
        except Exception as e:
            return f"❌ Error writing file: {e}"

    # ── List Directory ──

    @registry.tool(
        name="list_directory",
        description="List files and folders in a directory.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory.",
                },
            },
            "required": ["path"],
        },
    )
    async def list_directory(path: str) -> str:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            return f"❌ Directory not found: {path}"
        if not path.is_dir():
            return f"❌ Not a directory: {path}"
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            lines = []
            for entry in entries[:200]:
                prefix = "📁" if entry.is_dir() else "📄"
                size = ""
                if entry.is_file():
                    try:
                        s = entry.stat().st_size
                        size = f" ({s:,} bytes)"
                    except OSError:
                        pass
                lines.append(f"{prefix} {entry.name}{size}")
            result = "\n".join(lines)
            if len(entries) > 200:
                result += f"\n... and {len(entries) - 200} more items"
            return result
        except Exception as e:
            return f"❌ Error listing directory: {e}"

    # ── System Info ──

    @registry.tool(
        name="get_system_info",
        description="Get current system information (CPU, RAM, disk, OS, network).",
        parameters={
            "type": "object",
            "properties": {},
        },
    )
    async def get_system_info() -> str:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            boot_time = psutil.boot_time()
            import datetime
            boot_dt = datetime.datetime.fromtimestamp(boot_time)

            info = (
                f"OS: {platform.system()} {platform.release()} ({platform.machine()})\n"
                f"CPU: {psutil.cpu_count()} cores, {cpu_percent}% usage\n"
                f"RAM: {memory.used / (1024**3):.1f} / {memory.total / (1024**3):.1f} GB "
                f"({memory.percent}%)\n"
                f"Disk: {disk.used / (1024**3):.1f} / {disk.total / (1024**3):.1f} GB "
                f"({disk.percent}%)\n"
                f"Uptime since: {boot_dt.strftime('%Y-%m-%d %H:%M')}\n"
                f"Python: {platform.python_version()}"
            )
            return info
        except Exception as e:
            return f"❌ Error getting system info: {e}"

    # ── Open URL ──

    @registry.tool(
        name="open_url",
        description="Open a URL in the default web browser.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to open.",
                },
            },
            "required": ["url"],
        },
    )
    async def open_url(url: str) -> str:
        import webbrowser
        try:
            webbrowser.open(url)
            return f"✅ Opened {url} in browser"
        except Exception as e:
            return f"❌ Error opening URL: {e}"

    # ── Search Files ──

    @registry.tool(
        name="search_files",
        description="Search for files recursively by name pattern or content.",
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to search in.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Filename pattern to match (e.g., '*.py', 'report*').",
                },
                "content": {
                    "type": "string",
                    "description": "Optional: search for this text inside files.",
                },
            },
            "required": ["directory", "pattern"],
        },
    )
    async def search_files(directory: str, pattern: str, content: str = "") -> str:
        import fnmatch
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            return f"❌ Directory not found: {root}"
        matches = []
        try:
            for path in root.rglob("*"):
                if len(matches) >= 100:
                    break
                if not path.is_file():
                    continue
                if not fnmatch.fnmatch(path.name.lower(), pattern.lower()):
                    continue
                if content:
                    try:
                        text = path.read_text(encoding="utf-8", errors="replace")
                        if content.lower() not in text.lower():
                            continue
                    except (OSError, UnicodeDecodeError):
                        continue
                matches.append(str(path))
        except PermissionError:
            pass
        if not matches:
            return f"No files matching '{pattern}' found in {root}"
        result = "\n".join(matches)
        if len(matches) >= 100:
            result += "\n... (capped at 100 results)"
        return result

    # ── Copy / Move / Delete File ──

    @registry.tool(
        name="copy_file",
        description="Copy a file or directory to a new location.",
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source path."},
                "destination": {"type": "string", "description": "Destination path."},
            },
            "required": ["source", "destination"],
        },
    )
    async def copy_file(source: str, destination: str) -> str:
        src = Path(source).expanduser().resolve()
        dst = Path(destination).expanduser().resolve()
        try:
            if src.is_dir():
                shutil.copytree(str(src), str(dst))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
            return f"✅ Copied {src} → {dst}"
        except Exception as e:
            return f"❌ Copy failed: {e}"

    @registry.tool(
        name="move_file",
        description="Move/rename a file or directory.",
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source path."},
                "destination": {"type": "string", "description": "Destination path."},
            },
            "required": ["source", "destination"],
        },
    )
    async def move_file(source: str, destination: str) -> str:
        src = Path(source).expanduser().resolve()
        dst = Path(destination).expanduser().resolve()
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            return f"✅ Moved {src} → {dst}"
        except Exception as e:
            return f"❌ Move failed: {e}"

    @registry.tool(
        name="delete_file",
        description="Delete a file or directory (recursively for directories).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to delete."},
            },
            "required": ["path"],
        },
    )
    async def delete_file(path: str) -> str:
        p = Path(path).expanduser().resolve()
        try:
            if p.is_dir():
                shutil.rmtree(str(p))
                return f"✅ Deleted directory: {p}"
            elif p.is_file():
                p.unlink()
                return f"✅ Deleted file: {p}"
            else:
                return f"❌ Path not found: {p}"
        except Exception as e:
            return f"❌ Delete failed: {e}"

    @registry.tool(
        name="create_directory",
        description="Create a new directory (and parent dirs if needed).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to create."},
            },
            "required": ["path"],
        },
    )
    async def create_directory(path: str) -> str:
        p = Path(path).expanduser().resolve()
        try:
            p.mkdir(parents=True, exist_ok=True)
            return f"✅ Created directory: {p}"
        except Exception as e:
            return f"❌ Failed to create directory: {e}"

    # ── Process Management ──

    @registry.tool(
        name="get_running_processes",
        description="List running processes, optionally filtered by name.",
        parameters={
            "type": "object",
            "properties": {
                "filter_name": {
                    "type": "string",
                    "description": "Optional: filter processes by name substring.",
                },
            },
        },
    )
    async def get_running_processes(filter_name: str = "") -> str:
        try:
            procs = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
                try:
                    info = proc.info
                    if filter_name and filter_name.lower() not in info["name"].lower():
                        continue
                    mem_mb = info["memory_info"].rss / (1024 * 1024) if info["memory_info"] else 0
                    procs.append((info["pid"], info["name"], info["cpu_percent"] or 0, mem_mb))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            procs.sort(key=lambda x: x[3], reverse=True)  # Sort by memory
            lines = [f"{'PID':<8} {'Name':<30} {'CPU%':<8} {'RAM MB':<10}"]
            for pid, name, cpu, mem in procs[:50]:
                lines.append(f"{pid:<8} {name:<30} {cpu:<8.1f} {mem:<10.1f}")
            if len(procs) > 50:
                lines.append(f"... and {len(procs) - 50} more")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    @registry.tool(
        name="kill_process",
        description="Kill a running process by PID or name.",
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Process PID (number) or name to kill.",
                },
            },
            "required": ["target"],
        },
    )
    async def kill_process(target: str) -> str:
        try:
            # Try as PID first
            try:
                pid = int(target)
                proc = psutil.Process(pid)
                proc.terminate()
                return f"✅ Terminated process {proc.name()} (PID {pid})"
            except ValueError:
                pass

            # Kill by name
            killed = []
            for proc in psutil.process_iter(["pid", "name"]):
                if target.lower() in proc.info["name"].lower():
                    try:
                        proc.terminate()
                        killed.append(f"{proc.info['name']} (PID {proc.info['pid']})")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            if killed:
                return f"✅ Terminated: {', '.join(killed)}"
            return f"❌ No process found matching '{target}'"
        except Exception as e:
            return f"❌ Error: {e}"

    # ── Volume Control (Windows) ──

    @registry.tool(
        name="set_volume",
        description="Set system volume (0-100) or mute/unmute.",
        parameters={
            "type": "object",
            "properties": {
                "level": {
                    "type": "integer",
                    "description": "Volume level 0-100, or -1 to toggle mute.",
                },
            },
            "required": ["level"],
        },
    )
    async def set_volume(level: int) -> str:
        if platform.system() != "Windows":
            return "❌ Volume control only supported on Windows"
        try:
            if level == -1:
                # Toggle mute via nircmd or PowerShell
                result = await asyncio.create_subprocess_shell(
                    'powershell -c "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await result.communicate()
                return "✅ Toggled mute"
            else:
                level = max(0, min(100, level))
                # Use PowerShell to set volume
                ps_cmd = f"""
                $vol = {level} / 100;
                $obj = New-Object -ComObject WScript.Shell;
                # Reset to 0 then increment
                for($i=0; $i -lt 50; $i++) {{ $obj.SendKeys([char]174) }};
                $steps = [math]::Round({level} / 2);
                for($i=0; $i -lt $steps; $i++) {{ $obj.SendKeys([char]175) }};
                """
                result = await asyncio.create_subprocess_shell(
                    f'powershell -c "{ps_cmd}"',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(result.communicate(), timeout=10)
                return f"✅ Volume set to ~{level}%"
        except Exception as e:
            return f"❌ Error setting volume: {e}"

    # ── Screenshot ──

    @registry.tool(
        name="take_screenshot",
        description="Take a screenshot and save it to a file.",
        parameters={
            "type": "object",
            "properties": {
                "save_path": {
                    "type": "string",
                    "description": "Path to save the screenshot (default: ~/.jarvis/screenshot.png).",
                },
            },
        },
    )
    async def take_screenshot(save_path: str = "") -> str:
        try:
            import pyautogui
            if not save_path:
                save_dir = Path.home() / ".jarvis"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(save_dir / "screenshot.png")
            img = pyautogui.screenshot()
            img.save(save_path)
            return f"✅ Screenshot saved to {save_path}"
        except ImportError:
            return "❌ pyautogui not installed. Run: pip install pyautogui"
        except Exception as e:
            return f"❌ Screenshot failed: {e}"

    # ── Clipboard ──

    @registry.tool(
        name="clipboard_read",
        description="Read the current clipboard contents.",
        parameters={"type": "object", "properties": {}},
    )
    async def clipboard_read() -> str:
        try:
            import pyperclip
            content = pyperclip.paste()
            return content if content else "(clipboard is empty)"
        except ImportError:
            return "❌ pyperclip not installed. Run: pip install pyperclip"
        except Exception as e:
            return f"❌ Error: {e}"

    @registry.tool(
        name="clipboard_write",
        description="Write text to the clipboard.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to copy to clipboard."},
            },
            "required": ["text"],
        },
    )
    async def clipboard_write(text: str) -> str:
        try:
            import pyperclip
            pyperclip.copy(text)
            return f"✅ Copied {len(text)} chars to clipboard"
        except ImportError:
            return "❌ pyperclip not installed. Run: pip install pyperclip"
        except Exception as e:
            return f"❌ Error: {e}"

    # ── Keyboard / Window Control ──

    # App names → window title substrings for matching
    _WINDOW_TITLES: dict[str, str] = {
        "calculator": "Calculator", "calc": "Calculator",
        "notepad": "Notepad", "paint": "Paint",
        "word": "Word", "excel": "Excel", "powerpoint": "PowerPoint",
        "chrome": "Chrome", "firefox": "Firefox", "edge": "Edge",
        "file explorer": "File Explorer", "explorer": "File Explorer",
        "spotify": "Spotify", "discord": "Discord",
        "terminal": "Terminal", "command prompt": "Command Prompt",
        "powershell": "PowerShell", "visual studio code": "Visual Studio Code",
        "vscode": "Visual Studio Code",
    }

    @registry.tool(
        name="type_text",
        description="Type text into an app. Set 'app' to auto-open and focus it first (e.g. app='calculator'). Use for typing numbers in Calculator, text in Notepad, etc.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to type (e.g., '5+2', 'Hello World').",
                },
                "app": {
                    "type": "string",
                    "description": "App to type into (e.g., 'calculator', 'notepad'). Opens it if not already running.",
                },
            },
            "required": ["text"],
        },
    )
    async def type_text(text: str, app: str = "") -> str:
        try:
            import pyautogui
            import time

            # If app specified, ensure it's open and focused
            if app:
                app_key = app.strip().lower()
                win_title = _WINDOW_TITLES.get(app_key, app)
                windows = pyautogui.getWindowsWithTitle(win_title)
                if not windows:
                    # App not open — launch it
                    cmd = _WIN_ALIASES.get(app_key, app)
                    subprocess.Popen(
                        f'start "" "{cmd}"', shell=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    # Wait for window to appear (up to 5s)
                    for _ in range(25):
                        time.sleep(0.2)
                        windows = pyautogui.getWindowsWithTitle(win_title)
                        if windows:
                            break
                if windows:
                    try:
                        windows[0].activate()
                        time.sleep(0.5)
                    except Exception:
                        pass
                else:
                    return f"❌ Could not open or find {app} window"

            # Type each character via press() — works with UWP apps like Calculator
            for ch in text:
                pyautogui.press(ch)
                time.sleep(0.05)

            target = f" in {app}" if app else ""
            return f"✅ Typed '{text}'{target}"
        except ImportError:
            return "❌ pyautogui not installed. Run: pip install pyautogui"
        except Exception as e:
            return f"❌ Error typing text: {e}"

    @registry.tool(
        name="focus_window",
        description="Bring a window to the foreground by title. Use before sending keystrokes.",
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Window title to search for (partial match, e.g., 'Calculator', 'Notepad').",
                },
            },
            "required": ["title"],
        },
    )
    async def focus_window(title: str) -> str:
        if platform.system() != "Windows":
            return "❌ Only supported on Windows"
        try:
            import pyautogui
            import time
            windows = pyautogui.getWindowsWithTitle(title)
            if not windows:
                return f"❌ No window found with title containing '{title}'"
            win = windows[0]
            win.activate()
            time.sleep(0.3)
            return f"✅ Focused window: {win.title}"
        except ImportError:
            return "❌ pyautogui not installed"
        except Exception as e:
            return f"❌ Error: {e}"

    @registry.tool(
        name="send_keystrokes",
        description="Send special keys or hotkey combos. Use for shortcuts like 'ctrl+c', 'alt+tab', 'enter', 'backspace', 'escape', 'tab'. NOT for typing text — use type_text for that.",
        parameters={
            "type": "object",
            "properties": {
                "keys": {
                    "type": "string",
                    "description": "Keys to press. Use '+' for combos (e.g., 'ctrl+s', 'alt+f4').",
                },
            },
            "required": ["keys"],
        },
    )
    async def send_keystrokes(keys: str) -> str:
        try:
            import pyautogui
            import time
            time.sleep(0.3)  # Small delay for window focus
            if "+" in keys:
                pyautogui.hotkey(*keys.split("+"))
            else:
                pyautogui.press(keys)
            return f"✅ Sent keystrokes: {keys}"
        except ImportError:
            return "❌ pyautogui not installed. Run: pip install pyautogui"
        except Exception as e:
            return f"❌ Error: {e}"

    @registry.tool(
        name="window_management",
        description="Manage application windows: focus, minimize, maximize, or close by title.",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action: 'focus', 'minimize', 'maximize', or 'close'.",
                },
                "title": {
                    "type": "string",
                    "description": "Window title (partial match).",
                },
            },
            "required": ["action", "title"],
        },
    )
    async def window_management(action: str, title: str) -> str:
        if platform.system() != "Windows":
            return "❌ Window management only supported on Windows"
        try:
            import pyautogui
            windows = pyautogui.getWindowsWithTitle(title)
            if not windows:
                return f"❌ No window found with title containing '{title}'"
            win = windows[0]
            if action == "focus":
                win.activate()
            elif action == "minimize":
                win.minimize()
            elif action == "maximize":
                win.maximize()
            elif action == "close":
                win.close()
            else:
                return f"❌ Unknown action: {action}. Use focus/minimize/maximize/close."
            return f"✅ {action.title()}ed window: {win.title}"
        except ImportError:
            return "❌ pyautogui not installed. Run: pip install pyautogui"
        except Exception as e:
            return f"❌ Error: {e}"

    # ── Get Current Time/Date ──

    @registry.tool(
        name="get_datetime",
        description="Get the current date and time.",
        parameters={"type": "object", "properties": {}},
    )
    async def get_datetime() -> str:
        import datetime
        now = datetime.datetime.now()
        return now.strftime("Date: %A, %B %d, %Y\nTime: %I:%M %p")

    # ── HTTP Requests (NullClaw-inspired) ──

    @registry.tool(
        name="http_request",
        description="Make HTTP requests (GET, POST, PUT, DELETE). Supports headers, JSON body, and returns status + response.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to request."},
                "method": {
                    "type": "string",
                    "description": "HTTP method: GET, POST, PUT, DELETE (default: GET)",
                },
                "headers": {
                    "type": "string",
                    "description": "Optional JSON headers (e.g., {'Authorization': 'Bearer ...'})",
                },
                "body": {
                    "type": "string",
                    "description": "Optional JSON body for POST/PUT",
                },
            },
            "required": ["url"],
        },
    )
    async def http_request(url: str, method: str = "GET", headers: str = "", body: str = "") -> str:
        try:
            import httpx
            import json as _json
            hdrs = _json.loads(headers) if headers else {}
            bdy = _json.loads(body) if body else None

            async with httpx.AsyncClient() as client:
                resp = await client.request(
                    method.upper(), url, headers=hdrs, json=bdy, timeout=30.0
                )
                return (
                    f"Status: {resp.status_code}\n"
                    f"Headers: {dict(resp.headers)}\n"
                    f"Body: {resp.text[:2000]}"
                )
        except Exception as e:
            return f"❌ HTTP error: {e}"

    # ── Scheduling (NullClaw-inspired) ──

    @registry.tool(
        name="scheduler_create",
        description="Schedule a task to run later (Windows Task Scheduler). Requires app to be running.",
        parameters={
            "type": "object",
            "properties": {
                "task_name": {"type": "string", "description": "Name for the scheduled task."},
                "command": {"type": "string", "description": "Command to run."},
                "delay_seconds": {
                    "type": "integer",
                    "description": "Delay in seconds before running (default: 60)",
                },
            },
            "required": ["task_name", "command"],
        },
    )
    async def scheduler_create(task_name: str, command: str, delay_seconds: int = 60) -> str:
        import subprocess
        try:
            # Use Windows schtasks
            task_cmd = f'schtasks /create /tn "Jarvis_{task_name}" /tr "{command}" /sc once /st +00:{delay_seconds}'
            result = subprocess.run(
                task_cmd, shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                return f"✅ Scheduled '{task_name}' to run in {delay_seconds}s"
            return f"❌ Failed: {result.stderr}"
        except Exception as e:
            return f"❌ Scheduler error: {e}"

    @registry.tool(
        name="scheduler_list",
        description="List scheduled Jarvis tasks.",
        parameters={"type": "object", "properties": {}},
    )
    async def scheduler_list() -> str:
        import subprocess
        try:
            result = subprocess.run(
                'schtasks /query /fo csv /v | findstr "Jarvis_"',
                shell=True, capture_output=True, text=True
            )
            return result.stdout or "No Jarvis tasks scheduled."
        except Exception as e:
            return f"❌ List error: {e}"

    @registry.tool(
        name="scheduler_delete",
        description="Delete a scheduled Jarvis task.",
        parameters={
            "type": "object",
            "properties": {
                "task_name": {"type": "string", "description": "Task name to delete."},
            },
            "required": ["task_name"],
        },
    )
    async def scheduler_delete(task_name: str) -> str:
        import subprocess
        try:
            result = subprocess.run(
                f'schtasks /delete /tn "Jarvis_{task_name}" /f',
                shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                return f"✅ Deleted task: {task_name}"
            return f"❌ Failed: {result.stderr}"
        except Exception as e:
            return f"❌ Delete error: {e}"

    # ── Web Search (NullClaw-inspired) ──

    @registry.tool(
        name="web_search",
        description="Search the web for information. Returns top results with snippets.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results (default: 5)",
                },
            },
            "required": ["query"],
        },
    )
    async def web_search(query: str, num_results: int = 5) -> str:
        try:
            import httpx
            # Use DuckDuckGo HTML search (no API key needed)
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query, "b": num_results},
                    timeout=15.0,
                )
                # Simple regex parsing for results
                import re
                results = []
                for match in re.finditer(
                    r'<a class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>.*?'
                    r'<a class="result__snippet"[^>]*>([^<]+)</a>',
                    resp.text, re.DOTALL
                ):
                    url, title, snippet = match.groups()
                    if len(results) >= num_results:
                        break
                    results.append(
                        f"• {title.strip()}\n  {snippet.strip()[:200]}\n  {url}"
                    )
                if results:
                    return f"🔍 Results for '{query}':\n\n" + "\n\n".join(results)
                return f"No results found for '{query}'"
        except Exception as e:
            return f"❌ Search error: {e}"

    # ── System Uptime ──

    @registry.tool(
        name="get_uptime",
        description="Get system uptime and load averages.",
        parameters={"type": "object", "properties": {}},
    )
    async def get_uptime() -> str:
        import platform
        import subprocess
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    "systeminfo | findstr /C:\"System Boot Time\"",
                    shell=True, capture_output=True, text=True
                )
                return result.stdout.strip() or "Uptime info unavailable"
            else:
                import uptime
                return f"Uptime: {uptime.format_timespan(uptime.uptime())}"
        except Exception as e:
            return f"❌ Error: {e}"

    logger.info("✅ Registered %d system tools", len(registry))
