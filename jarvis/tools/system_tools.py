"""
JARVIS Built-in System Tools
==============================
Core tools for shell commands, file operations, app launching, etc.
Dispatch is subject to the configured permission gateway.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil

from jarvis.agent.permissions import validate_command
from jarvis.agent.tool_registry import ToolRegistry
from jarvis.utils.config import SecurityConfig
from jarvis.utils.logger import get_logger

logger = get_logger(__name__)

# Limits that keep a single tool call from stalling the assistant.
_COMMAND_TIMEOUT_S = 120.0
_READ_LIMIT_CHARS = 50_000
_SEARCH_MAX_RESULTS = 100
_SEARCH_MAX_CONTENT_BYTES = 2_000_000  # larger files are matched by name only
_SEARCH_TIME_BUDGET_S = 20.0


def _offloaded(fn: Callable[..., str]) -> Callable[..., Any]:
    """Run a blocking tool body on a worker thread.

    Tool handlers run on the event loop that also drives audio capture and
    barge-in, so file, process, and subprocess work must not run there.
    """
    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> str:
        return await asyncio.to_thread(fn, *args, **kwargs)
    return wrapper


def _kill_process_tree(pid: int) -> None:
    """Kill a process and everything it started (best effort)."""
    try:
        parent = psutil.Process(pid)
        doomed = [*parent.children(recursive=True), parent]
    except psutil.Error:
        return
    for proc in doomed:
        with contextlib.suppress(psutil.Error):
            proc.kill()


async def _run_shell(command: str, timeout: float) -> tuple[int, bytes, bytes]:
    """Run a shell command and return ``(exit_code, stdout, stderr)``.

    On timeout or cancellation the whole process tree is killed, so a hung or
    abandoned command never keeps running in the background.
    """
    proc = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, stdout, stderr
    except BaseException:
        _kill_process_tree(proc.pid)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=5)
        raise


def _write_atomic(path: Path, content: str) -> None:
    """Replace ``path`` with ``content`` so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        if path.exists():
            with contextlib.suppress(OSError):
                shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def register_system_tools(
    registry: ToolRegistry,
    security: SecurityConfig,
    llm_client: Any = None,
) -> None:
    """Register all built-in system tools."""

    # ── Shell Command Execution ──

    @registry.tool(
        name="run_command",
        description="Execute a shell/terminal command and return output, subject to configured permissions.",
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
        requires_confirmation=True,
    )
    async def run_command(command: str) -> str:
        try:
            validate_command(security, command)
        except ValueError as refusal:
            logger.warning("command blocked reason=policy")
            return str(refusal)
        try:
            code, stdout, stderr = await _run_shell(command, _COMMAND_TIMEOUT_S)
        except TimeoutError:
            logger.warning("command stopped after timeout=%gs", _COMMAND_TIMEOUT_S)
            return f"Command timed out after {_COMMAND_TIMEOUT_S:g} seconds and was stopped."
        except Exception as e:
            return f"Error: {e}"
        output = stdout.decode("utf-8", errors="replace").strip()
        errors = stderr.decode("utf-8", errors="replace").strip()
        if errors:
            output += f"\n[stderr]: {errors}"
        if code != 0:
            output += f"\n[exit code: {code}]"
        return output.strip() or "(no output)"

    # ── App name aliases (used by open_application) ──
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
        "vs code": "code",
        "vscode": "code",
        "visual studio code": "code",
        "code": "code",
        "chrome": "chrome",
        "google chrome": "chrome",
        "firefox": "firefox",
        "edge": "msedge",
        "microsoft edge": "msedge",
    }

    def _resolve_app_windows(name: str) -> str | None:
        """
        Resolve an app name to something actually launchable on Windows:
        alias table → PATH → Start Menu shortcuts → App Paths registry.
        Returns None if the app can't be found.
        """
        key = name.strip().lower()

        # 1. Alias table
        if key in _WIN_ALIASES:
            return _WIN_ALIASES[key]

        # 2. Already on PATH
        if shutil.which(name):
            return name

        # 3. Start Menu shortcuts (*.lnk)
        start_menu_dirs = [
            Path(os.environ.get("ProgramData", r"C:\ProgramData"))
            / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path.home()
            / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs",
        ]
        partial: str | None = None
        for sm_dir in start_menu_dirs:
            if not sm_dir.exists():
                continue
            try:
                for lnk in sm_dir.rglob("*.lnk"):
                    stem = lnk.stem.lower()
                    if stem == key:
                        return str(lnk)
                    if partial is None and key in stem:
                        partial = str(lnk)
            except OSError:
                continue
        if partial:
            return partial

        # 4. App Paths registry (HKLM then HKCU)
        try:
            import winreg
            for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
                try:
                    with winreg.OpenKey(root, base) as hive:
                        i = 0
                        while True:
                            try:
                                exe = winreg.EnumKey(hive, i)
                            except OSError:
                                break
                            i += 1
                            if key in exe.lower().removesuffix(".exe"):
                                try:
                                    with winreg.OpenKey(hive, exe) as sub:
                                        path, _ = winreg.QueryValueEx(sub, "")
                                        if path:
                                            return path
                                except OSError:
                                    continue
                except OSError:
                    continue
        except Exception as e:
            logger.debug("app path registry lookup failed error=%s", e)

        return None

    def _launch_windows(target: str) -> None:
        """Launch a resolved target (lnk file, exe path, URI scheme, or PATH command)."""
        lower = target.lower()
        if lower.endswith(".lnk"):
            os.startfile(target)  # type: ignore[attr-defined]
        elif lower.endswith(":") or "://" in lower:
            # URI scheme (ms-settings:, xbox:, ...) — needs shell start
            subprocess.Popen(
                f'start "" "{target}"', shell=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif os.path.isabs(target) and Path(target).exists():
            subprocess.Popen(
                [target],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                f'start "" "{target}"', shell=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

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
            if platform.system() == "Windows":
                resolved = await asyncio.to_thread(_resolve_app_windows, app_name)
                if resolved is None:
                    return (
                        f"Could not find an application named '{app_name}'. "
                        "It may not be installed, or may need its full path."
                    )
                _launch_windows(resolved)
                return f"Launched '{app_name}' (via {resolved})"
            subprocess.Popen(
                [app_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"Launched '{app_name}'"
        except Exception as e:
            return f"Failed to launch '{app_name}': {e}"

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
    @_offloaded
    def read_file(file_path: str) -> str:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return f"File not found: {path}"
        if not path.is_file():
            return f"Not a file: {path}"
        try:
            # Read only what can be returned; a huge file is never loaded whole.
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                content = handle.read(_READ_LIMIT_CHARS + 1)
            if len(content) > _READ_LIMIT_CHARS:
                return (content[:_READ_LIMIT_CHARS]
                        + f"\n\n... (truncated; file is {path.stat().st_size:,} bytes)")
            return content
        except Exception as e:
            return f"Error reading file: {e}"

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
    @_offloaded
    def write_file(file_path: str, content: str) -> str:
        path = Path(file_path).expanduser().resolve()
        try:
            _write_atomic(path, content)
            return f"Written {len(content)} chars to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

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
    @_offloaded
    def list_directory(path: str) -> str:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            return f"Directory not found: {path}"
        if not path.is_dir():
            return f"Not a directory: {path}"
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            lines = []
            for entry in entries[:200]:
                prefix = "[dir] " if entry.is_dir() else "[file]"
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
            return f"Error listing directory: {e}"

    # ── System Info ──

    @registry.tool(
        name="get_system_info",
        description="Get current system information (CPU, RAM, disk, OS, network).",
        parameters={
            "type": "object",
            "properties": {},
        },
    )
    @_offloaded
    def get_system_info() -> str:
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
            return f"Error getting system info: {e}"

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
    @_offloaded
    def open_url(url: str) -> str:
        import webbrowser
        try:
            if not webbrowser.open(url):
                return f"No browser accepted {url}; nothing was opened."
            return f"Opened {url} in browser"
        except Exception as e:
            return f"Error opening URL: {e}"

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
    @_offloaded
    def search_files(directory: str, pattern: str, content: str = "") -> str:
        import fnmatch
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            return f"Directory not found: {root}"
        wanted = pattern.lower()
        needle = content.lower()
        deadline = time.monotonic() + _SEARCH_TIME_BUDGET_S
        matches: list[str] = []
        timed_out = False

        def contains(path: Path) -> bool:
            try:
                if path.stat().st_size > _SEARCH_MAX_CONTENT_BYTES:
                    return False
                return needle in path.read_text(encoding="utf-8", errors="replace").lower()
            except OSError:
                return False

        for dirpath, _dirs, names in os.walk(root):
            for name in names:
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                if not fnmatch.fnmatch(name.lower(), wanted):
                    continue
                path = Path(dirpath) / name
                if needle and not contains(path):
                    continue
                matches.append(str(path))
                if len(matches) >= _SEARCH_MAX_RESULTS:
                    break
            else:
                continue
            break

        if not matches:
            note = f" (search stopped after {_SEARCH_TIME_BUDGET_S:g}s)" if timed_out else ""
            return f"No files matching '{pattern}' found in {root}{note}"
        result = "\n".join(matches)
        if len(matches) >= _SEARCH_MAX_RESULTS:
            result += f"\n... (capped at {_SEARCH_MAX_RESULTS} results)"
        if timed_out:
            result += f"\n... (search stopped after {_SEARCH_TIME_BUDGET_S:g}s; results may be incomplete)"
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
    @_offloaded
    def copy_file(source: str, destination: str) -> str:
        src = Path(source).expanduser().resolve()
        dst = Path(destination).expanduser().resolve()
        try:
            if src.is_dir():
                shutil.copytree(str(src), str(dst))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
            return f"Copied {src} -> {dst}"
        except Exception as e:
            return f"Copy failed: {e}"

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
    @_offloaded
    def move_file(source: str, destination: str) -> str:
        src = Path(source).expanduser().resolve()
        dst = Path(destination).expanduser().resolve()
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            return f"Moved {src} -> {dst}"
        except Exception as e:
            return f"Move failed: {e}"

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
        requires_confirmation=True,
    )
    @_offloaded
    def delete_file(path: str) -> str:
        p = Path(path).expanduser().resolve()
        try:
            if p.is_dir():
                shutil.rmtree(str(p))
                return f"Deleted directory: {p}"
            elif p.is_file():
                p.unlink()
                return f"Deleted file: {p}"
            else:
                return f"Path not found: {p}"
        except Exception as e:
            return f"Delete failed: {e}"

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
    @_offloaded
    def create_directory(path: str) -> str:
        p = Path(path).expanduser().resolve()
        try:
            p.mkdir(parents=True, exist_ok=True)
            return f"Created directory: {p}"
        except Exception as e:
            return f"Failed to create directory: {e}"

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
    @_offloaded
    def get_running_processes(filter_name: str = "") -> str:
        try:
            # cpu_percent is measured since the previous call per process, so
            # prime it and sample over a short window; a first call reads 0.0.
            for proc in psutil.process_iter():
                with contextlib.suppress(psutil.Error):
                    proc.cpu_percent(None)
            time.sleep(0.3)
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
            return f"Error: {e}"

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
        requires_confirmation=True,
    )
    @_offloaded
    def kill_process(target: str) -> str:
        try:
            # Try as PID first
            try:
                pid = int(target)
                proc = psutil.Process(pid)
                proc.terminate()
                return f"Terminated process {proc.name()} (PID {pid})"
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
                return f"Terminated: {', '.join(killed)}"
            return f"No process found matching '{target}'"
        except Exception as e:
            return f"Error: {e}"

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
            return "Volume control only supported on Windows"
        try:
            if level == -1:
                code, _, stderr = await _run_shell(
                    'powershell -c "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"',
                    timeout=10,
                )
                if code != 0:
                    return f"Could not toggle mute: {stderr.decode(errors='replace').strip()}"
                return "Toggled mute"
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
                code, _, stderr = await _run_shell(f'powershell -c "{ps_cmd}"', timeout=10)
                if code != 0:
                    return f"Could not set volume: {stderr.decode(errors='replace').strip()}"
                return f"Volume set to ~{level}%"
        except Exception as e:
            return f"Error setting volume: {e}"

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
    @_offloaded
    def take_screenshot(save_path: str = "") -> str:
        try:
            import pyautogui
            if not save_path:
                save_dir = Path.home() / ".jarvis"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(save_dir / "screenshot.png")
            img = pyautogui.screenshot()
            img.save(save_path)
            return f"Screenshot saved to {save_path}"
        except ImportError:
            return "pyautogui not installed. Run: pip install pyautogui"
        except Exception as e:
            return f"Screenshot failed: {e}"

    # ── Clipboard ──

    @registry.tool(
        name="clipboard_read",
        description="Read the current clipboard contents.",
        parameters={"type": "object", "properties": {}},
    )
    @_offloaded
    def clipboard_read() -> str:
        try:
            import pyperclip
            content = pyperclip.paste()
            return content if content else "(clipboard is empty)"
        except ImportError:
            return "pyperclip not installed. Run: pip install pyperclip"
        except Exception as e:
            return f"Error: {e}"

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
    @_offloaded
    def clipboard_write(text: str) -> str:
        try:
            import pyperclip
            pyperclip.copy(text)
            return f"Copied {len(text)} chars to clipboard"
        except ImportError:
            return "pyperclip not installed. Run: pip install pyperclip"
        except Exception as e:
            return f"Error: {e}"

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
            import json as _json

            import httpx
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
            return f"HTTP error: {e}"

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
    @_offloaded
    def scheduler_create(task_name: str, command: str, delay_seconds: int = 60) -> str:
        try:
            # schtasks /st expects a clock time (HH:MM) and /sd a date —
            # compute them from the requested delay.
            run_at = datetime.now() + timedelta(seconds=max(60, delay_seconds))
            start_time = run_at.strftime("%H:%M")
            start_date = run_at.strftime("%m/%d/%Y")
            result = subprocess.run(
                ["schtasks", "/create", "/tn", f"Jarvis_{task_name}", "/tr", command,
                 "/sc", "once", "/st", start_time, "/sd", start_date, "/f"],
                shell=False, capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return f"Scheduled '{task_name}' to run at {start_time} on {start_date}"
            return f"Failed: {result.stderr.strip() or result.stdout.strip()}"
        except Exception as e:
            return f"Scheduler error: {e}"

    @registry.tool(
        name="scheduler_list",
        description="List scheduled Jarvis tasks.",
        parameters={"type": "object", "properties": {}},
    )
    @_offloaded
    def scheduler_list() -> str:
        try:
            result = subprocess.run(
                'schtasks /query /fo csv /v | findstr "Jarvis_"',
                shell=True, capture_output=True, text=True, timeout=30,
            )
            return result.stdout or "No Jarvis tasks scheduled."
        except Exception as e:
            return f"List error: {e}"

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
    @_offloaded
    def scheduler_delete(task_name: str) -> str:
        try:
            result = subprocess.run(
                ["schtasks", "/delete", "/tn", f"Jarvis_{task_name}", "/f"],
                shell=False, capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return f"Deleted task: {task_name}"
            return f"Failed: {result.stderr}"
        except Exception as e:
            return f"Delete error: {e}"

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
            import re
            import urllib.parse

            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query, "kl": "us-en"},
                    timeout=15.0,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                text = resp.text
                results = []

                # Find all result blocks
                titles = list(re.finditer(
                    r'class="result__a" href="([^"]+)"[^>]*>(.*?)</a>', text
                ))
                snippets = re.findall(
                    r'class="result__snippet"[^>]*>(.*?)</a>', text
                )

                for i, tmatch in enumerate(titles):
                    if i >= num_results or i >= len(snippets):
                        break

                    raw_url = tmatch.group(1)
                    # Strip HTML tags from title
                    title = re.sub(r'<[^>]+>', '', tmatch.group(2)).strip()
                    # Strip HTML tags from snippet
                    snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()

                    # Decode DuckDuckGo redirect URLs
                    if raw_url.startswith("//duckduckgo.com/l/?uddg="):
                        real_url = urllib.parse.unquote(
                            raw_url.split("uddg=")[1]
                        )
                    else:
                        real_url = raw_url
                    # Strip DuckDuckGo tracking param
                    real_url = re.split(r'[&]amp;rut=', real_url)[0]
                    real_url = re.split(r'[&]rut=', real_url)[0]

                    results.append(
                        f"• {title}\n  {snippet[:200]}\n  {real_url}"
                    )

                if results:
                    return f"Results for '{query}':\n\n" + "\n\n".join(results)
                return f"No results found for '{query}'"
        except Exception as e:
            return f"Search error: {e}"

    # ── System Uptime ──

    @registry.tool(
        name="get_uptime",
        description="Get how long the system has been running since boot.",
        parameters={"type": "object", "properties": {}},
    )
    async def get_uptime() -> str:
        try:
            booted = datetime.fromtimestamp(psutil.boot_time())
            days, rest = divmod(int((datetime.now() - booted).total_seconds()), 86400)
            hours, rest = divmod(rest, 3600)
            return (f"Uptime: {days}d {hours}h {rest // 60}m "
                    f"(booted {booted:%Y-%m-%d %H:%M})")
        except Exception as e:
            return f"Error: {e}"

    # Register vision tools if LLM client is available
    if llm_client is not None:
        from jarvis.tools.vision_tools import register_vision_tools
        register_vision_tools(registry, llm_client)

    logger.debug("system tools registered count=%d", len(registry))
