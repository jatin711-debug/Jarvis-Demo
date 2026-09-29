"""Dispatch-time, config-scoped permission decisions for built-in tools.

This is an application policy, not an OS sandbox: handlers still need to keep
their targets stable between validation and use, and shell commands remain
powerful when a profile explicitly permits them.
"""

from __future__ import annotations

import hashlib
import json
import ntpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jarvis.utils.config import SecurityConfig

_RM_ROOT = re.compile(r"\brm\s+-\w*r\w*f\w*\s+(/|~|\*|[a-z]:[\\/]?)\s*$", re.IGNORECASE)
_DEL_SYS = re.compile(
    r"\b(del|erase|rd|rmdir)\b[^|&]*/s[^|&]*(c:\\|c:/|%systemdrive%|\\windows\b|\\system32\b)",
    re.IGNORECASE,
)
_FORMAT_DISK = re.compile(r"\bformat\s+[a-z]:", re.IGNORECASE)


def catastrophic_reason(command: str) -> str | None:
    """Hard stop for a few catastrophic shell operations, regardless of profile."""
    c = command.strip()
    if _FORMAT_DISK.search(c):
        return "formats a disk"
    if re.search(r"\bdiskpart\b", c, re.IGNORECASE):
        return "modifies disk partitions"
    if _RM_ROOT.search(c):
        return "recursively deletes a root/home directory"
    if _DEL_SYS.search(c):
        return "recursively deletes system paths"
    if re.search(r"\bbcdedit\b", c, re.IGNORECASE):
        return "modifies boot configuration"
    return None

READ_TOOLS = frozenset({
    "read_file", "list_directory", "search_files", "get_system_info", "get_running_processes",
    "get_datetime", "get_uptime", "clipboard_read", "scheduler_list", "web_search",
    "browser_read_page", "browser_get_info", "browser_list_tabs", "desktop_windows",
    "desktop_observe", "read_pdf", "read_spreadsheet", "read_image", "find_ui_elements",
})
LOCAL_TOOLS = frozenset({
    "write_file", "copy_file", "move_file", "delete_file", "create_directory", "set_volume",
    "take_screenshot", "clipboard_write", "browser_type", "browser_scroll", "browser_screenshot",
    "browser_back", "browser_forward", "browser_switch_tab", "browser_close", "desktop_click",
    "desktop_type", "desktop_key",
})
CONTROL_TOOLS = frozenset({
    "run_command", "open_application", "kill_process", "scheduler_create", "scheduler_delete",
    "desktop_launch",
})
EXTERNAL_TOOLS = frozenset({
    "open_url", "http_request", "browser_navigate", "browser_search", "browser_click",
    "browser_new_tab", "browser_youtube_search",
})
KNOWN_TOOLS = READ_TOOLS | LOCAL_TOOLS | CONTROL_TOOLS | EXTERNAL_TOOLS

PATH_FIELDS = {
    "read_file": ("file_path",), "write_file": ("file_path",),
    "list_directory": ("path",), "search_files": ("directory",),
    "copy_file": ("source", "destination"), "move_file": ("source", "destination"),
    "delete_file": ("path",), "create_directory": ("path",),
    "take_screenshot": ("save_path",), "browser_screenshot": ("save_path",),
    "read_pdf": ("path",), "read_spreadsheet": ("path",), "read_image": ("path",),
}
DEFAULT_PATHS = {"take_screenshot": Path.home() / ".jarvis" / "screenshot.png",
                 "browser_screenshot": Path.home() / ".jarvis" / "browser_screenshot.png"}
CONFIRM_TOOLS = frozenset({
    "write_file", "copy_file", "move_file", "delete_file", "create_directory",
    "take_screenshot", "browser_screenshot", "clipboard_write", "run_command",
    "open_application", "kill_process", "scheduler_create", "scheduler_delete",
    "desktop_launch", "desktop_click", "desktop_type", "desktop_key", "browser_click",
    "browser_type",
})


@dataclass(frozen=True)
class PermissionDecision:
    effect: str
    target: str
    action_id: str
    requires_confirmation: bool
    reason: str | None = None

    @property
    def allowed(self) -> bool:
        return self.reason is None


class PermissionGateway:
    def __init__(self, security: SecurityConfig) -> None:
        self.security = security

    def _path(self, raw: str, *, destructive: bool = False) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("An exact path is required")
        try:
            path = Path(raw).expanduser().resolve()
            roots = [Path(item).expanduser().resolve()
                     for item in self.security.allowed_directories]
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid path: {exc}") from exc
        if not any(path == root or path.is_relative_to(root) for root in roots):
            raise ValueError(f"Path is outside security.allowed_directories: {path}")
        protected = (Path.home().resolve(), Path.cwd().resolve())
        if destructive and (path == Path(path.anchor) or path in roots
                            or any(root == path or root.is_relative_to(path) for root in protected)):
            raise ValueError(f"Refusing a broad destructive target: {path}")
        return path

    def _url(self, raw: str) -> str:
        if not isinstance(raw, str):
            raise ValueError("An exact URL is required")
        try:
            parsed = urlsplit(raw.strip())
            host = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"Invalid URL: {exc}") from exc
        if (parsed.scheme not in {"http", "https"} or not host
                or parsed.username or parsed.password):
            raise ValueError("Only absolute http(s) URLs without embedded credentials are allowed")
        if (self.security.allowed_hosts is not None
                and host.casefold() not in {h.casefold() for h in self.security.allowed_hosts}):
            raise ValueError(f"Host is outside security.allowed_hosts: {host}")
        return f"{parsed.scheme}://{host}{':' + str(port) if port else ''}"

    def _app(self, raw: str) -> str:
        if not isinstance(raw, str) or not raw.strip() or re.search(r'["\r\n;&|<>%^!]', raw):
            raise ValueError("An exact application name or path is required")
        name = raw.strip()
        if ntpath.isabs(name) or "/" in name or "\\" in name:
            return str(self._path(name))
        if (self.security.allowed_applications is not None
                and name.casefold() not in {a.casefold() for a in self.security.allowed_applications}):
            raise ValueError(f"App is outside security.allowed_applications: {name}")
        return name

    def _command(self, raw: str) -> str:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("An exact command is required")
        command = raw.strip()
        reason = catastrophic_reason(command)
        if reason:
            raise ValueError(f"Command {reason}")
        if any(block.casefold() in command.casefold() for block in self.security.blocked_commands if block):
            raise ValueError("Command matches security.blocked_commands")
        if self.security.command_whitelist:
            # A first-executable whitelist cannot safely classify shell
            # compositions or interpreter arguments as independent commands.
            if re.search(r"[\r\n;&|><`]|\$\(|%[^%]+%", command):
                raise ValueError("Shell compositions are not allowed with command_whitelist enabled")
            executable = ntpath.basename(command.split()[0].strip("\"'")).casefold()
            executable = re.sub(r"\.(exe|cmd|bat|ps1|com)$", "", executable)
            if executable in {"cmd", "powershell", "pwsh", "bash", "sh", "python", "py", "node"}:
                raise ValueError("Command interpreters cannot be constrained by a first-executable whitelist")
            if executable not in {a.casefold() for a in self.security.allowed_commands}:
                raise ValueError(f"Command is outside security.allowed_commands: {executable}")
        return command

    def inspect(self, name: str, args: dict[str, Any], *, tool_requires_confirmation: bool = False
                ) -> PermissionDecision:
        effect = ("read" if name in READ_TOOLS else "local_edit" if name in LOCAL_TOOLS
                  else "system_control" if name in CONTROL_TOOLS else "external"
                  if name in EXTERNAL_TOOLS else "unknown")
        action_id = hashlib.sha256(json.dumps(
            {"tool": name, "args": args}, sort_keys=True, default=str,
            separators=(",", ":")).encode()).hexdigest()[:16]
        target = name
        reason = None
        confirm = name in CONFIRM_TOOLS or tool_requires_confirmation
        try:
            if effect == "unknown":
                raise ValueError("Tool has no permission classification")
            if name in PATH_FIELDS:
                targets = []
                for field in PATH_FIELDS[name]:
                    raw = args.get(field) or str(DEFAULT_PATHS.get(name, ""))
                    destructive = name == "delete_file" or (name == "move_file" and field == "source")
                    path = self._path(raw, destructive=destructive)
                    targets.append(f"{field}={path}")
                target = ", ".join(targets)
            elif name in {"open_application", "desktop_launch"}:
                target = self._app(args.get("app_name") if name == "open_application"
                                   else args.get("name"))
            elif name in {"open_url", "browser_navigate", "browser_new_tab", "http_request"}:
                url = args.get("url", "")
                if url:
                    self._url(url)
                target = url.strip() if url else "new blank tab"
                if name != "browser_new_tab" and not url:
                    raise ValueError("An exact URL is required")
                if name == "http_request":
                    method = str(args.get("method", "GET")).upper()
                    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                        raise ValueError("Unsupported HTTP method")
                    target = f"{method} {target}"
                    confirm = method != "GET" or bool(args.get("headers")) or bool(args.get("body"))
            elif name in {"browser_search", "web_search"}:
                target = f"search query {str(args.get('query', ''))[:100]}"
                if self.security.allowed_hosts is not None:
                    self._url("https://www.google.com" if name == "browser_search"
                              else "https://html.duckduckgo.com")
            elif name == "browser_youtube_search":
                target = f"YouTube query {str(args.get('query', ''))[:100]}"
                self._url("https://www.youtube.com")
            elif name == "run_command":
                target = self._command(args.get("command"))
            elif name == "scheduler_create":
                task_name = str(args.get("task_name", ""))
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", task_name):
                    raise ValueError("Scheduled task name must be 1–64 letters, digits, '_' or '-'")
                target = f"task={args.get('task_name')}; command={self._command(args.get('command'))}"
            elif name == "kill_process":
                target = str(args.get("target", ""))
                if not target.isdecimal():
                    raise ValueError("Process termination requires an exact numeric PID")
            elif name.startswith("desktop_") and name not in {"desktop_windows", "desktop_launch"}:
                target = f"pid={args.get('pid')}, window_id={args.get('window_id')}"
            elif name.startswith("browser_") and self.security.allowed_hosts is not None:
                # Active-page actions need a persisted URL handoff before a
                # host allowlist can be enforced. Fail closed for now.
                if name not in {"browser_list_tabs", "browser_close"}:
                    raise ValueError("Active browser URL is unavailable to the host allowlist")
            elif name == "scheduler_delete":
                target = str(args.get("task_name"))
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", target):
                    raise ValueError("Scheduled task name must be 1–64 letters, digits, '_' or '-'")
            elif name == "clipboard_write":
                target = f"clipboard ({len(str(args.get('text', '')))} chars)"
            elif name == "browser_click":
                target = f"active browser page, selector={args.get('selector')}"
            elif name == "browser_type":
                target = f"active browser page, selector={args.get('selector')}"
                confirm = bool(args.get("press_enter", False)) or confirm
        except ValueError as exc:
            reason = str(exc)
        return PermissionDecision(effect, target, action_id,
                                  self.security.confirm_dangerous_ops and confirm, reason)
