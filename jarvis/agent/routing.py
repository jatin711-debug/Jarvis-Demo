"""
Capability routing
==================
Decide which single tool surface a request needs, so the model is offered a
small, relevant menu instead of every tool. Surfaces are mutually exclusive:

    small_talk  greetings; fast path with no tools, memory, or vision
    chat        questions that need no tool; full path with an empty tool menu
    browser     web tasks (Playwright tools)
    desktop     window/app tasks (Cua tools)
    system      files, shell, clock, clipboard, ... (subset chosen by capability)

Routing is tool exposure, not authorization: the permission gateway still
checks every call. Unknown tools are not offered until a capability route
is defined for them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class Surface(StrEnum):
    SMALL_TALK = "small_talk"
    CHAT = "chat"
    SYSTEM = "system"
    BROWSER = "browser"
    DESKTOP = "desktop"


@dataclass(frozen=True)
class Route:
    """The routing decision for one request."""

    surface: Surface
    system_tools: frozenset[str] = frozenset()
    groups: tuple[str, ...] = ()
    vision: bool = False

    def describe(self) -> str:
        """One-line summary for logs."""
        parts = [f"surface={self.surface.value}"]
        if self.groups:
            parts.append(f"groups={','.join(self.groups)}")
        if self.system_tools:
            parts.append(f"system_tools={len(self.system_tools)}")
        if self.vision:
            parts.append("vision=yes")
        return " ".join(parts)


# ── Capability tables ───────────────────────────────────────────────

SYSTEM_CAPABILITIES = {
    "clock": {"get_datetime"},
    "system": {"get_system_info", "get_uptime", "get_running_processes"},
    "files_read": {"read_file", "list_directory", "search_files"},
    "files_write": {"write_file", "copy_file", "move_file", "delete_file", "create_directory"},
    "shell": {"run_command"},
    "apps": {"open_application"},
    "screen": {"take_screenshot"},
    "screen_elements": {"find_ui_elements"},
    "documents": {"read_pdf", "read_spreadsheet", "read_image"},
    "clipboard": {"clipboard_read", "clipboard_write"},
    "volume": {"set_volume"},
    "schedule": {"scheduler_create", "scheduler_list", "scheduler_delete"},
    "network": {"http_request", "open_url", "web_search"},
    "process": {"kill_process"},
}

# Ordered (group, pattern) pairs; "files_write" additionally requires "files_read".
_SYSTEM_PATTERNS = (
    ("clock", r"\b(time|date|day|clock|timezone)\b"),
    ("system", r"\b(system info|system status|cpu|ram|memory usage|uptime|processes|running apps)\b"),
    ("files_read", r"\b(files?|folders?|directories|directory|paths?|documents?|project|codebase|"
                   r"source code|repository|repo)\b"),
    ("files_write", r"\b(write|create|edit|modify|rename|move|copy|delete|remove|save|build|implement)\b"),
    ("shell", r"\b(command|terminal|powershell|shell|script|run tests|pytest|execute|compile|build)\b"),
    ("apps", r"\b(open|launch|focus|switch|type|click|window|app|application|keystroke)\b"),
    ("screen", r"\b(screenshot|capture screen)\b"),
    ("screen_elements", r"\b(find|locate|where is)\b.*\b(buttons?|icons?|menus?|elements?|tabs?|fields?)\b"),
    ("documents", r"\b(pdf|spreadsheet|excel|xlsx|csv|image|picture|photo|png|jpe?g)\b"),
    ("clipboard", r"\b(clipboard|paste|copied text)\b"),
    ("volume", r"\b(volume|mute|unmute)\b"),
    ("schedule", r"\b(schedule|remind|reminder|timer|alarm)\b"),
    ("network", r"\b(http|api request|fetch url|web search|online search|weather)\b"),
    ("process", r"\b(kill|terminate|stop process)\b"),
)

_WEB = re.compile(
    r"\b(browse|browser|web|website|search|google|youtube|url|http|navigate|"
    r"open.*site|go to|look up|surf|online|internet|wiki|reddit|github|stackoverflow|"
    r"facebook|twitter|instagram|linkedin|amazon|netflix|spotify)\b",
    re.IGNORECASE,
)
# Local-file wording without any explicit web wording is not a web task.
_LOCAL_FILES = re.compile(
    r"\b(files?|folders?|directories|directory|paths?|codebase|repository|repo)\b", re.IGNORECASE)
_EXPLICIT_WEB = re.compile(r"\b(browser|web|website|online|internet|http|url|github)\b", re.IGNORECASE)

_DESKTOP = re.compile(
    r"\b(desktop|window|notepad|calculator|paint|dialog|button|type|click|"
    r"open (?:an? )?app|fill|save (?:it|this|the))\b", re.IGNORECASE)
# References that only make sense while a desktop task is in progress.
_DESKTOP_FOLLOW_UP = re.compile(r"\b(it|that|this|there|next|again)\b", re.IGNORECASE)

_VISION = re.compile(
    r"\b(screen|screenshot|what do you see|what's on my screen|"
    r"this window|this dialog|this error|this message|"
    r"look at|can you see|show me|"
    r"click the|where is|find the|locate the|"
    r"button|icon|menu bar|taskbar|desktop)\b",
    re.IGNORECASE,
)


# ── Predicates ──────────────────────────────────────────────────────

def is_small_talk(text: str) -> bool:
    normalized = re.sub(r"[^\w\s']", "", text.casefold()).strip()
    normalized = " ".join(normalized.split())
    normalized = re.sub(r"^(?:hey|hi|hello)(?: jarvis)?\s+", "", normalized)
    normalized = re.sub(r"\s+jarvis$", "", normalized)
    return normalized in {
        "hi", "hello", "hey", "hello jarvis", "hey jarvis", "hi jarvis",
        "how are you", "how are you doing", "how's it going", "hows it going",
        "how you doing", "how you doing today", "how are you doing today",
        "how are you today", "how're you", "how're you doing", "how you been",
        "how have you been", "whats up", "what's up", "you doing okay",
        "good morning", "good afternoon", "good evening", "thanks", "thank you",
    }


def needs_web(text: str) -> bool:
    """True when the request is about the web rather than local files."""
    if _LOCAL_FILES.search(text) and not _EXPLICIT_WEB.search(text):
        return False
    return bool(_WEB.search(text))


def needs_vision(text: str) -> bool:
    """True when the request implies looking at the screen."""
    return bool(_VISION.search(text))


def select_system_tools(text: str, available: set[str]) -> tuple[set[str], tuple[str, ...]]:
    """Choose the system tools that match the request's capabilities.

    Returns ``(tool_names, matched_groups)``, limited to ``available`` tools.
    """
    message = text.casefold()
    groups = [name for name, pattern in _SYSTEM_PATTERNS if re.search(pattern, message)]
    if "files_write" in groups and "files_read" not in groups:
        groups.remove("files_write")  # write verbs alone are too ambiguous to offer file writes
    selected: set[str] = set()
    for group in groups:
        selected.update(SYSTEM_CAPABILITIES[group] & available)
    return selected, tuple(groups)


# ── Router ──────────────────────────────────────────────────────────

def route_request(
    text: str,
    *,
    system_tools: set[str],
    browser_available: bool = False,
    desktop_available: bool = False,
    desktop_active: bool = False,
) -> Route:
    """
    Pick the tool surface for ``text``.

    ``desktop_active`` marks an ongoing desktop task, which lets short
    follow-ups ("do that again") stay on the desktop surface.
    """
    if is_small_talk(text):
        return Route(Surface.SMALL_TALK)

    web = needs_web(text)
    if desktop_available and not web:
        if _DESKTOP.search(text) or (desktop_active and _DESKTOP_FOLLOW_UP.search(text)):
            return Route(Surface.DESKTOP)
    if browser_available and web:
        return Route(Surface.BROWSER, vision=needs_vision(text))

    tools, groups = select_system_tools(text, system_tools)
    return Route(Surface.SYSTEM if tools else Surface.CHAT, frozenset(tools), groups,
                 vision=needs_vision(text))
