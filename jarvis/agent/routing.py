"""Small, explicit capability selection for ordinary single-surface requests."""

import re

SYSTEM_CAPABILITIES = {
    "clock": {"get_datetime"},
    "system": {"get_system_info", "get_uptime", "get_running_processes"},
    "files_read": {"read_file", "list_directory", "search_files"},
    "files_write": {"write_file", "copy_file", "move_file", "delete_file", "create_directory"},
    "shell": {"run_command"},
    "apps": {"open_application", "focus_window", "window_management", "type_text",
             "send_keystrokes"},
    "screen": {"take_screenshot"},
    "clipboard": {"clipboard_read", "clipboard_write"},
    "volume": {"set_volume"},
    "schedule": {"scheduler_create", "scheduler_list", "scheduler_delete"},
    "network": {"http_request", "open_url", "web_search"},
    "process": {"kill_process"},
}
def select_system_tools(text: str, available: set[str], *, browser: bool = False,
                        desktop: bool = False) -> set[str]:
    """Offer a bounded set of tools for the user's current capability request.

    This is tool exposure, not the permission gateway. Unknown tools are not
    offered until a capability route is defined for them.
    """
    if browser or desktop:
        return set()
    selected: set[str] = set()
    message = text.casefold()

    def matches(pattern: str) -> bool:
        return bool(re.search(pattern, message))

    groups = set()
    if matches(r"\b(time|date|day|clock|timezone)\b"):
        groups.add("clock")
    if matches(r"\b(system info|system status|cpu|ram|memory usage|uptime|processes|running apps)\b"):
        groups.add("system")
    if matches(r"\b(files?|folders?|directories|directory|paths?|documents?|project|codebase|source code|repository|repo)\b"):
        groups.add("files_read")
    if matches(r"\b(write|create|edit|modify|rename|move|copy|delete|remove|save|build|implement)\b"):
        if "files_read" in groups:
            groups.add("files_write")
    if matches(r"\b(command|terminal|powershell|shell|script|run tests|pytest|execute|compile|build)\b"):
        groups.add("shell")
    if matches(r"\b(open|launch|focus|switch|type|click|window|app|application|keystroke)\b"):
        groups.add("apps")
    if matches(r"\b(screenshot|capture screen)\b"):
        groups.add("screen")
    if matches(r"\b(clipboard|paste|copied text)\b"):
        groups.add("clipboard")
    if matches(r"\b(volume|mute|unmute)\b"):
        groups.add("volume")
    if matches(r"\b(schedule|remind|reminder|timer|alarm)\b"):
        groups.add("schedule")
    if matches(r"\b(http|api request|fetch url|web search|online search|weather)\b"):
        groups.add("network")
    if matches(r"\b(kill|terminate|stop process)\b"):
        groups.add("process")
    for group in groups:
        selected.update(SYSTEM_CAPABILITIES[group] & available)
    return selected


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
