"""
JARVIS logging
==============
Readable console output on stderr plus a full-detail rotating log file.

Conventions for log messages (see docs/logging.md):

* Start with a short lower-case event word, then ``key=value`` fields:
  ``tool ok name=open_application ms=340``.
* One line per event. No emoji, no banners, no markup.
* INFO is the story of a turn (heard, route, tool, reply, latency).
  DEBUG is diagnostics; it goes to the file always and to the console with
  ``--verbose``.

Every record emitted while a turn is active carries its turn number
(``t7``), so the lines of one interaction can be grepped from the file.
"""

from __future__ import annotations

import contextvars
import itertools
import json
import logging
import os
import sys
import uuid
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_turn_var: contextvars.ContextVar[int | None] = contextvars.ContextVar("jarvis_turn", default=None)
_turn_counter = itertools.count(1)
_run_id = uuid.uuid4().hex


def begin_turn() -> int:
    """Start a new turn; log records in this context are tagged with its number."""
    turn = next(_turn_counter)
    _turn_var.set(turn)
    return turn


def end_turn() -> None:
    _turn_var.set(None)


def current_turn() -> int | None:
    return _turn_var.get()


# ──────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────

# Logger name -> short console label. Unlisted jarvis modules fall back to the
# last dotted segment; third-party loggers keep their own name.
_SHORT_NAMES = {
    "__main__": "main",
    "jarvis.core.orchestrator": "orchestrator",
    "jarvis.agent.react_agent": "agent",
    "jarvis.agent.llm_client": "llm",
    "jarvis.agent.tool_registry": "tools",
    "jarvis.agent.permissions": "permissions",
    "jarvis.tools.system_tools": "system",
    "jarvis.tools.browser_tools": "browser",
    "jarvis.tools.vision_tools": "vision",
    "jarvis.voice.implementations": "voice",
    "jarvis.voice.engines": "voice",
    "jarvis.voice.search_monitor": "voice",
    "jarvis.memory.manager": "memory",
    "jarvis.desktop.cua": "desktop",
    "jarvis.tasks.runner": "tasks",
    "jarvis.tasks.store": "tasks",
    "jarvis.utils.deadlines": "deadline",
}
_LABEL_WIDTH = 12

_LEVEL_TAGS = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO ",
    logging.WARNING: "WARN ",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRIT ",
}
_LEVEL_COLORS = {
    logging.DEBUG: "2",       # dim
    logging.INFO: "32",       # green
    logging.WARNING: "33",    # yellow
    logging.ERROR: "31",      # red
    logging.CRITICAL: "1;31",  # bold red
}

# Third-party loggers that are chatty at INFO or below.
_QUIET_LOGGERS = {
    logging.WARNING: (
        "httpx", "httpcore", "urllib3", "asyncio", "chromadb", "mem0", "faster_whisper",
        "numexpr", "NumExpr", "transformers", "torch", "tensorflow", "playwright", "PIL",
        "filelock", "fsspec", "matplotlib", "openwakeword", "onnxruntime", "sounddevice",
        "phonemizer", "markdown_it", "multipart",
    ),
    # Emits an unauthenticated-request notice on every model download.
    logging.ERROR: ("huggingface_hub",),
}


def _label(name: str) -> str:
    short = _SHORT_NAMES.get(name)
    if short is None:
        short = name.rsplit(".", 1)[-1] if name.startswith("jarvis.") else name
    return short[:_LABEL_WIDTH].ljust(_LABEL_WIDTH)


class _TurnFilter(logging.Filter):
    """Attach the active turn number to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        turn = _turn_var.get()
        record.turn = f"t{turn}" if turn is not None else "-"
        return True


class ConsoleFormatter(logging.Formatter):
    """``12:00:01 INFO  agent        t3 tool ok name=...`` with optional color."""

    def __init__(self, *, color: bool) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        stamp = self.formatTime(record, self.datefmt)
        level = _LEVEL_TAGS.get(record.levelno, record.levelname[:5].ljust(5))
        turn = getattr(record, "turn", "-")
        turn_col = f"{turn:<4}" if turn != "-" else "    "
        message = record.getMessage()
        if self._color:
            stamp = f"\x1b[2m{stamp}\x1b[0m"
            level = f"\x1b[{_LEVEL_COLORS.get(record.levelno, '0')}m{level}\x1b[0m"
            turn_col = f"\x1b[36m{turn_col}\x1b[0m"
            if record.levelno == logging.DEBUG:
                message = f"\x1b[2m{message}\x1b[0m"
        line = f"{stamp} {level} {_label(record.name)} {turn_col} {message}"
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            line = f"{line}\n{record.exc_text}"
        return line


class FileFormatter(logging.Formatter):
    """Grep-friendly, fixed-column format with full logger names."""

    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(turn)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class JsonFormatter(logging.Formatter):
    """One JSON object per event for log collectors; exceptions stay in the event."""

    def format(self, record: logging.LogRecord) -> str:
        event = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": _run_id,
            "turn": getattr(record, "turn", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            event["stack"] = self.formatStack(record.stack_info)
        return json.dumps(event, ensure_ascii=False)


# ──────────────────────────────────────────────
# Terminal handling
# ──────────────────────────────────────────────

def configure_stdio() -> None:
    """Make stdout/stderr UTF-8 so spoken text and paths never crash a print."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _enable_windows_vt() -> bool:
    """Turn on ANSI escape processing for the Windows console (stderr)."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-12)  # STD_ERROR_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    if not (hasattr(stream, "isatty") and stream.isatty()):
        return False
    return _enable_windows_vt() if sys.platform == "win32" else True


# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_dir: str | None = None, *,
                  file_format: str = "text") -> None:
    """
    Configure the root logger.

    Args:
        level: Console level (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for ``jarvis.log``; the file always records DEBUG.
                 None = console only.
        file_format: ``text`` or newline-delimited ``json`` for the file.
    """
    if file_format not in {"text", "json"}:
        raise ValueError("file_format must be 'text' or 'json'")
    console_level = getattr(logging, level.upper(), logging.INFO)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(ConsoleFormatter(color=_supports_color(sys.stderr)))
    # Warnings routed from the warnings module are diagnostics, not events.
    console.addFilter(lambda record: record.name != "py.warnings" or console_level <= logging.DEBUG)

    handlers: list[logging.Handler] = [console]

    if log_dir:
        log_path = Path(log_dir).expanduser().resolve()
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / "jarvis.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter() if file_format == "json" else FileFormatter())
        handlers.append(file_handler)

    for handler in handlers:
        handler.addFilter(_TurnFilter())

    logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)
    logging.captureWarnings(True)

    for quiet_level, names in _QUIET_LOGGERS.items():
        for name in names:
            logging.getLogger(name).setLevel(quiet_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger; ``setup_logging`` configures handlers once at startup.

    Usage:
        from jarvis.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("started port=%d", port)
    """
    return logging.getLogger(name)
