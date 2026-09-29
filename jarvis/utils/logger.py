"""
JARVIS Logging Framework
=========================
Rich-formatted console logging with file rotation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# Force UTF-8 on Windows to prevent charmap codec errors with emojis
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_LOG_FORMAT = "%(message)s"
_DATE_FORMAT = "%H:%M:%S"

console = Console(force_terminal=True, legacy_windows=False)


def setup_logging(
    level: str = "INFO",
    log_dir: str | None = None,
) -> None:
    """
    Configure the global logging system.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files. None = console only.
    """

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Rich console handler (pretty terminal output)
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    rich_handler.setLevel(log_level)

    handlers: list[logging.Handler] = [rich_handler]

    # File handler (optional, with rotation)
    if log_dir:
        log_path = Path(log_dir).expanduser().resolve()
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / "jarvis.log"

        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )

    # Silence noisy libraries
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "uvicorn.access",
                   "mem0", "mem0.memory", "mem0.memory.main"):
        logging.getLogger(noisy).setLevel(logging.WARNING)



def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger; startup configures handlers explicitly.

    Usage:
        from jarvis.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Hello from %s", __name__)
    """
    return logging.getLogger(name)
