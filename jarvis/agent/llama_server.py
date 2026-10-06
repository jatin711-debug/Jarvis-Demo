"""
Launching a local ``llama-server`` (llama.cpp) for the ``llamacpp`` provider.

Jarvis starts the server only when ``llm.server.model`` is set and nothing is
answering at ``llm.base_url`` yet; a server you started yourself is reused and
left running. A server Jarvis started is stopped when Jarvis shuts down.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import psutil

from jarvis.utils.config import LLMConfig, expand_path
from jarvis.utils.logger import get_logger

logger = get_logger(__name__)

_STOP_GRACE_S = 5.0
_LOG_TAIL_BYTES = 2000


def server_address(base_url: str) -> tuple[str, int]:
    """Host and port a server must listen on for ``base_url`` to reach it."""
    parsed = urlparse(base_url)
    return parsed.hostname or "127.0.0.1", parsed.port or 8080


def resolve_executable(command: str) -> str | None:
    """The llama-server executable for ``command`` (a name on PATH or a path), or None."""
    found = shutil.which(command)
    if found:
        return found
    path = expand_path(command)
    return str(path) if path.is_file() else None


def build_command(config: LLMConfig) -> list[str]:
    """The llama-server argv for ``config``; raises RuntimeError when it cannot be built."""
    server = config.server
    if not server.model:
        raise RuntimeError("llm.server.model is not set")
    executable = resolve_executable(server.command)
    if executable is None:
        raise RuntimeError(f"llama-server not found ({server.command!r}). Install llama.cpp "
                           "or set llm.server.command to its full path")
    model = expand_path(server.model)
    if not model.is_file():
        raise RuntimeError(f"model file not found: {model}")
    host, port = server_address(config.base_url)
    # User flags come first so the flags Jarvis depends on (model, address, context) win.
    command = [executable, *server.args, "-m", str(model), "--host", host, "--port", str(port),
               "-c", str(config.context_window), "-np", "1", "--alias", config.model]
    if server.mmproj:
        projector = expand_path(server.mmproj)
        if not projector.is_file():
            raise RuntimeError(f"multimodal projector not found: {projector}")
        command += ["--mmproj", str(projector)]
    return command


class LlamaServer:
    """One llama-server child process, started and stopped by Jarvis."""

    def __init__(self, config: LLMConfig, log_path: Path) -> None:
        self._config = config
        self._log_path = log_path
        self._process: subprocess.Popen[bytes] | None = None
        self._log = None

    async def start(self) -> None:
        command = build_command(self._config)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("starting llama-server model=%s log=%s",
                    Path(self._config.server.model or "").name, self._log_path)
        self._log = open(self._log_path, "ab")  # noqa: SIM115 - closed in stop()
        flags = 0
        if sys.platform == "win32":
            # No console window, and Ctrl+C in Jarvis's console stops Jarvis, which then stops us.
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            self._process = await asyncio.to_thread(
                subprocess.Popen, command, stdin=subprocess.DEVNULL, stdout=self._log,
                stderr=subprocess.STDOUT, creationflags=flags)
        except OSError as exc:
            self._close_log()
            raise RuntimeError(f"cannot start llama-server: {exc}") from exc

    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def exit_code(self) -> int | None:
        """The exit code of a process that has ended, else None."""
        return self._process.poll() if self._process is not None else None

    async def wait_exit(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for the process to end; True when it is not running.

        A dropped connection can be seen a moment before the exit is reported.
        """
        process = self._process
        if process is None:
            return True
        try:
            await asyncio.to_thread(process.wait, timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def log_tail(self) -> str:
        """The end of the server's output, for error messages."""
        try:
            with open(self._log_path, "rb") as handle:
                handle.seek(0, 2)
                handle.seek(max(0, handle.tell() - _LOG_TAIL_BYTES))
                return handle.read().decode("utf-8", "replace").strip()
        except OSError:
            return ""

    async def stop(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            logger.info("stopping llama-server")
            await asyncio.to_thread(self._terminate, process)
        self._close_log()

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        """Stop the process and anything it spawned (a launcher script or venv shim has children)."""
        try:
            family = psutil.Process(process.pid).children(recursive=True)
        except psutil.Error:
            family = []
        process.terminate()
        try:
            process.wait(timeout=_STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        for child in family:
            with contextlib.suppress(psutil.Error):
                child.kill()
        # kill() returns before the OS has torn the process down; the port and the VRAM are
        # only free once it is gone.
        psutil.wait_procs(family, timeout=_STOP_GRACE_S)

    def _close_log(self) -> None:
        if self._log is not None:
            with contextlib.suppress(OSError):
                self._log.close()
            self._log = None
