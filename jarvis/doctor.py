"""``jarvis --doctor``: check the environment and say how to fix what is wrong.

Every check is a small function that returns a ``Check``; nothing here starts
the assistant, loads a model, or opens a microphone. Heavy packages are looked
up with ``importlib.util.find_spec`` instead of being imported.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from jarvis.utils.config import JarvisConfig, expand_path

OK, WARN, FAIL = "ok", "warn", "fail"

# Embedding model the Mem0 memory backend asks Ollama for (see memory/manager.py).
EMBEDDING_MODEL = "nomic-embed-text"

FindSpec = Callable[[str], Any]


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    hint: str = ""


def check_python() -> Check:
    # Python 3.11+ is enforced by the package metadata; this just says which one runs.
    return Check("python", OK, f"{platform.python_version()} at {sys.executable}")


def check_config(config_path: Path | None) -> Check:
    if config_path is None:
        return Check("config", WARN, "no config file found; using built-in defaults",
                     "copy config.yaml next to the project or pass --config PATH")
    return Check("config", OK, str(config_path.resolve()))


def check_security(config: JarvisConfig) -> Check:
    sec = config.security
    relaxed = []
    if not sec.confirm_dangerous_ops:
        relaxed.append("confirmations off")
    if not sec.command_whitelist:
        relaxed.append("command whitelist off")
    scope = ", ".join(sec.allowed_directories) or "none"
    if relaxed:
        return Check("security", WARN, f"full access ({', '.join(relaxed)}); files: {scope}",
                     "intended for a personal machine only; see docs/security.md")
    return Check("security", OK, f"standard profile; files: {scope}")


def check_packages(config: JarvisConfig, find: FindSpec = importlib.util.find_spec) -> list[Check]:
    """Optional packages each enabled feature needs."""
    def have(module: str) -> bool:
        try:
            return find(module) is not None
        except (ImportError, ValueError):
            return False

    wanted: list[tuple[str, str, str, str]] = [
        ("speech-to-text", "faster_whisper", FAIL, "pip install faster-whisper"),
        ("wake word", "openwakeword", FAIL, "pip install openwakeword"),
        ("microphone", "pyaudio", FAIL, "pip install pyaudio"),
        ("audio output", "sounddevice", FAIL, "pip install sounddevice"),
        ("text-to-speech", "piper" if config.voice.tts.engine == "piper" else "kokoro", FAIL,
         "pip install piper-tts" if config.voice.tts.engine == "piper" else "pip install kokoro"),
    ]
    if config.llm.provider == "huggingface":
        wanted += [("huggingface model", "transformers", FAIL, 'pip install -e ".[lfm]"'),
                   ("huggingface model", "torch", FAIL, 'pip install -e ".[lfm]"')]
    if config.memory.enabled:
        wanted += [("memory", "mem0", WARN, "pip install mem0ai"),
                   ("memory", "chromadb", WARN, "pip install chromadb")]
    if config.desktop.backend == "cua":
        wanted.append(("desktop control", "cua_driver", WARN, 'pip install -e ".[desktop]"'))
    wanted.append(("browser tools", "playwright", WARN, "pip install playwright"))

    checks = []
    for area, module, severity, hint in wanted:
        if have(module):
            checks.append(Check(area, OK, f"{module} installed"))
        else:
            consequence = "that feature is unavailable" if severity == WARN else "Jarvis cannot start"
            checks.append(Check(area, severity, f"{module} is not installed ({consequence})", hint))
    return checks


def check_browser_binary(find: FindSpec = importlib.util.find_spec) -> Check | None:
    """Playwright is installed but its Chromium download may be missing."""
    if find("playwright") is None:
        return None
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if configured:
        root = Path(configured)
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ms-playwright"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / "ms-playwright"
    else:
        root = Path.home() / ".cache" / "ms-playwright"
    if any(root.glob("chromium-*")):
        return Check("browser binary", OK, f"chromium found in {root}")
    return Check("browser binary", WARN, "chromium has not been downloaded",
                 "run: playwright install chromium")


async def check_ollama(config: JarvisConfig,
                       transport: httpx.AsyncBaseTransport | None = None) -> list[Check]:
    """Reachability, the chat model, and (when memory is on) the embedding model."""
    base = config.llm.base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(base_url=base, timeout=3.0, transport=transport) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            pulled = [m.get("name", "") for m in response.json().get("models", [])]
    except (httpx.HTTPError, ValueError) as exc:
        return [Check("ollama", FAIL, f"cannot reach {base} ({type(exc).__name__})",
                      "start it with: ollama serve")]

    def is_pulled(model: str) -> bool:
        return model in pulled or f"{model}:latest" in pulled

    checks = [Check("ollama", OK, f"reachable at {base}")]
    model = config.llm.model
    if is_pulled(model):
        checks.append(Check("chat model", OK, model))
    else:
        checks.append(Check("chat model", FAIL, f"{model} is not pulled", f"run: ollama pull {model}"))
    if config.memory.enabled:
        if is_pulled(EMBEDDING_MODEL):
            checks.append(Check("embedding model", OK, EMBEDDING_MODEL))
        else:
            checks.append(Check("embedding model", WARN,
                                f"{EMBEDDING_MODEL} is not pulled (memory will not work)",
                                f"run: ollama pull {EMBEDDING_MODEL}"))
    return checks


def check_audio(config: JarvisConfig, sd: Any = None) -> list[Check]:
    """Audio devices, including whether the configured microphone index is usable."""
    if sd is None:
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:  # OSError: PortAudio library missing
            return [Check("audio devices", FAIL, f"sounddevice unavailable ({exc})",
                          "pip install sounddevice")]
    try:
        devices = list(sd.query_devices())
    except Exception as exc:
        return [Check("audio devices", FAIL, f"cannot list devices ({exc})")]

    inputs = [(i, d["name"]) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
    outputs = [(i, d["name"]) for i, d in enumerate(devices) if d.get("max_output_channels", 0) > 0]
    listing = ", ".join(f"{i}={name}" for i, name in inputs[:6]) or "none"
    checks = []

    wanted = config.voice.input_device
    if wanted is None:
        if inputs:
            checks.append(Check("microphone", OK, f"system default ({len(inputs)} inputs)"))
        else:
            checks.append(Check("microphone", FAIL, "no input device found",
                                "connect a microphone, then run --doctor again"))
    elif any(i == wanted for i, _ in inputs):
        name = next(n for i, n in inputs if i == wanted)
        checks.append(Check("microphone", OK, f"voice.input_device {wanted} = {name}"))
    else:
        checks.append(Check("microphone", FAIL,
                            f"voice.input_device {wanted} is not an input device (inputs: {listing})",
                            "device indexes change when headsets reconnect; fix it in config.local.yaml"))

    out_wanted = config.voice.output_device
    if out_wanted is not None and not any(i == out_wanted for i, _ in outputs):
        checks.append(Check("speaker", FAIL, f"voice.output_device {out_wanted} is not an output device"))
    elif outputs:
        checks.append(Check("speaker", OK, "output device available"))
    else:
        checks.append(Check("speaker", FAIL, "no output device found"))
    return checks


def check_directories(config: JarvisConfig) -> list[Check]:
    checks = []
    for label, raw in (("data directory", config.system.data_dir), ("log directory", config.system.log_dir)):
        path = expand_path(raw)
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".doctor-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            checks.append(Check(label, OK, str(path)))
        except OSError as exc:
            checks.append(Check(label, FAIL, f"{path} is not writable ({exc.strerror or exc})"))
    return checks


async def run_checks(config: JarvisConfig, config_path: Path | None,
                     *, transport: httpx.AsyncBaseTransport | None = None,
                     find: FindSpec = importlib.util.find_spec, sd: Any = None) -> list[Check]:
    checks = [check_python(), check_config(config_path), check_security(config)]
    if config.llm.provider == "ollama":
        checks += await check_ollama(config, transport)
    checks += check_packages(config, find)
    browser = check_browser_binary(find)
    if browser is not None:
        checks.append(browser)
    checks += check_audio(config, sd)
    checks += check_directories(config)
    return checks


def render(checks: Iterable[Check]) -> str:
    """A fixed-width report: one line per check, hints indented under failures."""
    checks = list(checks)
    width = max((len(c.name) for c in checks), default=0)
    lines = ["Jarvis doctor", ""]
    for check in checks:
        lines.append(f"  {check.status:<4}  {check.name:<{width}}  {check.detail}")
        if check.hint and check.status != OK:
            lines.append(f"  {'':<4}  {'':<{width}}  -> {check.hint}")
    failures = sum(c.status == FAIL for c in checks)
    warnings = sum(c.status == WARN for c in checks)
    lines += ["", f"{failures} problem(s), {warnings} warning(s)"
              if failures or warnings else "everything looks good"]
    return "\n".join(lines) + "\n"


def exit_code(checks: Iterable[Check]) -> int:
    return 1 if any(c.status == FAIL for c in checks) else 0
